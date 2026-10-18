# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR
import os

from data_utils_csmcir import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils_csmcir import collate_fn, update_train_running_results,update_train_running_results_dict, set_train_bar_description_dict,set_train_bar_description, extract_index_blip_features,extract_index_blip_caption_features, \
    save_model, generate_randomized_fiq_caption,combine_and_generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip_csmcir import compute_cirr_val_metrics, compute_fiq_val_metrics

def calculate_blip_parameters(model, verbose=True):
    """
    准确计算BLIP模型的参数元素总数（以M为单位），区分不同组件的参数
    
    Args:
        model: BLIP模型实例
        verbose: 是否打印详细信息
    
    Returns:
        tuple: (总参数M, 可训练参数M, 冻结参数M, 各组件参数字典)
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    # 用于统计各组件参数
    component_stats = {
        'visual_encoder': 0,
        'text_encoder': 0,
        'qformer': 0,
        'projection': 0,
        'custom_layers': 0,
        'other': 0
    }
    
    component_trainable = {key: 0 for key in component_stats.keys()}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # 统计可训练和冻结参数
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count
        
        # 根据参数名称分类到不同组件
        name_lower = name.lower()
        if 'visual_encoder' in name_lower or 'vision' in name_lower:
            component_stats['visual_encoder'] += param_count
            if param.requires_grad:
                component_trainable['visual_encoder'] += param_count
        elif 'text_encoder' in name_lower or 'bert' in name_lower:
            component_stats['text_encoder'] += param_count
            if param.requires_grad:
                component_trainable['text_encoder'] += param_count
        elif 'qformer' in name_lower or 'query' in name_lower:
            component_stats['qformer'] += param_count
            if param.requires_grad:
                component_trainable['qformer'] += param_count
        elif 'proj' in name_lower or 'fc' in name_lower:
            component_stats['projection'] += param_count
            if param.requires_grad:
                component_trainable['projection'] += param_count
        elif any(x in name_lower for x in ['custom', 'combiner', 'adapter', 'lora']):
            component_stats['custom_layers'] += param_count
            if param.requires_grad:
                component_trainable['custom_layers'] += param_count
        else:
            component_stats['other'] += param_count
            if param.requires_grad:
                component_trainable['other'] += param_count
    
    # 转换为M单位
    total_params_m = total_params / 1_000_000
    trainable_params_m = trainable_params / 1_000_000
    frozen_params_m = frozen_params / 1_000_000
    
    component_stats_m = {k: v / 1_000_000 for k, v in component_stats.items()}
    component_trainable_m = {k: v / 1_000_000 for k, v in component_trainable.items()}
    
    if verbose:
        print("=" * 80)
        print("模型参数统计")
        print("=" * 80)
        print(f"总参数量: {total_params_m:.2f}M ({total_params:,})")
        print(f"  - 可训练参数: {trainable_params_m:.2f}M ({trainable_params:,}) [{trainable_params/total_params*100:.2f}%]")
        print(f"  - 冻结参数: {frozen_params_m:.2f}M ({frozen_params:,}) [{frozen_params/total_params*100:.2f}%]")
        print("-" * 80)
        print("各组件参数分布:")
        for component, params_m in component_stats_m.items():
            if params_m > 0:
                trainable_m = component_trainable_m[component]
                trainable_pct = (trainable_m / params_m * 100) if params_m > 0 else 0
                print(f"  {component:20s}: {params_m:8.2f}M (可训练: {trainable_m:8.2f}M [{trainable_pct:5.1f}%])")
        print("=" * 80)
    
    return total_params_m, trainable_params_m, frozen_params_m, component_stats_m, component_trainable_m



def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, 
                      **kwargs):
    """
    Fine-tune CLIP on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning leanring rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned CLIP model
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best CLIP model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"save-memory-in: {save_memory}")
    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    print("\n模型加载完成，统计参数信息:")
    total_m, trainable_m, frozen_m, comp_stats, comp_trainable = calculate_blip_parameters(blip_model, verbose=True)

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
        #   'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay':0.05}])
        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    # scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()


    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions,opt_captions,target_caption) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            # flattened_captions = [sentence for batch_sentences in captions for sentence in batch_sentences]
            #处理target caption
            target_captions=[txt_processors["eval"](caption) for caption in target_caption]

            opt_captions=[txt_processors["eval"](caption) for caption in opt_captions]

            captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions,"opt_text":opt_captions, "image_captions":target_captions})
                loss = 0.
                for key in loss_dict.keys():

                    loss += loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()

            # 添加这一行来修复FP16梯度问题
            # for param in blip_model.parameters():
            #     if param.grad is not None and param.grad.dtype == torch.float16:
            #         param.grad = param.grad.float()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                        idx_to_dress_mapping):
             
                index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model, save_memory)
                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                                    index_features, index_names, txt_processors, save_memory)
                
                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })

            print(json.dumps(results_dict, indent=4))
          
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)



def clip_finetune_cirr(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                       **kwargs):
    """
    Fine-tune CLIP on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param blip_model_name: BLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """
    rtc_weights = kwargs['loss_rtc']
    align_weights = kwargs['loss_align']
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_cirr_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    print("\n模型加载完成，统计参数信息:")
    total_m, trainable_m, frozen_m, comp_stats, comp_trainable = calculate_blip_parameters(blip_model, verbose=True)


    # clip_model.eval().float()
    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs

    # Define the train dataset and the combining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)

    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    # debug 1 
    # val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
    # # 
    # results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
    #                                     val_index_names, txt_processors)
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions,captions_cot,target_captions) in enumerate(train_bar):
            # print(scheduler.optimizer.param_groups[0]['lr'])
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            target_captions=[txt_processors["eval"](caption) for caption in target_captions]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions,"text_cot":captions_cot,"image_captions":target_captions})
                loss = 0.
                for key in loss_dict.keys():
                    # if key != 'loss_itc':
                    #     loss += kwargs[key] * loss_dict[key]
                    # else:
                    loss += loss_dict[key]
            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)


        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
                # extract target image features
            val_index_features, val_index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model)
            # 
            results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                                val_index_names, txt_processors)
            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

            results_dict = {
                'group_recall_at1': group_recall_at1,
                'group_recall_at2': group_recall_at2,
                'group_recall_at3': group_recall_at3,
                'recall_at1': recall_at1,
                'recall_at5': recall_at5,
                'recall_at10': recall_at10,
                'recall_at50': recall_at50,
                'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                'arithmetic_mean': mean(results),
                'harmonic_mean': harmonic_mean(results),
                'geometric_mean': geometric_mean(results)
            }
            print(json.dumps(results_dict, indent=4))
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # if save_training and epoch > 18:
            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('tuned_clip_arithmetic', epoch, blip_model, training_path)



def clip_finetune_shoes(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, 
                      **kwargs):
    """
    Fine-tune CLIP on the Shoes dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param blip_model_name: BLIP model you want to use
    :param backbone: backbone model
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned model
    :param save_best: when True save only the weights of the best model wrt the average_recall metric
    :param save_memory: when True use memory efficient approach during validation
    :param kwargs: additional arguments
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_shoes_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"save-memory-in: {save_memory}")
    
    # 保存所有超参数到文件
    training_hyper_params = {
        'num_epochs': num_epochs,
        'blip_model_name': blip_model_name,
        'backbone': backbone,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'validation_frequency': validation_frequency,
        'transform': transform,
        'save_training': save_training,
        'save_best': save_best,
        'save_memory': save_memory,
        **kwargs
    }
    best_rmean=0
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    # 加载BLIP模型和预处理器
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    # 设置预处理流程
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # 定义验证数据集
    relative_val_dataset = ShoesDataset('val', 'relative', preprocess)
    classic_val_dataset = ShoesDataset('val', 'classic', preprocess)

    # 定义训练数据集和数据加载器
    relative_train_dataset = ShoesDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset, 
        batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4), 
        pin_memory=False, 
        collate_fn=collate_fn,
        drop_last=True, 
        shuffle=True
    )

    # 定义优化器、损失函数和梯度缩放器
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 
          'lr': learning_rate,
          'betas': (0.9, 0.98), 
          'eps': 1e-7, 
          'weight_decay': 0.05}]
    )
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        pct_start=1.5/num_epochs, 
        div_factor=100., 
        steps_per_epoch=len(relative_train_loader), 
        epochs=num_epochs
    )
    
    scaler = torch.cuda.amp.GradScaler()

    # 当save_best为True时，初始化最佳结果为零
    if save_best:
        best_avg_recall = 0

    # 定义用于CSV日志记录的数据帧
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # 开始训练循环
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        
        for idx, (reference_images, target_images, captions, target_caption) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # 处理修饰文本 - 在Shoes数据集中，通常只有一个修饰描述
            # 因此不需要像FashionIQ那样随机化
            captions = [txt_processors["eval"](caption) for caption in captions[0]]
            # 处理target caption
            target_captions = [txt_processors["eval"](caption) for caption in target_caption]
            
            
            # 训练模式
            blip_model.train()
            
            # 提取特征，计算损失
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({
                    "image": reference_images, 
                    "target": target_images, 
                    "text_input": captions,
                    "image_captions": target_captions
                })
                
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # 反向传播和更新权重
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # 更新训练结果
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # 记录损失
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])
        
        # 训练CSV日志记录
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # 验证阶段
        if epoch % validation_frequency == 0:
            blip_model.eval()
            
            # 提取索引特征
            index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model, save_memory)
            
            # 计算验证指标
            recall_at1, recall_at10, recall_at50, recall_mean = compute_shoes_val_metrics(
                relative_val_dataset, 
                blip_model,
                index_features, 
                index_names, 
                txt_processors, 
                save_memory
            )
            
            torch.cuda.empty_cache()

            # 记录结果
            results_dict = {
                'shoes_recall_at1': recall_at1,
                'shoes_recall_at10': recall_at10,
                'shoes_recall_at50': recall_at50,
                'shoes_recall_mean': recall_mean
            }

            print(json.dumps(results_dict, indent=4))
          
            # 验证CSV日志记录
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # 保存模型
            if save_training:
                if not save_best:
                    save_model(f'tuned_clip_{epoch}', epoch, blip_model, training_path)
                elif results_dict['shoes_recall_mean'] > best_rmean:
                    best_rmean = results_dict['shoes_recall_mean']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)



def clip_finetune_lasco(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, 
                       **kwargs):
    """
    在LaSCo数据集上微调CLIP模型，使用图像-文本元素级求和作为组合函数
    :param num_epochs: 训练轮数
    :param blip_model_name: 要使用的BLIP模型名称
    :param backbone: 使用的主干网络
    :param learning_rate: 微调学习率
    :param batch_size: 批量大小
    :param validation_frequency: 验证频率（以轮为单位）
    :param transform: 要使用的预处理变换，应为['clip', 'squarepad', 'targetpad']之一。
                当使用targetpad时，还需要提供`target_ratio`参数。
    :param save_training: 当为True时，保存微调后的CLIP模型权重
    :param save_best: 当为True时，仅保存在平均召回率指标上表现最佳的CLIP模型权重
    :param save_memory: 当为True时启用内存节省模式
    :param kwargs: 如果使用`targetpad`变换，应提供`target_ratio`作为kwarg，还可以提供`num_workers`等参数
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_lasco_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"save-memory-in: {save_memory}")
    
    
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    # 加载模型和预处理器
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    # 设置预处理转换
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # 定义验证数据集
    relative_val_dataset = LaSCoDataset('val', 'relative', preprocess)
    classic_val_dataset = LaSCoDataset('val', 'classic', preprocess)

    # 定义训练数据集
    relative_train_dataset = LaSCoDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                    num_workers=kwargs.get('num_workers', 4), pin_memory=False, 
                                    collate_fn=collate_fn,
                                    drop_last=True, shuffle=True)

    # 定义优化器、损失函数和梯度缩放器
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., 
                         steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    # 当save_best == True时，初始化最佳结果为零
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # 开始训练循环
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, query_texts, reference_captions, target_captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # 处理查询文本和图像描述
            processed_query_texts = [txt_processors["eval"](query_text) for query_text in query_texts]
            processed_reference_captions = [txt_processors["eval"](caption) for caption in reference_captions]
            processed_target_captions = [txt_processors["eval"](caption) for caption in target_captions]
            
            blip_model.train()
            # 提取特征，计算logits和损失
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({
                    "image": reference_images, 
                    "target": target_images, 
                    "text_input": processed_query_texts,
                    "text_cot": processed_reference_captions, 
                    "image_captions": processed_target_captions
                })
                
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # 反向传播并更新权重
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch'])

        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # 定期验证
        if epoch % validation_frequency == 0:
            blip_model.eval()
            
            # 提取索引特征
            index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model, save_memory)
            
            # 计算验证指标
            recalls = compute_lasco_val_metrics(relative_val_dataset, blip_model,
                                             index_features, index_names, txt_processors, save_memory)


            # 记录结果
            results_dict = {
                'recall_at1': recalls[0],
                'recall_at5': recalls[1],
                'recall_at10': recalls[2],
                'recall_at50': recalls[3],
                'recall_at500': recalls[4],
                'average_recall': sum(recalls) / len(recalls)
            }

            print(json.dumps(results_dict, indent=4))
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)
            # 保存验证结果
            with open(training_path / f"validation_results_epoch_{epoch}.json", 'w+') as file:
                json.dump(results_dict, file, indent=4)

            # 保存模型
            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)
                elif not save_best:
                    save_model(f'tuned_clip_epoch_{epoch}', epoch, blip_model, training_path)





def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='fashionIQ' ,help="should be 'CIRR', 'fashionIQ', 'shoes' or 'Lasco'")
    parser.add_argument("--data-path", type=str, default="./fashionIQ_dataset")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", default=10, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt_csmcir", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
    parser.add_argument("--loss-align", default=0.8, type=float)
    parser.add_argument("--loss-target-reference-align", default=0.8, type=float)
    parser.add_argument("--loss-rtc", default=0.4, type=float)
    parser.add_argument("--loss-itm", default=1, type=float)
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',default='True',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',default='True',
                        help="Save only the best model during training")
    parser.add_argument("--save-memory", dest="save_memory", action='store_true',default='True',
                        help="Save only the best model during training")

    args = parser.parse_args()
    # if args.dataset.lower() not in ['fashioniq', 'cirr']:
    #     raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")
    print(f"save-memory: {args.save_memory}")
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "loss_rtc": args.loss_rtc,
        "loss_align": args.loss_align,
        "loss_target_reference_align": args.loss_target_reference_align,
        "loss_itm": args.loss_itm,
        "save_memory": args.save_memory
    }
    # set_seed(912)
    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        clip_finetune_fiq(**training_hyper_params)
    elif args.dataset.lower() == 'shoes':
        clip_finetune_shoes(**training_hyper_params)
    elif args.dataset.lower() == 'lasco':
        clip_finetune_lasco(**training_hyper_params)


