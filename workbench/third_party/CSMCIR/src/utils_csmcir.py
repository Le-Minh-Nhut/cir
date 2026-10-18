import multiprocessing
import random
from pathlib import Path
from typing import Union, Tuple, List
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils_csmcir import CIRRDataset, CIRCODataset,ShoesDataset,LaSCoDataset,FashionIQDataset 
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import json
base_path = Path(__file__).absolute().parents[1].absolute()
def extract_index_features(dataset: Union[CIRRDataset, FashionIQDataset], clip_model) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names


def extract_index_blip_features(dataset: Union[CIRRDataset, FashionIQDataset,CIRCODataset], blip_model, save_memory=False) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = []
    index_features_raw = []
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features, image_embeds_frozen = blip_model.extract_target_features(images,  mode="mean")
            if save_memory:
                image_features = image_features.cpu()
                image_embeds_frozen = image_embeds_frozen.cpu()
            index_features.append(image_features)
            index_features_raw.append(image_embeds_frozen)
            index_names.extend(names)
    
    index_features = torch.vstack(index_features)
    index_features_raw = torch.vstack(index_features_raw)

    return (index_features, index_features_raw), index_names



def extract_index_blip_caption_features(dataset: Union[CIRRDataset, FashionIQDataset, CIRCODataset, LaSCoDataset], blip_model, save_memory=False) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ, CIRR, CIRCO or LaSCo index features
    :param dataset: FashionIQ, CIRR, CIRCO or LaSCo dataset in 'classic' mode
    :param blip_model: BLIP model
    :param save_memory: whether to save memory
    :return: a tuple of features and a list of images
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = []
    index_features_raw = []
    index_names = []

    target_image_captions: dict = {}
    target_image_captions_buchong: dict = {}
    if isinstance(dataset, FashionIQDataset):
        dress_types=['dress', 'shirt', 'toptee']
        for dress_type in dress_types:
            file_path = os.path.join(base_path, 'COT_ours2', 'fashioniq',f'{dress_type}_cot_val.json')
            with open(file_path) as f:
                captions_chunk = json.load(f)  # 假设每个文件返回一个字典
                target_image_captions.update(captions_chunk)  # 合并字典
    elif isinstance(dataset, CIRRDataset):
        with open(os.path.join(base_path,'COT_ours2','cirr','target_image_caption_val.json')) as f:
            captions_chunk = json.load(f)  # 假设每个文件返回一个字典
            target_image_captions.update(captions_chunk)  # 合并字典
    elif isinstance(dataset, ShoesDataset):
        with open(os.path.join(base_path,'shoes_dataset','shoes_cot_val.json')) as f:
            print('base path:',base_path)
            captions_chunk = json.load(f)  # 假设每个文件返回一个字典
            target_image_captions.update(captions_chunk)  # 合并字典 
        # with open(os.path.join(base_path,'shoes_datasetimage_captions_shoes.json')) as f:
        #     captions_chunk = json.load(f)  # 假设每个文件返回一个字典
        #     target_image_captions_buchong.update(captions_chunk)  # 合并字典 

    elif isinstance(dataset, LaSCoDataset):
        caption_file = os.path.join(dataset.base_path, "annotations/captions_val2014.json")
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
            for ann in caption_data['annotations']:
                image_id = str(ann['image_id'])
                if image_id not in target_image_captions:
                    target_image_captions[image_id] = {"Final_Caption": ann['caption']}

    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
        for names, images in tqdm(classic_val_loader):
            images = images.to(device, non_blocking=True)
            target_text=[]
            for name in names:
                try:
                    target_text.append(target_image_captions[name]['Final_Caption'])
                except:
                    # target_text.append(target_image_captions_buchong[name])
                    print(name)
                    continue

            with torch.no_grad():
                image_features, image_embeds_frozen = blip_model.extract_target_caption_features(images, target_text,mode="mean")
                if save_memory:
                    image_features = image_features.cpu()
                    image_embeds_frozen = image_embeds_frozen.cpu()
                index_features.append(image_features)
                index_features_raw.append(image_embeds_frozen)
                index_names.extend(names)
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
        for names, images in tqdm(classic_val_loader):
            images = images.to(device, non_blocking=True)
            target_text=[]
            for name in names:
                try:
                    # target_text.append(target_image_captions[name]['Final_Caption'])
                    target_text.append(target_image_captions[name]['Final_Caption'])
                except:
                    target_text.append(target_image_captions[name]['Final_Caption'])
                    continue

            with torch.no_grad():
                image_features, image_embeds_frozen = blip_model.extract_target_caption_features(images, target_text,mode="mean")
                if save_memory:
                    image_features = image_features.cpu()
                    image_embeds_frozen = image_embeds_frozen.cpu()
                index_features.append(image_features)
                index_features_raw.append(image_embeds_frozen)
                index_names.extend(names)
    elif isinstance(dataset, LaSCoDataset):
        print(f"extracting LaSCo {dataset.split} index features")
        for names, images in tqdm(classic_val_loader):
            images = images.to(device, non_blocking=True)
            target_text = []
            for name in names:
                try:
                    target_text.append(target_image_captions[name]['Final_Caption'])
                except:
                    print(f"Missing caption for image {name}")
                    # 如果没有找到对应的描述，使用一个空字符串
                    target_text.append("")
                    continue

            with torch.no_grad():
                image_features, image_embeds_frozen = blip_model.extract_target_caption_features(images, target_text, mode="mean")
                if save_memory:
                    image_features = image_features.cpu()
                    image_embeds_frozen = image_embeds_frozen.cpu()
                index_features.append(image_features)
                index_features_raw.append(image_embeds_frozen)
                index_names.extend(names)
    elif isinstance(dataset, ShoesDataset):
        print(f"extracting shoes index features")
        for names, images in tqdm(classic_val_loader):
            images = images.to(device, non_blocking=True)
            target_text=[]
            for name in names:
                # try:
                    target_text.append(target_image_captions[name]['Final_Caption'])
                # except:
                #     target_text.append(target_image_captions_buchong[name])
            with torch.no_grad():
                image_features, image_embeds_frozen = blip_model.extract_target_caption_features(images, target_text,mode="mean")
                if save_memory:
                    image_features = image_features.cpu()
                    image_embeds_frozen = image_embeds_frozen.cpu()
                index_features.append(image_features)
                index_features_raw.append(image_embeds_frozen)
                index_names.extend(names)

    index_features = torch.vstack(index_features)
    index_features_raw = torch.vstack(index_features_raw)

    return (index_features, index_features_raw), index_names

def extract_index_blip_caption_reference_features(dataset: Union[CIRRDataset, FashionIQDataset,CIRCODataset], blip_model, txt_processors,save_memory=False) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = []
    index_features_raw = []
    index_names = []

    dress_types=['dress', 'shirt', 'toptee']
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images ,captions in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        captions = generate_randomized_fiq_caption(flattened_captions)
        captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.no_grad():
            image_features, image_embeds_frozen = blip_model.extract_target_caption_features(images, captions,mode="mean")
            if save_memory:
                image_features = image_features.cpu()
                image_embeds_frozen = image_embeds_frozen.cpu()
            index_features.append(image_features)
            index_features_raw.append(image_embeds_frozen)
            index_names.extend(names)
    
    index_features = torch.vstack(index_features)
    index_features_raw = torch.vstack(index_features_raw)

    return (index_features, index_features_raw), index_names


def extract_index_fuse_features(dataset: Union[CIRRDataset, FashionIQDataset], fuse_model) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = []
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features = fuse_model.retrieval_transformer.encode_image(images)
            index_features.append(image_features)
            index_names.extend(names)
    
    index_features = torch.vstack(index_features)

    return (index_features), index_names


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def generate_randomized_fiq_caption_cot(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions)):
        captions.append(f"{flattened_captions[i ].strip('.?, ').capitalize()}")
    return captions



def apply_random_mask(caption: str) -> str:
    """
    Apply random masking to a caption by replacing parts of it with [MASK] token.
    
    Masking strategies:
    1. Mask a random word
    2. Mask a random span of 2-3 consecutive words
    3. Mask the latter part of the caption
    4. Mask the beginning of the caption
    
    :param caption: The caption to mask
    :return: The masked caption
    """
    words = caption.split()
    
    # If caption is too short, don't mask
    if len(words) <= 2:
        return caption
    
    mask_strategy = random.randint(1, 4)
    
    if mask_strategy == 1:
        # Mask a random word
        idx_to_mask = random.randint(0, len(words) - 1)
        words[idx_to_mask] = "[MASK]"
    
    elif mask_strategy == 2 and len(words) >= 4:
        # Mask a random span of 2-3 consecutive words
        span_length = min(random.randint(2, 3), len(words) - 1)
        start_idx = random.randint(0, len(words) - span_length)
        for i in range(start_idx, start_idx + span_length):
            words[i] = "[MASK]"
    
    elif mask_strategy == 3:
        # Mask the latter part of the caption
        split_point = len(words) // 2
        for i in range(split_point, len(words)):
            words[i] = "[MASK]"
    
    elif mask_strategy == 4:
        # Mask the beginning of the caption
        split_point = len(words) // 2
        for i in range(0, split_point):
            words[i] = "[MASK]"
    
    return " ".join(words)



def combine_and_generate_randomized_fiq_caption(flattened_captions,original_caption):
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.3:
            captions.append(
                f"{original_caption[i//2].capitalize()}, but {flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.3 < random_num < 0.6:
            captions.append(
                f"{original_caption[i//2].capitalize()}, but {flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}")
        elif 0.6 < random_num < 0.8:
            captions.append(f"{original_caption[i//2].capitalize()}, but {flattened_captions[i].strip('.?, ')}")
        else:
            captions.append(f"{original_caption[i//2].capitalize()}, but {flattened_captions[i + 1].strip('.?, ')}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)







def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )

def update_train_running_results_dict(train_running_results: dict, loss_dict: dict, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    for key in loss_dict.keys():
        if key not in train_running_results:
            train_running_results[key] = 0
        train_running_results[key] += loss_dict[key].to('cpu', non_blocking=True).detach().item() * images_in_batch

    train_running_results["images_in_epoch"] += images_in_batch

def set_train_bar_description_dict(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """ 
    images_in_epoch = train_running_results['images_in_epoch']
    bar_content = ''
    for key in train_running_results:
        if key != 'images_in_epoch':
            bar_content += f'{key}: {train_running_results[key] / images_in_epoch:.3f}, '
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"{bar_content}"
    )   

def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))
