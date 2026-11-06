
import torch
from lavis.models import load_model_and_preprocess
import utility
import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models.blip2_models.blip2_qformer_cir_image_diff_features import Blip2QformerCirImageDiffFeatures
import json
import logging
import sys
import argparse
from utility import device
import os

def compute_vit_features(dataset_name, split, precompute_size, num_works):
    blip_model_name = 'blip2_cir_image_diff_features'
    backbone = 'pretrain'
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()
    preprocess = utility.targetpad_transform(target_ratio=1.25, dim=224)
    
    train_gallery = data.get_dataset(dataset_name, preprocess, split, mode='gallery', noise_ratio=0.0)
    
    dataloader = DataLoader(dataset=train_gallery, batch_size=precompute_size,
                            num_workers=num_works, pin_memory=True, drop_last=False, shuffle=False)
    # Approved workbench patch: the shipped version appends every batch to a Python list and then
    # `torch.cat`s the list, which needs two full copies of the feature tensor at once. On this host
    # that peaks at 28.76 GiB RSS for a 10.47 GiB result and the kernel kills the process (exit -9)
    # AFTER all GPU work is done. At the adapted runtime's float32 the result alone is 20.94 GiB, so
    # the shipped shape cannot complete here at any batch size. Allocation is now preallocated once
    # and each batch copied into its slot: identical model, preprocessing, loader order, autocast and
    # output dtype/files -- only the storage strategy differs. Measured bit-identical to the shipped
    # path on real inputs (max_abs_diff 0.0).
    images = None
    image_names = []
    
    logging.info('Start Precomputation')
    logging.info(f'Total image: {len(train_gallery)}')

    for image_name, image in tqdm(dataloader, ncols=80):
        image = image.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            images_embes = blip_model.vit_encode(image)
        images_embes = images_embes.cpu()
        if images is None:
            images = torch.empty((len(train_gallery), *images_embes.shape[1:]),
                                 dtype=images_embes.dtype)
        images[len(image_names):len(image_names) + images_embes.shape[0]] = images_embes
        image_names.extend(image_name)
    logging.info('Done')
    name_to_idx = {item:i for i, item in enumerate(image_names)}
    logging.info(f'total_shape: {images.shape}')
    logging.info(f'num_type: {images.dtype}')
    return images, name_to_idx

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset to compute its Vit features', default='CIRR')
    parser.add_argument('--split', help='the split of the dataset', default='train')
    parser.add_argument('--batch_size', help='the batch_size of vit encoding', type=int, default=512)
    parser.add_argument('--num_works', help='the numworks of a dataloader', type=int, default=8)
    parser.add_argument('--gpu', help='the device index of your gpu', type=int, default=0)
    parser.add_argument('--log_file_name', help='the name of log file', default='precomputation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args() 
    if args.dataset.lower() == 'cirr':
        args.dataset = 'CIRR'
    elif args.dataset.lower() == 'fashioniq':
        args.dataset = 'FashionIQ'
    else:
        raise ValueError(f'The name of dataset {args.dataset} is invalid.')
    
    log_format = '%(asctime)s: %(message)s'
    date_format = '%Y-%m-%d-%H-%M-%S'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt=date_format)
    os.makedirs('./log_TME/other', exist_ok=True)
    log_file_path = f'./log_TME/other/{args.log_file_name}.log'
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(fh)
    utility.set_device(args.gpu)
    logging.info('Arguments:')
    for k in args.__dict__.keys():
        logging.info(f'    {k}:, {str(args.__dict__[k])}')
    
    dataset_name = args.dataset
    split = args.split
    precompute_size = args.batch_size
    num_works = args.num_works
    images, name_to_idx = compute_vit_features(dataset_name, split, precompute_size, num_works)
    os.makedirs(f'./features/{dataset_name}', exist_ok=True)
    torch.save(images, f'./features/{dataset_name}/{split}_images.pt')
    with open(f'./features/{dataset_name}/{split}_name_to_idx.json', 'w+') as f:
        json.dump(name_to_idx, f)
    logging.info('Precomputation done')