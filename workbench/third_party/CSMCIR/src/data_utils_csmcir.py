import json
from pathlib import Path
from typing import Union, List, Dict, Literal

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch 
import random
from torchvision import transforms
import os
import glob
import string 
base_path = Path(__file__).absolute().parents[1].absolute()
import random
def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# 自定义 RandomMask 类
class RandomMask:
    def __init__(self, probability=0.3, min_size=0.1,max_size=0.3):
        """
        :param probability: 随机 mask 的概率
        :param mask_size: mask 区域的大小比例，相对于图像尺寸
        """
        self.probability = probability
        self.mask_size = (min_size, max_size)  # 例如 (0.1, 0.3)
        

    def __call__(self, img):
        if random.random() < self.probability:
            width, height = img.size
            mask_width = int(width * random.uniform(*self.mask_size))
            mask_height = int(height * random.uniform(*self.mask_size))
            # 随机选择 mask 的起始位置
            x = random.randint(0, width - mask_width)
            y = random.randint(0, height - mask_height)

            # 生成随机颜色遮挡
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            mask = PIL.Image.new('RGB', (mask_width, mask_height), color=(r, g, b))
            img.paste(mask, (x, y))

        return img

#-------------数据增强----------------

def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        # transforms.RandomHorizontalFlip(p=0.3),  # 添加随机反转结构
        # RandomMask(probability=0.2),  # 添加随机 mask 步骤
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class RandomPerspective(object):
    def __init__(self, distortion_scale=0.5, prob=0.5):
        self.distortion_scale = distortion_scale
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            w, h = img.size
            points = []
            for _ in range(4):
                x = random.randint(0, int(w * self.distortion_scale))
                y = random.randint(0, int(h * self.distortion_scale))
                points.append((x, y))
            return img.transform(img.size, PIL.Image.PERSPECTIVE, 
                               data=points, resample=PIL.Image.BICUBIC)
        return img


class GaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))


import cv2
import numpy as np
import PIL

class FashionIQDataset_imageedit(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic','classic_reference']:
            raise ValueError("mode should be in ['relative', 'classic','classic_reference']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # Load keywords like the first code
        self.key_words = {}
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'keywords_in_mods_{dress_type}.json') as f:
                self.key_words.update(json.load(f))

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        self.target_image_captions: dict = {}
        for dress_type in dress_types:
            with open(base_path / 'COT_ours2'/'bert_captions'/'fashioniq'/ f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))
            #target image captions    
            with open(base_path/'COT_ours2'/'fashioniq' /f'{dress_type}_cot_{split}.json') as f:
                captions_chunk = json.load(f)  # 假设每个文件返回一个字典
                self.target_image_captions.update(captions_chunk)  # 合并字典

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def draw_text(self, img, point, text, drawType="custom"):
        """
        Draw text on image
        """
        fontScale = 0.7
        thickness = 5
        text_thickness = 2
        bg_color = (255, 255, 255)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        
        if drawType == "custom":
            text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
            text_loc = (point[0], point[1] + text_size[1])
            cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                          (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
            cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                        (255, 0, 0), text_thickness, 8)
        elif drawType == "simple":
            cv2.putText(img, str(text), point, fontFace, 0.5, (255, 0, 0))
        return img

    def draw_text_line(self, img, point, text_line: str, drawType="custom"):
        """
        Draw multiple lines of text on image
        """
        fontScale = 0.7
        thickness = 5
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        text_line = text_line.split(", ")
        text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
        
        for i, text in enumerate(text_line):
            if text:
                draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
                img = self.draw_text(img, draw_point, text, drawType)
        return img

    def get_written_img(self, reference_name, target_name):
        """
        Get image with keywords written on it (following first code's approach)
        """
        image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
        
        # Get keywords like the first code does
        key_word_key = f"{reference_name}_{target_name}"
        if key_word_key in self.key_words:
            key_word = self.key_words[key_word_key][-1]  # Get the last keyword like first code
        else:
            key_word = ""  # fallback if key not found
        
        # Read and resize image
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (512, 512))
        
        # Draw keywords on image (not captions)
        written_img = self.draw_text_line(img, (15, 15), key_word)
        
        # Convert back to PIL Image and preprocess
        written_img = PIL.Image.fromarray(cv2.cvtColor(written_img, cv2.COLOR_BGR2RGB))
        written_img = self.preprocess(written_img)
        
        return written_img, str(image_path)

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions=self.triplets[index]['captions']
                if len(image_captions)==1:
                    image_captions.append(image_captions[0])
                
                reference_name = self.triplets[index]['candidate']
                image_captions_cot=self.target_image_captions[reference_name]['Final_Caption']
                target_name = self.triplets[index]['target']
                target_image_caption=self.target_image_captions[target_name]['Final_Caption']

                if self.split == 'train':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    # Get written image with keywords (not captions)
                    written_reference_img, _ = self.get_written_img(reference_name, target_name)
                    
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    opt_caption=self.triplets[index]['optimized_caption']
                    
                    return written_reference_img, target_image, image_captions, opt_caption, target_image_caption

                elif self.split == 'val':
                    return reference_name, target_name, image_captions, image_captions_cot, target_image_caption

                elif self.split == 'test':
                    # Get written image with keywords (replaces reference_image)
                    written_reference_img, _ = self.get_written_img(reference_name, target_name)
                    
                    return reference_name, written_reference_img, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image
                
            elif self.mode == 'classic_reference':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                image_captions=self.triplets[index]['captions']
                return image_name, image, image_captions
            else:
                raise ValueError("mode should be in ['relative', 'classic',classic_reference]")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        elif self.mode == 'classic_reference':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic','classic_reference']")


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic','classic_reference']:
            raise ValueError("mode should be in ['relative', 'classic','classic_reference']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        self.target_image_captions: dict = {}
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset'/'captions'/ f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))
            #target image captions    
            with open(base_path/'fashionIQ_dataset'/'qwen_captions' /f'{dress_type}_cot_{split}.json') as f:
                captions_chunk = json.load(f)  # 假设每个文件返回一个字典
                self.target_image_captions.update(captions_chunk)  # 合并字典

            # with open(base_path/'fashionIQ_dataset' / 'captions' /f'reference_image_caption_{dress_type}_{split}.json') as f:
            #     captions_chunk = json.load(f)  # 假设每个文件返回一个字典
            #     self.target_image_captions.update(captions_chunk)  # 合并字典

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")



    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions=self.triplets[index]['captions']# optimized_caption
                if len(image_captions)==1:
                    image_captions.append(image_captions[0])
                
                reference_name = self.triplets[index]['candidate']
                # try:
                image_captions_cot=self.target_image_captions[reference_name]['Final_Caption']#Final_Caption
                # image_captions_cot=self.target_image_captions[reference_name]['Visual_Attributes']#Final_Caption
                # except:
                # image_captions_cot=self.target_image_captions[reference_name]['Core_Essence']#Final_Caption
                # image_captions_cot=self.target_image_captions[reference_name]#Final_Caption
                target_name = self.triplets[index]['target']
                # try:
                target_image_caption=self.target_image_captions[target_name]['Final_Caption']#Final_Caption
                #target_image_caption=self.target_image_captions[target_name]['Visual_Attributes']#Final_Caption
                # except:
                # target_image_caption=self.target_image_captions[target_name]['Core_Essence']#Final_Caption
                # target_image_caption=self.target_image_captions[target_name]

                if self.split == 'train':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    opt_caption=self.triplets[index]['optimized_caption']
                    return reference_image, target_image, image_captions, opt_caption,target_image_caption

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions,image_captions_cot,target_image_caption

                elif self.split == 'test':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image
                
            elif self.mode == 'classic_reference':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                image_captions=self.triplets[index]['captions']
                return image_name, image, image_captions
            else:
                raise ValueError("mode should be in ['relative', 'classic',classic_reference]")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        elif self.mode == 'classic_reference':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic','classic_reference']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / 'cirr_dataset'  / 'cirr' / 'captions'/ f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)
        #qwen2.5 image captions 
        self.target_image_captions: dict = {}
        with open(base_path/'cirr_dataset'  / 'qwen_captions'/f'target_image_caption_{split}.json') as f:
            captions_chunk = json.load(f)  # 假设每个文件返回一个字典
            self.target_image_captions.update(captions_chunk)  # 合并字典
        # with open(base_path/'cirr_dataset' / 'cirr'/'captions' /f'reference_image_caption_{split}_retry.json') as f:
        #     captions_chunk = json.load(f)  # 假设每个文件返回一个字典
        #     self.target_image_captions.update(captions_chunk)  # 合并字典


        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                modified_captions=self.triplets[index]['caption']
                target_image_captions=self.target_image_captions
                reference_name = self.triplets[index]['reference']
                if self.split == 'train' or self.split == 'val':
                    target_name=self.triplets[index]['target_hard']
                    target_image_caption = target_image_captions[target_name]['Final_Caption'] #Final_Caption
                    image_captions_cot=target_image_captions[reference_name]['Final_Caption'] #Final_Caption
                    # target_image_caption = target_image_captions[target_name]
                    # image_captions_cot=target_image_captions[reference_name]

                if self.split == 'train':
                    reference_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, modified_captions,image_captions_cot,target_image_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, modified_captions,image_captions_cot, target_image_caption,group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    image_captions_cot=target_image_captions[reference_name]['Final_Caption'] #Final_Caption
                    # image_captions_cot=target_image_captions[reference_name]
                    return pair_id, reference_name, modified_captions, image_captions_cot, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class ShoesDataset(Dataset):
    """
    Shoes dataset class which manages Shoes data.
    Similar to FashionIQ, this dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yields tuples made of (image_name, image)
        - In 'relative' mode the dataset yields tuples made of:
            - (reference_image, target_image, mod_text, target_image_caption) when split == train
            - (reference_name, target_name, mod_text, target_image_caption) when split == val
            - (reference_name, reference_image, mod_text) when split == test
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic', 'classic_reference']
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.split = split

        if mode not in ['relative', 'classic', 'classic_reference']:
            raise ValueError("mode should be in ['relative', 'classic', 'classic_reference']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        self.preprocess = preprocess
        self.base_path = base_path / 'shoes_dataset'  # 假设有一个基础路径变量

        # 加载图像名称列表
        if split == 'train':
            with open(self.base_path / 'train_im_names.txt', 'r') as f:
                self.image_names = [name.strip() for name in f.readlines()]
        else:  # val 或 test
            with open(self.base_path / 'eval_im_names.txt', 'r') as f:
                self.image_names = [name.strip() for name in f.readlines()]

        # 加载所有图像路径
        self.all_images_paths = []
        self.all_images_names = []
        img_folders = [folder for folder in os.listdir(self.base_path) if 'womens' in folder]
        for folder in img_folders:
            path = os.path.join(self.base_path, folder)
            img_files = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
            self.all_images_paths.extend(img_files)
        self.all_images_names = [os.path.basename(img_path) for img_path in self.all_images_paths]

        # 加载修正字典 - 用于文本纠正
        with open(self.base_path / 'correction_dict_shoes.json', 'r') as f:
            self.correction_dict = json.load(f)

        # 加载相对描述信息
        with open(self.base_path / 'relative_captions_shoes.json', 'r') as f:
            all_triplets = json.loads(f.read())
        
        # 加载图像描述信息 - 对应于target_image_captions
        # with open(self.base_path / 'captions_shoes_new.json', 'r') as f:
        with open(self.base_path / 'image_captions_shoes.json', 'r') as f:
        #with open(self.base_path / f'shoes_cot_{split}.json', 'r') as f:
            self.image_captions = json.load(f)

        # 根据split区分triplets
        self.triplets = []
        for triplet in all_triplets:
            reference_name = triplet['ReferenceImageName']
            target_name = triplet['ImageName']
            
            # 检查是否在当前拆分中
            if split == 'train' and reference_name in self.image_names:
                self.triplets.append({
                    'candidate': reference_name,
                    'target': target_name,
                    'captions': [triplet['RelativeCaption']]
                })
            elif split in ['val', 'test'] and reference_name in self.image_names:
                self.triplets.append({
                    'candidate': reference_name,
                    'target': target_name,
                    'captions': [triplet['RelativeCaption']]
                })
        
        print(f"Shoes {split} dataset in {mode} mode initialized with {len(self.triplets)} triplets.")

    def correct_text(self, text):
        """Apply text correction using the correction dictionary."""
        trans = str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([self.correction_dict.get(word, word) for word in tokens])
        return text

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                # 在相对模式下，返回参考图像、目标图像和描述
                triplet = self.triplets[index]
                reference_name = triplet['candidate']
                target_name = triplet['target']
                mod_text = self.correct_text(triplet['captions'][0])
                
                # 获取目标图像的描述
                #target_image_caption = self.image_captions[target_name]['Final_Caption']
                target_image_caption = self.image_captions[target_name]
                
                if self.split == 'train':
                    # 为训练集，加载参考图像和目标图像
                    reference_image_path = self.get_image_path(reference_name)
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    
                    target_image_path = self.get_image_path(target_name)
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert('RGB'))
                    
                    return reference_image, target_image, [mod_text], target_image_caption
                
                elif self.split == 'val':
                    # 为验证集，只返回图像名称和描述
                    return reference_name, target_name, [mod_text], target_image_caption,target_image_caption
                
                elif self.split == 'test':
                    # 为测试集，加载参考图像
                    reference_image_path = self.get_image_path(reference_name)
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    
                    return reference_name, reference_image, [mod_text]
            
            elif self.mode == 'classic':
                # 在经典模式下，只返回单个图像
                image_name = self.image_names[index]
                image_path = self.get_image_path(image_name)
                image = self.preprocess(PIL.Image.open(image_path).convert('RGB'))
                
                return image_name, image
            
            elif self.mode == 'classic_reference':
                # 在经典参考模式下，返回图像和描述
                image_name = self.image_names[index]
                image_path = self.get_image_path(image_name)
                image = self.preprocess(PIL.Image.open(image_path).convert('RGB'))
                
                # 找到与该图像相关的描述（如果有）
                captions = []
                for triplet in self.triplets:
                    if triplet['candidate'] == image_name:
                        captions.extend(triplet['captions'])
                        break
                
                return image_name, image, captions
            
            else:
                raise ValueError("mode should be in ['relative', 'classic', 'classic_reference']")
                
        except Exception as e:
            print(f"Exception in __getitem__: {e}")
            # 返回一个默认值或者重新尝试获取另一个索引
            if index + 1 < len(self):
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(0)

    def get_image_path(self, image_name):
        """Helper function to get the full path of an image by its name."""
        if image_name in self.all_images_names:
            idx = self.all_images_names.index(image_name)
            return self.all_images_paths[idx]
        else:
            raise FileNotFoundError(f"Image {image_name} not found")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode in ['classic', 'classic_reference']:
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic', 'classic_reference']")




class Fashion200kDataset(Dataset):
    """
    Fashion200k dataset class which manage Fashion200k data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, mod_string) when split == train
            - (reference_name, target_name, mod_string) when split == val
            - (reference_name, reference_image, mod_string) when split == test
    """

    def __init__(self, split: str, mode: str, preprocess: callable, path: str):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, mod_string) when split == train
                - (reference_name, target_name, mod_string) when split == val
                - (reference_name, reference_image, mod_string) when split == test
        :param preprocess: function which preprocesses the image
        :param path: base path to the dataset
        """
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.img_path = path + '/'

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        # 读取标签文件
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile, join
        label_files = [f for f in listdir(label_path) if isfile(join(label_path, f))]
        label_files = [f for f in label_files if split in f]

        # 从标签文件读取图像信息
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print(f'读取 {filename}')
            with open(label_path + '/' + filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print(f'Fashion200k: {len(self.imgs)} 张图像')

        # 为训练或测试生成查询
        if split == 'train':
            self.caption_index_init_()
            # 存储查询和目标对
            self.query_target_pairs = []
            for i in range(len(self.imgs)):
                if self.imgs[i]['modifiable']:
                    idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(i)
                    self.query_target_pairs.append({
                        'source_img_id': idx,
                        'target_img_id': target_idx,
                        'mod_str': mod_str
                    })
        else:
            self.generate_test_queries_()

        print(f"Fashion200k {split} 数据集在 {mode} 模式下初始化完成")

    def get_different_word(self, source_caption, target_caption):
        """获取两个描述之间的差异词"""
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        """生成测试查询"""
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        
        # 读取测试查询文件
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries.append({
                'source_img_id': idx,
                'target_img_id': target_idx,
                'mod_str': mod_str
            })

    def caption_index_init_(self):
        """初始化描述索引，用于后续生成训练查询-目标示例"""
        # 为描述建立索引
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        
        self.caption2imgids = caption2imgids
        print(f'{len(caption2imgids)} 个唯一描述')

        # 父描述比其子描述少一个词
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        
        self.parent2children_captions = parent2children_captions

        # 为每个图像标识父描述
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print(f'可修改的图像: {num_modifiable_imgs}')

    def caption_index_sample_(self, idx):
        """基于索引采样查询-目标对"""
        # 确保索引对应的图像是可修改的
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # 查找随机目标图像（相同父描述）
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # 查找查询和目标之间的词差异（不在父描述中）
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        
        return idx, target_idx, source_word, target_word, mod_str

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                if self.split == 'train':
                    # 使用预先生成的查询-目标对
                    pair = self.query_target_pairs[index]
                    source_idx = pair['source_img_id']
                    target_idx = pair['target_img_id']
                    mod_str = pair['mod_str']
                    
                    # 获取源图像和目标图像
                    source_image_path = self.img_path + self.imgs[source_idx]['file_path']
                    source_image = self.preprocess(PIL.Image.open(source_image_path).convert('RGB'))
                    
                    target_image_path = self.img_path + self.imgs[target_idx]['file_path']
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert('RGB'))
                    
                    return source_image, target_image, mod_str
                
                elif self.split == 'val':
                    # 返回源图像名称和目标图像名称
                    query = self.test_queries[index]
                    source_idx = query['source_img_id']
                    target_idx = query['target_img_id']
                    mod_str = query['mod_str']
                    
                    source_name = self.imgs[source_idx]['file_path']
                    target_name = self.imgs[target_idx]['file_path']
                    
                    return source_name, target_name, mod_str
                
                elif self.split == 'test':
                    # 返回源图像名称和源图像
                    query = self.test_queries[index]
                    source_idx = query['source_img_id']
                    mod_str = query['mod_str']
                    
                    source_name = self.imgs[source_idx]['file_path']
                    source_image_path = self.img_path + source_name
                    source_image = self.preprocess(PIL.Image.open(source_image_path).convert('RGB'))
                    
                    return source_name, source_image, mod_str
            
            elif self.mode == 'classic':
                # 返回图像名称和图像
                img = self.imgs[index]
                image_name = img['file_path']
                image_path = self.img_path + image_name
                image = self.preprocess(PIL.Image.open(image_path).convert('RGB'))
                
                return image_name, image
            
            else:
                raise ValueError("mode should be in ['relative', 'classic']")
                
        except Exception as e:
            print(f"异常: {e}")
            # 发生异常时返回一个默认项
            if self.mode == 'relative':
                if self.split == 'train':
                    return None, None, None
                elif self.split == 'val':
                    return None, None, None
                else:  # test
                    return None, None, None
            else:  # classic
                return None, None

    def __len__(self):
        if self.mode == 'relative':
            if self.split == 'train':
                return len(self.query_target_pairs)
            else:  # val or test
                return len(self.test_queries)
        elif self.mode == 'classic':
            return len(self.imgs)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def get_loader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        """获取数据加载器"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)



class LaSCoDataset(Dataset):
    """
    LaSCo数据集类，用于管理LaSCo数据。
    该数据集可以在'relative'或'classic'模式下使用：
        - 在'classic'模式下，数据集生成由(image_name, image)组成的元组
        - 在'relative'模式下，数据集生成的元组由以下组成：
            - 当split == train时: (reference_image, target_image, query_text, reference_caption, target_caption)
            - 当split == val时: (reference_name, target_name, query_text, reference_caption, target_caption)
            - 当split == test时: (reference_name, reference_image, query_text)
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: 数据集分割，应为['test', 'train', 'val']中的一个
        :param mode: 数据集模式，应为['relative', 'classic', 'classic_reference']中的一个：
            - 在'classic'模式下，数据集生成由(image_name, image)组成的元组
            - 在'relative'模式下，数据集生成的元组由以下组成：
                - 当split == train时: (reference_image, target_image, query_text, reference_caption, target_caption)
                - 当split == val时: (reference_name, target_name, query_text, reference_caption, target_caption)
                - 当split == test时: (reference_name, reference_image, query_text)
        :param preprocess: 预处理图像的函数
        :param base_path: LaSCo数据集的基础路径
        """
        self.mode = mode
        self.split = split
        self.base_path = os.path.join(base_path, 'LaSCo_dataset')
        print( self.base_path)
        self.preprocess = preprocess

        if mode not in ['relative', 'classic', 'classic_reference']:
            raise ValueError("mode应为['relative', 'classic', 'classic_reference']中的一个")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split应为['test', 'train', 'val']中的一个")

        # 加载数据集的图像ID映射
        self.image_corpus = {}
        corpus_file = os.path.join(self.base_path, f"lasco_{split}_corpus.json")
        if split == 'train':
            with open(corpus_file, 'r') as f:
                self.image_corpus = json.load(f)
            self.image_names = list(self.image_corpus.keys())
        else:  # val/test
            with open(corpus_file, 'r') as f:
                items = json.load(f)
                for item in items:
                    self.image_corpus[str(item['id'])] = item['path']
            self.image_names = [str(item['id']) for item in items]

        # 加载三元组数据
        self.triplets = []
        triplets_file = os.path.join(self.base_path, f"lasco_{split}.json")
        with open(triplets_file, 'r') as f:
            self.triplets = json.load(f)

        # 加载图像描述
        self.image_captions = {}
        if split == 'train':
            caption_file = os.path.join(self.base_path, "annotations/captions_train2014.json")
            print(caption_file )
        else:
            caption_file = os.path.join(self.base_path, "annotations/captions_val2014.json")
        
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
            annotations = caption_data['annotations']
            
            # 组织图像描述，以图像ID为键，所有相关描述为值
            for ann in annotations:
                image_id = str(ann['image_id'])
                if image_id not in self.image_captions:
                    self.image_captions[image_id] = []
                self.image_captions[image_id].append(ann['caption'])

        print(f"LaSCo {split} 数据集以 {mode} 模式初始化")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                triplet = self.triplets[index]
                query_text = triplet['query-text']
                reference_id, reference_path = triplet['query-image']
                target_id, target_path = triplet['target-image']
                reference_id_str = str(reference_id)
                target_id_str = str(target_id)
                
                # # 获取图像描述
                # if self.split == 'val':
                reference_caption = self.image_captions.get(reference_id_str, [""])[0]
                target_caption = self.image_captions.get(target_id_str, [""])[0]
                # else:
                #     reference_caption_list = self.image_captions.get(reference_id_str, [""])
                #     target_caption_list = self.image_captions.get(target_id_str, [""])

                #     if reference_caption_list:
                #         reference_caption = random.choice(reference_caption_list)
                #     else:
                #         reference_caption = ""

                #     if target_caption_list:
                #         target_caption = random.choice(target_caption_list)
                #     else:
                #         target_caption = ""



                if self.split == 'train':
                    reference_image_path = os.path.join(self.base_path, reference_path)
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    target_image_path = os.path.join(self.base_path, target_path)
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    return reference_image, target_image, query_text, reference_caption, target_caption

                elif self.split == 'val':
                    return reference_id_str, target_id_str, query_text, reference_caption, target_caption

                elif self.split == 'test':
                    reference_image_path = os.path.join(self.base_path, reference_path)
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_id_str, reference_image, query_text

            elif self.mode == 'classic':
                image_id = self.image_names[index]
                image_path = os.path.join(self.base_path, self.image_corpus[image_id][4:])
                image = self.preprocess(PIL.Image.open(image_path))
                return image_id, image
                
            elif self.mode == 'classic_reference':
                image_id = self.image_names[index]
                image_path = os.path.join(self.base_path, self.image_corpus[image_id][4:])
                image = self.preprocess(PIL.Image.open(image_path))
                captions = self.image_captions.get(image_id, [""])
                return image_id, image, captions
            else:
                raise ValueError("mode应为['relative', 'classic', 'classic_reference']中的一个")
        except Exception as e:
            print(f"异常: {e}")
            # 发生异常时返回默认值，以防止数据加载崩溃
            if self.mode == 'relative':
                if self.split == 'train':
                    # 返回空图像和空文本
                    return self[0]  # 尝试返回第一个元素
                else:
                    return "0", "0", "", "", ""
            else:
                return "0", None

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode in ['classic', 'classic_reference']:
            return len(self.image_names)
        else:
            raise ValueError("mode应为['relative', 'classic', 'classic_reference']中的一个")