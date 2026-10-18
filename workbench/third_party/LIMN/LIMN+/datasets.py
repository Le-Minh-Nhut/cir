"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import string 
import glob
import torchvision
import pickle
import pathlib
import random

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
dataset_name_dict = {
    'dress': 0,
    'shirt': 1,
    'toptee': 2,
    'shoes': 3,
    'birds': 4,
    'cirr': 5,
    'fashion200k': 6,
    'dress_aug': 7,
    'shirt_aug': 8,
    'toptee_aug': 9,
    'shoes_aug': 10,
    'birds_aug': 11,
    'cirr_aug': 12,
    'fashion200k_aug': 13
}

class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, name, transform=None):
        super().__init__()

        self.path = path
        self.name = name
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        self.dataset_type = dataset_name_dict[self.name]
        if not os.path.exists(os.path.join(self.path, '{}_train_data.json'.format(self.name))):
            self.train_data = []
            self.train_init_process()
            with open(os.path.join(self.path, '{}_train_data.json'.format(self.name)), 'w') as f:
                json.dump(self.train_data, f)
        else:
            with open(os.path.join(self.path, '{}_train_data.json'.format(self.name)), 'r') as f:
                self.train_data = json.load(f) 
        
        if not os.path.exists(os.path.join(self.path, 'test_queries_{}.pkl'.format(self.name))):
            self.test_queries, self.test_targets = self.get_test_data()
            save_obj(self.test_queries, os.path.join(self.path, 'test_queries_{}.pkl'.format(self.name)))
            save_obj(self.test_targets, os.path.join(self.path, 'test_targets_{}.pkl'.format(self.name)))
        else:
            self.test_queries = load_obj(os.path.join(self.path, 'test_queries_{}.pkl'.format(self.name)))
            self.test_targets = load_obj(os.path.join(self.path, 'test_targets_{}.pkl'.format(self.name)))
    

    def train_init_process(self):
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.name, 'train')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.name)), 'r') as f:
            correction_dict = json.load(f)
        for triplets in ref_captions:
            ref_id = triplets['candidate']
            tag_id = triplets['target']
            cap = self.concat_text(triplets['captions'], correction_dict)
            self.train_data.append({
                'target': self.name + '_' + tag_id,
                'candidate': self.name + '_' + ref_id,
                'captions': cap
            })

    def correct_text(self, text, correction_dict):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions, correction_dict):
        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
        return text

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        caption = self.train_data[idx]
        mod_str = caption['captions']
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        out['data'] = self.name
        out['dataset_type'] = self.dataset_type

        return out

    def get_img(self, image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[0](img)
        return img

    def get_test_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        img = self.transform[1](img)

        return img

    def get_all_texts(self):
        texts = []
        for caption in self.train_data:
            mod_texts = caption['captions']
            texts.append(mod_texts)
        return texts

    def get_test_data(self):
        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(self.name, 'val')), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.name, 'val')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.name)), 'r') as f:
            correction_dict = json.load(f)
        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = self.concat_text(caption['captions'], correction_dict)
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_test_img(self.name + '_' + candidate)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_test_img(self.name + '_' + target)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)

        test_targets_id = []
        for i in test_queries:
            if i['source_img_id'] not in test_targets_id:
                test_targets_id.append(i['source_img_id'])
            if i['target_img_id'] not in test_targets_id:
                test_targets_id.append(i['target_img_id'])
        test_targets = []
        for i in test_targets_id:
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_test_img(self.name + '_' + images[i])      
            test_targets.append(out)
        return test_queries, test_targets

class FashionIQ_augment(torch.utils.data.Dataset):
    def __init__(self, path, name, augment_data_path, augment_iteration, transform=None):
        super().__init__()
        self.path = path
        self.name = name 
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.split = 'train'
        self.transform = transform
        self.augment_iteration = augment_iteration
        self.dataset_type = dataset_name_dict[self.name + '_aug']

        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.name)), 'r') as f:
            self.correction_dict = json.load(f)
           
        with open(os.path.join(augment_data_path, 'augment_data_{}_{}.json'.format(self.name, augment_iteration)), 'r') as f:
            self.ref_captions = json.load(f)

    def correct_text(self, text):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([self.correction_dict.get(word) if word in self.correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions):
        text = "{}".format(self.correct_text(captions))
        return text

    def __len__(self):
        return len(self.ref_captions)

    def __getitem__(self, idx):
        caption = self.ref_captions[idx]
        mod_str = self.concat_text(caption['captions'])
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        out['dataset_type'] = self.dataset_type

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, self.name, image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform[0](img)
        return img

class Shoes(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super(Shoes, self).__init__()
        self.transform = transform
        self.path = path
        # self.split = split
        self.dataset_type = dataset_name_dict['shoes']

        with open(os.path.join(self.path, 'relative_captions_shoes.json')) as f:
            self.all_triplets = json.loads(f.read())
        
        train_image_file = 'train_im_names.txt'
        eval_image_file = 'eval_im_names.txt'
        train_image_file = open(os.path.join(self.path, train_image_file), 'r')
        train_image_names = train_image_file.readlines()
        train_image_names = [train_image_name.strip('\n') for train_image_name in train_image_names]

        eval_image_file = open(os.path.join(self.path, eval_image_file), 'r')
        eval_image_names = eval_image_file.readlines()
        eval_image_names = [eval_image_name.strip('\n') for eval_image_name in eval_image_names]

        self.imgfolder = os.listdir(self.path)
        self.imgfolder = [self.imgfolder[i] for i in range(len(self.imgfolder)) if 'womens' in self.imgfolder[i]]
        self.imgimages_all = []
        for i in range(len(self.imgfolder)):
            path = os.path.join(self.path,self.imgfolder[i])
            imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
            self.imgimages_all += imgfiles
        self.imgimages_raw = [os.path.basename(imgname) for imgname in self.imgimages_all]

        with open(os.path.join(self.path, 'correction_dict_{}.json'.format('shoes')), 'r') as f:
            self.correction_dict = json.load(f)

        if not os.path.exists(os.path.join(self.path, 'shoes_train_triplets-limn.pkl')):
            self.train_relative_pairs = []
            self.eval_relative_pairs = []
            for triplets in self.all_triplets:
                if triplets['ReferenceImageName'] in train_image_names:
                    source = self.imgimages_all[self.imgimages_raw.index(triplets['ReferenceImageName'])]
                    target = self.imgimages_all[self.imgimages_raw.index(triplets['ImageName'])]
                    mod = triplets['RelativeCaption']
                    self.train_relative_pairs.append({
                        'source': source,
                        'target': target,
                        'mod': mod.strip()
                    })
                elif triplets['ReferenceImageName'] in eval_image_names:
                    source = self.imgimages_all[self.imgimages_raw.index(triplets['ReferenceImageName'])]
                    target = self.imgimages_all[self.imgimages_raw.index(triplets['ImageName'])]
                    mod = triplets['RelativeCaption']
                    self.eval_relative_pairs.append({
                        'source': source,
                        'target': target,
                        'mod': mod.strip()
                    })
            self.test_queries = self.get_test_queries()
            self.test_targets = self.get_test_targets()

            save_obj(self.train_relative_pairs, os.path.join(self.path, 'shoes_train_triplets-limn.pkl'))
            save_obj(self.eval_relative_pairs, os.path.join(self.path, 'shoes_eval_triplets-limn.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'shoes_test_queries-limn.pkl'))
            save_obj(self.test_targets, os.path.join(self.path, 'shoes_test_targets.pkl'))
        else:
            self.train_relative_pairs = load_obj(os.path.join(self.path, 'shoes_train_triplets-limn.pkl'))
            self.eval_relative_pairs = load_obj(os.path.join(self.path, 'shoes_eval_triplets-limn.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'shoes_test_queries-limn.pkl'))
            self.test_targets = load_obj(os.path.join(self.path, 'shoes_test_targets-limn.pkl'))


    def correct_text(self, text):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([self.correction_dict.get(word) if word in self.correction_dict else word for word in tokens])

        return text

    def __len__(self):
        return len(self.train_relative_pairs)

    def __getitem__(self, idx):

        caption = self.train_relative_pairs[idx]
        out = {}
        out['source_img_data'] = self.get_img(caption['source'])
        out['target_img_data'] = self.get_img(caption['target'])
        out['mod'] = {'str': self.correct_text(caption['mod'])}
        out['dataset_type'] = self.dataset_type

        return out

    def get_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[0](img)
        return img

    def get_test_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        img = self.transform[1](img)
        return img
    
    def get_all_texts(self):
        texts = []
        for caption in self.train_relative_pairs:
            mod_texts = self.correct_text(caption['mod'])
            texts.append(mod_texts)
        return texts

    def get_test_queries(self):
        test_queries = []
        for idx in range(len(self.eval_relative_pairs)):
            caption = self.eval_relative_pairs[idx]
            mod_str = self.correct_text(caption['mod'])
            candidate = caption['source']
            target = caption['target']

            out = {}
            out['source_img_id'] = self.imgimages_all.index(candidate)
            out['source_img_data'] = self.get_test_img(candidate)
            out['target_img_id'] = self.imgimages_all.index(target)
            out['target_img_data'] = self.get_test_img(target)
            out['mod'] = {'str': mod_str}
            test_queries.append(out)
        return test_queries
    
    def get_test_targets(self):
        text_file = open(os.path.join(self.path, 'eval_im_names.txt'),'r')
        imgnames = text_file.readlines()
        imgnames = [imgname.strip('\n') for imgname in imgnames] # img list
        test_target = []
        for i in imgnames:
            out = {}
            out['target_img_id'] = self.imgimages_raw.index(i)
            out['target_img_data'] = self.get_test_img(self.imgimages_all[self.imgimages_raw.index(i)])
            test_target.append(out)
        return test_target
    


class Shoes_augment(torch.utils.data.Dataset):
    def __init__(self, path, augment_data_path, augment_iteration, transform=None):
        super(Shoes_augment, self).__init__()

        self.transform = transform
        self.path = path
        self.dataset_type = dataset_name_dict['shoes' + '_aug']

        with open(os.path.join(self.path, 'correction_dict_{}.json'.format('shoes')), 'r') as f:
            self.correction_dict = json.load(f)

        self.imgfolder = os.listdir(self.path)
        self.imgfolder = [self.imgfolder[i] for i in range(len(self.imgfolder)) if 'womens' in self.imgfolder[i]]

        self.imgimages_all = []
        for i in range(len(self.imgfolder)):
            path = os.path.join(self.path,self.imgfolder[i])
            imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
            self.imgimages_all += imgfiles
        
        self.imgimages_raw = [os.path.basename(imgname) for imgname in self.imgimages_all]
        with open(os.path.join(augment_data_path, 'augment_data_shoes_{}.json'.format(augment_iteration)), 'r') as f:
            self.relative_pairs = json.load(f)

    def correct_text(self, text):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([self.correction_dict.get(word) if word in self.correction_dict else word for word in tokens])

        return text

    def __len__(self):
        return len(self.relative_pairs)

    def __getitem__(self, idx):

        caption = self.relative_pairs[idx]
        out = {}
        ref_dir = self.imgimages_all[self.imgimages_raw.index(caption['candidate'])]
        tag_dir = self.imgimages_all[self.imgimages_raw.index(caption['target'])]
        out['source_img_data'] = self.get_img(ref_dir)
        out['target_img_data'] = self.get_img(tag_dir)
        out['mod'] = {'str': self.correct_text(caption['captions'])}
        out['dataset_type'] = self.dataset_type

        return out

    def get_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform[0](img)
        return img




class Birds(torch.utils.data.Dataset):
    def __init__(self, path, gallery_all=False, transform=None):
        super(Birds, self).__init__()

        self.path = path
        self.image_dir = self.path + 'bird'

        self.transform = transform
        self.gallery_all = gallery_all
        self.dataset_type = dataset_name_dict['birds']

        if not os.path.exists(os.path.join(self.path, 'birds_train_triplets-limn.pkl')):
            with open(os.path.join(self.path, "cap.{}.{}.json".format('birds', 'train')), 'r') as f:
                self.train_ref_captions = json.load(f)
            with open(os.path.join(self.path, "split.{}.{}.json".format('birds', 'train')), 'r') as f:
                self.train_images = json.load(f)

            with open(os.path.join(self.path, "cap.{}.{}.json".format('birds', 'test-merge')), 'r') as f:
                self.test_merge_ref_captions = json.load(f)
            with open(os.path.join(self.path, "split.{}.{}.json".format('birds', 'test-merge')), 'r') as f:
                self.test_merge_images = json.load(f)

            self.test_queries = self.get_test_queries(self.test_merge_ref_captions, self.test_merge_images)
            self.test_targets = self.get_test_targets(self.test_merge_ref_captions, self.test_merge_images)
            save_obj(self.test_queries, os.path.join(self.path, 'birds_test_merge_queries-limn.pkl'))
            save_obj(self.test_targets, os.path.join(self.path, 'birds_test_merge_targets-limn.pkl'))
        else:
            self.test_queries = load_obj(os.path.join(self.path, 'birds_test_merge_queries-limn.pkl'))
            self.test_targets = load_obj(os.path.join(self.path, 'birds_test_merge_targets-limn.pkl'))
        
    
    def concat_text(self, captions):
        text = "<BOS> {} <EOS>".format(captions)
        return text
    
    def __len__(self):
        return len(self.train_ref_captions)
        
    
    def __getitem__(self, idx):
        caption = self.train_ref_captions[idx]
        mod_str = self.concat_text(caption['captions'])
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        out['dataset_type'] = self.dataset_type
        # if self.split == 'train':
        #     if caption in self.augment_captions:
        #         out['weight'] = torch.ones(1) * 0.8
        #     else:
        #         out['weight'] = torch.ones(1)

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')


        img = self.transform[0](img)
        return img

    def get_test_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[1](img)
        return img

    def get_all_texts(self):
        texts = []
        with open(os.path.join(self.path, "cap.{}.{}.json".format('birds', 'train')), 'r') as f:
            train_captions = json.load(f)
        for caption in train_captions:
            mod_texts = caption['captions']
            texts.append(mod_texts)
        return texts

    def get_test_queries(self, ref_captions, images):       # query
        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = self.concat_text(caption['captions'])
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_test_img(candidate)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_test_img(target)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)
        
        return test_queries

    def get_test_targets(self, ref_captions, images):       # 所有的image
        if self.gallery_all:
            test_targets = []
            for idx in range(len(images)):
                target = images[idx]
                out = {}
                out['target_img_id'] = idx
                out['target_img_data'] = self.get_test_img(target)
                test_targets.append(out)
        else:
            test_targets_id = []
            queries = self.get_test_queries(ref_captions, images)
            for i in queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
        
            test_targets = []
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_test_img(images[i])
                test_targets.append(out)   
        return test_targets

class CIRR(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path 
        self.caption_dir = self.path + 'captions/captions/'
        self.split_dir = self.path + 'captions/image_splits/'
        self.transform = transform
        self.dataset_type = dataset_name_dict['cirr']
        # train data
        with open(os.path.join(self.caption_dir, "cap.rc2.train.json"), 'r') as f:
            self.cirr_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys()) 

        # val data
        if not os.path.exists(os.path.join(self.path, 'cirr_val_queries.pkl')):
            self.val_queries, self.val_targets = self.get_val_queries()
            save_obj(self.val_queries, os.path.join(self.path, 'cirr_val_queries.pkl'))
            save_obj(self.val_targets, os.path.join(self.path, 'cirr_val_targets.pkl'))
        else:
            self.val_queries = load_obj(os.path.join(self.path, 'cirr_val_queries.pkl'))
            self.val_targets = load_obj(os.path.join(self.path, 'cirr_val_targets.pkl'))
        # test data
        if not os.path.exists(os.path.join(self.path, 'cirr_test_queries.pkl')):
            self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()
            save_obj(self.test_name_list, os.path.join(self.path, 'cirr_test_name_list.pkl'))
            save_obj(self.test_img_data, os.path.join(self.path, 'cirr_test_img_data.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'cirr_test_queries.pkl'))
        else:
            self.test_name_list = load_obj(os.path.join(self.path, 'cirr_test_name_list.pkl'))
            self.test_img_data = load_obj(os.path.join(self.path, 'cirr_test_img_data.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'cirr_test_queries.pkl'))

    def __len__(self):
        return len(self.cirr_data)

    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['caption']
        target_name = caption['target_hard']
        
        out = {}
        out['source_img_data'] = self.get_img(os.path.join('images', pathlib.Path(self.train_image_path[reference_name])))
        out['target_img_data'] = self.get_img(os.path.join('images', pathlib.Path(self.train_image_path[target_name])))
        out['mod'] = {'str':mod_str}
        out['dataset_type'] = self.dataset_type
        return out
    
    def get_img(self, img_path):
        img_path = os.path.join(self.path, img_path)
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[0](img)
        return img

    def get_test_img(self, img_path):
        img_path = os.path.join(self.path, img_path)
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[1](img)
        return img

    def get_val_queries(self):
        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())

        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['caption']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_test_img(os.path.join(pathlib.Path(val_image_path[reference_name])))
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_test_img(os.path.join(pathlib.Path(val_image_path[target_name])))
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids

            test_queries.append(out)

        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_test_img(os.path.join(pathlib.Path(val_image_path[name])))
            test_targets.append(out)

        return test_queries, test_targets

    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_test_img(os.path.join(pathlib.Path(test_image_path[caption['reference']])))
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_test_img(os.path.join(pathlib.Path(test_image_path[name])))
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries

class Fashion200k(torch.utils.data.Dataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'
        self.dataset_type = dataset_name_dict['fashion200k']

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []
        self.test_queries = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                                         '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ', filename)
            with open(label_path + '/' + filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

    def get_loader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_different_word(self, source_caption, target_caption):
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
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
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
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                #if not caption2id.has_key(c):
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                #if not parent2children_captions.has_key(p):
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
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
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)
   
    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out['source_img_id'] = idx
        out['source_img_data'] = self.get_img(idx)
        out['source_caption'] = self.imgs[idx]['captions'][0]
        out['target_img_id'] = target_idx
        out['target_img_data'] = self.get_img(target_idx)
        out['target_caption'] = self.imgs[target_idx]['captions'][0]
        out['mod'] = {'str': mod_str}
        out['dataset_type'] = self.dataset_type
        return out

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img