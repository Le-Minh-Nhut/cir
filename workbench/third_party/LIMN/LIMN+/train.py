import os 
import argparse
import logging
import warnings 
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
from tqdm import tqdm

import open_clip 
import utils
import datasets
import model as img_text_model
import test
import math
from itertools import product 

from torch.cuda.amp import autocast as autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'fashioniq', help = "data set type")
parser.add_argument('--name', default = 'dress', help = "data set type")
parser.add_argument('--fashioniq_path', default = '[your path]/fashion_iq_data/')
parser.add_argument('--shoes_path', default = '[your path]/shoes_data/')
parser.add_argument('--fashion200k_path', default = '[your path]/fashion200k_data')
parser.add_argument('--cirr_path', default = '[your path]/CIRR/')

parser.add_argument('--augment_iteration', type=int, default = 1) # 1 2
parser.add_argument('--augment_data', type=str, default= './pseudo_data')

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.5)

parser.add_argument('--p_origin', type=float, default=0.3) 
parser.add_argument('--p_aug', type=float, default=0.6) 
parser.add_argument('--interval', type=int, default=4) 

parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-4) 
parser.add_argument('--clip_lr', type=float, default=5e-6) 

parser.add_argument('--lr_decay', type=int, default=5)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=10)  
parser.add_argument('--temperature', type=float, default=0.1) 

parser.add_argument('--head', type=int, default=8)
parser.add_argument('--prefix_num', type=int, default=10)
parser.add_argument('--layer', type=int, default=2)

parser.add_argument('--peer_ranking', type=float, default=1.0)
parser.add_argument('--norm', type=float, default=0.5)
parser.add_argument('--tolerance_epoch', type=int, default=8)

 
parser.add_argument('--model_dir', default='./experiment',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--i', type=float, default=0)
parser.add_argument('--node', type=str, default='')
args = parser.parse_args()


def load_dataset():
    """Loads the input datasets."""
    
    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-L-14', pretrained=os.path.join('[CLIP path]/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'))
    if args.dataset == 'dress':
        print('Reading dress')
        dress_dataset = datasets.FashionIQ(path = args.fashioniq_path, name = 'dress', transform = [preprocess_train, preprocess_val])
        return [dress_dataset]
    elif args.dataset == 'shirt':
        print('Reading shirt')
        shirt_dataset = datasets.FashionIQ(path = args.fashioniq_path, name = 'shirt', transform = [preprocess_train, preprocess_val])
        return [shirt_dataset]
    elif args.dataset == 'toptee':
        print('Reading toptee')
        toptee_dataset = datasets.FashionIQ(path = args.fashioniq_path, name = 'toptee', transform = [preprocess_train, preprocess_val])
        return [toptee_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets.Shoes(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]
    elif args.dataset == 'cirr':
        print('Reading cirr')
        cirr_dataset = datasets.CIRR(path = args.cirr_path, transform = [preprocess_train, preprocess_val])
        return [cirr_dataset]
    elif args.dataset == 'fashion200k':
        print('Reading fashion200k')
        fashion200k_dataset = datasets.Fashion200k(path = args.fashion200k_path, split = 'train', transform = preprocess_train)
        fashion200k_testset = datasets.Fashion200k(path = args.fashion200k_path, split = 'test', transform = preprocess_val)
        return [fashion200k_dataset, fashion200k_testset]
    
    
def load_aug_dataset():
    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-L-14', pretrained=os.path.join('[CLIP path]/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'))
    if args.dataset == 'dress':
        dress_dataset = datasets.FashionIQ_augment(path = args.fashioniq_path, name = 'dress',
                                                    augment_data_path = args.augment_data,
                                                    augment_iteration = str(args.augment_iteration),
                                                    transform = [preprocess_train, preprocess_val])
        return [dress_dataset]
    elif args.dataset == 'shirt':
        shirt_dataset = datasets.FashionIQ_augment(path = args.fashioniq_path, name = 'shirt',
                                                    augment_data_path = args.augment_data,
                                                    augment_iteration = str(args.augment_iteration),
                                                    transform = [preprocess_train, preprocess_val])
        return [shirt_dataset]
    elif args.dataset == 'toptee':
        toptee_dataset = datasets.FashionIQ_augment(path = args.fashioniq_path, name = 'toptee',
                                                    augment_data_path = args.augment_data,
                                                    augment_iteration = str(args.augment_iteration),
                                                    transform = [preprocess_train, preprocess_val])
        return [toptee_dataset]
    elif args.dataset == 'shoes':
        shoes_dataset = datasets.Shoes_augment(path = args.shoes_path,
                                                    augment_data_path = args.augment_data,
                                                    augment_iteration = str(args.augment_iteration),
                                                    transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]

def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    model = img_text_model.LIMN(args.prefix_num, args.head, args.layer, 768, args.temperature)
    model.cuda()

    params = list(model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if any(nd in n for nd in ['clip'])], 'lr': args.clip_lr},
        {'params': [p for n, p in params if not any(nd in n for nd in ['clip'])], 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay = args.weight_decay)

    return model, optimizer


pre_model = torch.load('../checkpoint/{}_{}_best_model.pt'.format(str(args.augment_iteration-1), args.dataset)).cuda()
def train(model, optimizer, dataloader, scaler, epoch):
    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:

        for i, data in enumerate(dataloader):
            img1 = data['source_img_data'].cuda()
            img2 = data['target_img_data'].cuda()
            mods = data['mod']['str']   
            dataset_type = torch.tensor(data['dataset_type']).cuda()
            optimizer.zero_grad()
            with autocast():
                loss = model.compute_loss(img1, mods, img2, dataset_type, pre_model, args.p_origin, args.p_aug)
                total_loss = loss['leader_ranking'] + args.norm * loss['norm'] + args.peer_ranking * loss['peer_ranking']

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['total_loss'] = total_loss.item()
                summ.append(summary_batch)
            loss_avg.update(total_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

def compute_rampup_weight(epoch, lambd=1.0, alpha=15):
    if epoch > alpha:
        return lambd
    else:
        return lambd * math.exp(-5 * (1 - epoch / lambd) ** 2)

class DatasetSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.dataset_lengths)
        self.droplast_dataset_lengths = [length_ - length_ % self.batch_size for length_ in self.dataset_lengths]

    def __iter__(self):
        order = []
        for dataset_idx, dataset_length in enumerate(self.dataset_lengths):
            indices_ = list(range(dataset_length))
            random.shuffle(indices_)
            indices_= indices_[:dataset_length - dataset_length % self.batch_size]
            indices_ = [i + sum(self.dataset_lengths[:dataset_idx]) for i in indices_]
            indices_ = [indices_[i:i+self.batch_size] for i in range(0, self.droplast_dataset_lengths[dataset_idx], self.batch_size)]
            order.extend(indices_)
        random.shuffle(order)
        flatten_order = [item for sublist in order for item in sublist]
        return iter(flatten_order)
    
    def __len__(self):
        return sum(self.droplast_dataset_lengths)

def train_and_evaluate(model, optimizer, dataset_ori, dataset_aug):

    combined_dataset_ori = torch.utils.data.ConcatDataset(dataset_ori)
    sampler_ori = DatasetSampler(dataset_ori, batch_size = args.batch_size)
    trainloader_ori = dataloader.DataLoader(combined_dataset_ori,
                                batch_size = args.batch_size,
                                sampler = sampler_ori,
                                num_workers=args.num_workers)

    combined_dataset_aug = torch.utils.data.ConcatDataset(dataset_aug)
    sampler_aug = DatasetSampler(dataset_aug, batch_size = args.batch_size)
    trainloader_aug = dataloader.DataLoader(combined_dataset_aug,
                                batch_size = args.batch_size,
                                sampler = sampler_aug,
                                num_workers=args.num_workers)


    current_best_score = [float('-inf')]
    current_best_score = float('-inf')

    dataset_names = [args.dataset]

    scaler = GradScaler()
    epoches = args.num_epochs
    tolerance = 0
    for epoch in range(epoches):
        tolerance += 1
        if tolerance == 10:
            break


        if epoch !=0 and (epoch+1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div

        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        if (epoch+1) % args.interval == 0:
            train(model, optimizer, trainloader_aug, scaler, epoch)
        else:
            train(model, optimizer, trainloader_ori, scaler, epoch)
        current_score = 0
        if args.dataset in ['dress', 'shirt', 'toptee', 'birds', 'shoes']:
            t = test.test(args, model, dataset_ori[0], args.dataset)
            logging.info(t)
            current_score = current_score + t[1][1] + t[2][1]

            if current_score > current_best_score:
                current_best_score = current_score
                tolerance = 0
                best_json_path = os.path.join(
                    args.model_dir, "{}_{}_metrics_best.json".format(args.dataset, str(args.augment_iteration)))
                test_metrics = {}
                for metric_name, metric_value in t:
                    test_metrics[metric_name] = metric_value

                test_metrics['prefix_num'] = str(args.prefix_num)
                test_metrics['head'] = str(args.head)
                test_metrics['layer'] = str(args.layer)
                test_metrics['norm'] = str(args.norm)
                test_metrics['temperature'] = str(args.temperature)
                test_metrics['interval'] = str(args.interval)
                test_metrics['p_origin'] = str(args.p_origin)
                test_metrics['p_aug'] = str(args.p_aug)

                utils.save_dict_to_json(test_metrics, best_json_path)
                # save model
                # if args.dataset == 'cirr':
                torch.save(model, os.path.join(args.model_dir, "{}_{}_best_model.pt".format(str(args.augment_iteration), args.dataset)))



if __name__ == '__main__':

    # Load the parameters from json file

    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))

    
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')
    # fetch dataloaders

    ori_dataset_list = load_dataset()
    aug_dataset_list = load_aug_dataset()
 
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, optimizer, ori_dataset_list, aug_dataset_list)
