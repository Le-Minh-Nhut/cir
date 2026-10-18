import os 
import argparse
import logging
import warnings 
import random

import numpy as np 
import torch 
from torch.autograd import Variable
from torch.utils.data import dataloader
from tqdm import tqdm

import open_clip 
import utils
import datasets
import model as img_text_model
import test
import math

from torch.cuda.amp import autocast as autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.5)

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
parser.add_argument('--tolerance_epoch', type=int, default=10)
parser.add_argument('--model_dir', default='./experiment',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--i', type=float, default=0)
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
                loss = model.compute_loss(img1, mods, img2, dataset_type)
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

def train_and_evaluate(model, optimizer, dataset_list):
    if args.dataset == 'fashion200k':
        fashion200k_testset = dataset_list.pop(-1)
    combined_dataset = torch.utils.data.ConcatDataset(dataset_list)
    sampler = DatasetSampler(dataset_list, batch_size = args.batch_size)
    trainloader = dataloader.DataLoader(combined_dataset,
                                batch_size = args.batch_size,
                                sampler = sampler,
                                num_workers=args.num_workers)


    current_best_score = [float('-inf')]
    tolerance = [0]
    dataset_names = [args.dataset]
 
    scaler = GradScaler()
    epoches = args.num_epochs

    for epoch in range(epoches):
        pop_flag = 0
        for _i, _t in enumerate(tolerance):
            tolerance[_i] = tolerance[_i] + 1
            if tolerance[_i] == args.tolerance_epoch:
                pop_flag = 1
        if pop_flag == 1:
            keep_dataset_list = [dataset_list[i] for i in range(len(tolerance)) if tolerance[i] < args.tolerance_epoch]
            keep_dataset_names = [dataset_names[i] for i in range(len(tolerance)) if tolerance[i] < args.tolerance_epoch]
            if len(keep_dataset_names) == 0:
                break
            combined_dataset = torch.utils.data.ConcatDataset(keep_dataset_list)
            sampler = DatasetSampler(keep_dataset_list, batch_size = args.batch_size)
            trainloader = dataloader.DataLoader(combined_dataset,
                                        batch_size = args.batch_size,
                                        sampler = sampler,
                                        num_workers=args.num_workers)

        if epoch !=0 and (epoch+1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div

        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        train(model, optimizer, trainloader, scaler, epoch)

        current_score = [0]
        for _id, _name in enumerate(zip(tolerance, dataset_names)):
            if _name[0] < args.tolerance_epoch:
                if _name[1] in ['dress', 'shirt', 'toptee', 'shoes']:
                    t = test.test(args, model, dataset_list[_id], _name[1])
                    logging.info(t)
                    current_score[_id] = current_score[_id] + t[1][1] + t[2][1]
                
                elif _name[1] in ['fashion200k']:
                    t = test.test_fashion200k_dataset(args, model, fashion200k_testset)
                    logging.info(t)
                    current_score[_id] = current_score[_id] + t[0][1] + t[1][1] + t[2][1]
                
                elif _name[1] in ['cirr']:
                    t = test.test_cirr_valset(args, model, dataset_list[_id])
                    logging.info(t)
                    current_score[_id] = current_score[_id] + t[0][1] + t[1][1] + t[2][1] + t[3][1] + t[4][1] + t[5][1] + t[6][1] # mean best
                
                if current_score[_id] > current_best_score[_id]:
                    current_best_score[_id] = current_score[_id]
                    tolerance[_id] = 0
                    best_json_path = os.path.join(
                        args.model_dir, "{}_{}_metrics_best.json".format(_name[1], args.i))
                    test_metrics = {}
                    for metric_name, metric_value in t:
                        test_metrics[metric_name] = metric_value

                    test_metrics['prefix_num'] = str(args.prefix_num)
                    test_metrics['head'] = str(args.head)
                    test_metrics['layer'] = str(args.layer)
                    test_metrics['norm'] = str(args.norm)
                    test_metrics['temperature'] = str(args.temperature)

                    utils.save_dict_to_json(test_metrics, best_json_path)
                    # save model
                    # if _name[1] == 'cirr':
                    torch.save(model, os.path.join(args.model_dir, "{}_{}_best_model.pt".format(args.i, _name[1])))


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

    dataset_list = load_dataset()
 
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, optimizer, dataset_list)
