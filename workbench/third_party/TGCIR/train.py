import os 
import sys
import argparse
import logging
import warnings 
import time 
import itertools
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
import torchvision
from tqdm import tqdm
# from tensorboardX import SummaryWriter

import clip 
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

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'fashioniq', help = "data set type")
parser.add_argument('--fashioniq_path', default = ...)
parser.add_argument('--shoes_path', default = ...)
parser.add_argument('--cirr_path', default = ...)

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-4) 
parser.add_argument('--clip_lr', type=float, default=1e-5) 
parser.add_argument('--img_encoder', type=str, default='ViT-B/16')
parser.add_argument('--lr_decay', type=int, default=5)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=10) 
parser.add_argument('--feature_dim', type=int, default=512)

parser.add_argument('--lambda_', type=float, default=1.0) 
parser.add_argument('--eta_', type=float, default=1.0) 
parser.add_argument('--mu_', type=float, default=0.1)
parser.add_argument('--nu_', type=float, default=10)
parser.add_argument('--kappa_', type=float, default=0.5)
parser.add_argument('--tau_', type=float, default=0.1)
parser.add_argument('--P', type=int, default=4)
parser.add_argument('--Q', type=int, default=8)
 

parser.add_argument('--model_dir', default='./experiment',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--node', type=str, default='')
args = parser.parse_args()

# if args.local_rank != -1:
#     torch.cuda.set_device(args.local_rank)
#     device=torch.device("cuda", args.local_rank)
#     torch.distributed.init_process_group(backend="nccl", init_method='env://')


clip, preprocess = clip.load(args.img_encoder, device='cuda', jit=False)
def load_dataset():
    """Loads the input datasets."""

    print('Reading dataset ', args.dataset)
    img_transform = preprocess

    if args.dataset == 'fashioniq':
        trainset = datasets.FashionIQ(
            path = args.fashioniq_path,
            transform=img_transform)
    elif args.dataset == 'shoes':
        trainset = datasets.Shoes(
            path = args.shoes_path,
            transform=img_transform)
    elif args.dataset == 'cirr':
        trainset = datasets.CIRR(
            path = args.cirr_path,
            transform = img_transform,
            case_look=False
        )
   
    else:
        print('Invalid dataset', args.dataset)
        sys.exit()

    print('trainset size:', len(trainset))

    return trainset

def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    model = img_text_model.VCG_CIR(img_encoder=args.img_encoder, hidden_dim=args.feature_dim, dropout=args.dropout_rate, local_token_num=args.Q, global_token_num = args.P, t = args.tau_)
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
        #dataloader.sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            img1 = data['source_img_data'].cuda()
            img2 = data['target_img_data'].cuda()
            mods = data['mod']['str']   

            optimizer.zero_grad()
            with autocast():
                loss = model.compute_loss(img1, mods, img2)
                total_loss = loss['stu_rank'] \
                              + args.lambda_ * loss['tea_rank'] \
                              + args.eta_ * loss['tea_mask'] \
                              + args.mu_ * loss['ortho'] \
                              + args.nu_ * loss['ckd'] \
                              + args.kappa_ * loss['kl']

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


def train_and_evaluate(model, optimizer, trainset):
    trainloader = dataloader.DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=args.num_workers)
    
    current_best_score = float('-inf')
    best_parameters_model = None
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

        train(model, optimizer, trainloader, scaler, epoch)
        current_score = 0
        current_result = []
        if args.dataset == 'fashioniq':
            for ci, category in enumerate(['dress', 'shirt', 'toptee']):
                t = test.test(args, model, trainset, category)
                logging.info(t)
                current_score = current_score + t[1][1]
                current_result.append(t)

            if current_score > current_best_score:
                current_best_score = current_score
                tolerance = 0
                best_json_path_combine = os.path.join(
                        args.model_dir, "metrics_best.json")
                test_metrics = {}
                
                for _ in current_result:
                    for metric_name, metric_value in _:
                        test_metrics[metric_name] = metric_value

                utils.save_dict_to_json(test_metrics, best_json_path_combine)
                best_parameters_model = model
        else:
            if args.dataset == 'shoes':
                t = test.test(args, model, trainset, 'shoes')
                logging.info(t)
                current_score = current_score + t[1][1] + t[2][1]
            elif args.dataset == 'cirr':
                t = test.test_cirr_valset(args, model, trainset)
                logging.info(t)
                current_score = t[0][1] + t[1][1] + t[2][1] + t[3][1] + t[4][1] + t[5][1] + t[6][1] # mean best
            if current_score > current_best_score:
                current_best_score = current_score
                tolerance = 0
                best_json_path_combine = os.path.join(
                        args.model_dir, "metrics_best.json")
                test_metrics = {}
                for metric_name, metric_value in t:
                    test_metrics[metric_name] = metric_value

                utils.save_dict_to_json(test_metrics, best_json_path_combine)
                best_parameters_model = model 
        
    return current_best_score, test_metrics, best_parameters_model


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
    torch.backends.cudnn.deterministic = False

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')

    trainset = load_dataset()

    best_score = float('-inf')
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    _best_score, _metrics, current_model = train_and_evaluate(model, optimizer, trainset)
    if _best_score > best_score:
        best_score = _best_score
        utils.save_dict_to_json(_metrics, os.path.join(args.model_dir, "metrics_best.json"))
        torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))