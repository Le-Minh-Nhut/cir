import os
import sys
import argparse
import logging
import warnings
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import dataloader
from tqdm import tqdm
import utils
import datasets
import test as test
from data_utils import squarepad_transform, targetpad_transform
from torch.cuda.amp import autocast as autocast, GradScaler
import setproctitle
from lavis.models import load_model_and_preprocess

proc_title = "python-c"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
setproctitle.setproctitle(proc_title)
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

default_paths = {
    'fashioniq_path': "./data/fashion_iq_data/",
    'cirr_path': "./data/cirr_data/CIRR/",
    'shoes_path': "",
    'birds_path': "",
    'Fashion200k_path': "",
    'lasco_path': ""
}

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default='cirr', help="data set type")
parser.add_argument('--optimizer', default='adamw')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--False_lr', type=float, default=1e-4)
parser.add_argument('--clip_lr', type=float, default=1e-5)
parser.add_argument('--False_clip_lr', type=float, default=1e-5)
parser.add_argument('--img_encoder', type=str, default='ViT-B/16')
parser.add_argument('--lr_decay', type=int, default=5)
parser.add_argument('--lr_div', type=float, default=0.1)
parser.add_argument('--clip_lr_div', type=float, default=0.1)
parser.add_argument('--max_decay_epoch', type=int, default=10)
parser.add_argument('--feature_dim', type=int, default=512)
parser.add_argument('--noise_ratio', type=float, default=0, help='noise_ratio')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--model_dir', default='./checkpoints/HINT', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--node', type=str, default='')

for key, value in default_paths.items():
    parser.add_argument(f'--{key}', default=value)

args = parser.parse_args()


def load_dataset():
    print('Reading dataset ', args.dataset)
    transform = "targetpad"
    input_dim = 224
    target_ratio = 1.25
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")
    img_transform = preprocess
    if args.dataset == 'fashioniq':
        trainset = datasets.FashionIQ(
            path=args.fashioniq_path,
            transform=img_transform,
            noise_ratio=args.noise_ratio)
    elif args.dataset == 'shoes':
        trainset = datasets.Shoes(
            path=args.shoes_path,
            transform=img_transform)
    elif args.dataset == 'cirr':
        trainset = datasets.CIRR(
            path=args.cirr_path,
            transform=img_transform,
            case_look=False,
            noise_ratio=args.noise_ratio
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
    blip_model_name = "HINT"
    backbone = "pretrain"
    model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name,
                                                                      model_type=backbone,
                                                                      is_eval=False, device=args.device)
    model.cuda()
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])

    return model, optimizer, txt_processors


def train(model, optimizer, dataloader, scaler, epoch, txt_processors, step):
    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        # dataloader.sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            if args.dataset == 'fashion200k':
                assert type(data) is list
                img1 = np.stack([d['source_img_data'] for d in data])
                img1 = torch.from_numpy(img1).float()
                img1 = torch.autograd.Variable(img1).cuda()
                img2 = np.stack([d['target_img_data'] for d in data])
                img2 = torch.from_numpy(img2).float()
                img2 = torch.autograd.Variable(img2).cuda()
                mods = [str(d['mod']['str']) for d in data]
                mods = [t.encode('utf-8').decode('utf-8') for t in mods]
            else:
                img1 = data['source_img_data'].cuda()
                img2 = data['target_img_data'].cuda()
                mods = data['mod']['str']
            captions = [txt_processors["eval"](caption) for caption in mods]
            optimizer.zero_grad()
            with autocast():
                samples = {"image": img1, "target": img2, "text_input": captions}
                loss_dict = model(samples, args.device)
                total_loss = 0.
                total_loss = loss_dict['loss_stu_rank']

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


def train_and_evaluate(model, optimizer, trainset, testset, txt_processors):

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
        step = epoch + 1
        tolerance += 1
        if tolerance == 10:
            break

        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        train(model, optimizer, trainloader, scaler, epoch, txt_processors, step=step)
        current_score = 0
        current_result = []
        if args.dataset == 'fashioniq':
            for ci, category in enumerate(['dress', 'shirt', 'toptee']):
                t = test.test(args, model, trainset, category, txt_processors)
                logging.info(t)
                current_score = current_score + t[1][1]
                current_result.append(t)

            torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
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
            if args.dataset == 'cirr':
                torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
                t = test.test_cirr_valset(args, model, trainset, txt_processors)
                logging.info(t)
                current_score = t[0][1] + t[1][1] + t[2][1] + t[3][1] + t[4][1] + t[5][1] + t[6][1]  # mean best
            
            if current_score > current_best_score:
                current_best_score = current_score
                tolerance = 0
                best_json_path_combine = os.path.join(
                    args.model_dir, "metrics_best.json")
                test_metrics = {}
                for metric_name, metric_value in t:
                    test_metrics[metric_name] = metric_value
                torch.save(model, os.path.join(args.model_dir, 'best_model.pt'))
                utils.save_dict_to_json(test_metrics, best_json_path_combine)
                best_parameters_model = model

    return current_best_score, test_metrics, best_parameters_model


if __name__ == '__main__':
    print("Here")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    # Load the parameters from json file
    import setproctitle

    proc_title = "python-c"
    setproctitle.setproctitle(proc_title)
    print('Arguments:')
    for k in args.__dict__.keys():
        info = '    ' + k + ':' + str(args.__dict__[k])
        logging.info(info)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')
    if args.dataset == "birds" or args.dataset == "fashion200k":
        trainset, testset = load_dataset()
    else:
        trainset = load_dataset()
        testset = None

    best_score = float('-inf')
    model, optimizer, txt_processors = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    _best_score, _metrics, current_model = train_and_evaluate(model, optimizer, trainset, testset, txt_processors)
    if _best_score > best_score:
        best_score = _best_score
        utils.save_dict_to_json(_metrics, os.path.join(args.model_dir, "metrics_best.json"))
        torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))
