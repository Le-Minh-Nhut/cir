import os
import csv
import json
import utils
import torch
import open_clip
import numpy as np
from torch.utils.data import dataloader
import torchvision
import datasets

from tqdm import tqdm as tqdm

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

"""get cirr testset result, save to json"""
@torch.no_grad()
def test_cirr_submit_result(model, testset, save_dir, batch_size = 16):
    # eval
    model.eval()

    # query feature
    test_queries = testset.test_queries
    all_queries = []
    imgs = []
    mods = []
    pairid = []
    subset = []
    reference_name = []
    for i, data in enumerate(tqdm(test_queries)):
        imgs += [data['reference_data']]
        mods += [data['mod']]
        pairid += [data['pairid']]
        reference_name += [data['reference_name']]
        subset.append(list(data['subset']))
        if len(imgs) >= batch_size or i == len(test_queries) - 1:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            dataset_type = torch.ones(imgs.shape[0]).cuda() * dataset_name_dict['cirr']
            q = model.compose_query(imgs, mods, dataset_type)[0].data.cpu().numpy()
            all_queries += [q]
            imgs = []
            mods = []
    # all_queries = torch.vstack(all_queries) # (M,D)
    all_queries = np.concatenate(all_queries)

    # targets feature
    candidate_names, candidate_img = testset.test_name_list, testset.test_img_data
    candidate_features = []
    imgs = []
    for i, img_data in enumerate(tqdm(candidate_img)):
        imgs += [img_data]
        if len(imgs) >= batch_size or i == len(candidate_img) - 1:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            dataset_type = torch.ones(imgs.shape[0]).cuda() * dataset_name_dict['cirr']
            features = model.compose_target(imgs, dataset_type)[0].data.cpu().numpy()
            candidate_features += [features]
            imgs = []
    # candidate_features = torch.vstack(candidate_features) 
    candidate_features = np.concatenate(candidate_features) # (N,D)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(candidate_features.shape[0]):
        candidate_features[i, :] /= np.linalg.norm(candidate_features[i, :])

    sims = - all_queries.dot(candidate_features.T) # (M,N)
    sorted_inds = np.argsort(sims, axis=-1)
    sorted_ind_names = np.array(candidate_names)[sorted_inds] # (M,N)

    mask = torch.tensor(sorted_ind_names != np.repeat(np.array(reference_name), len(candidate_names)).reshape(len(sorted_ind_names),-1)) # (M,N)
    sorted_ind_names = sorted_ind_names[mask].reshape(sorted_ind_names.shape[0], sorted_ind_names.shape[1] - 1) # (M,N-1)

    subset = np.array(subset) # (M,6)
    subset_mask = (sorted_ind_names[..., None] == subset[:, None, :]).sum(-1).astype(bool) # (M,N-1) label elements in subset
    sorted_subset_names = sorted_ind_names[subset_mask].reshape(sorted_ind_names.shape[0], -1) # (M,6)

    pairid_to_gengeral_pred = {str(int(pair_id)): prediction[:50].tolist()  for pair_id, prediction in zip(pairid, sorted_ind_names)}
    pairid_to_subset_pred = {str(int(pair_id)): prediction[:3].tolist() for pair_id, prediction in zip(pairid, sorted_subset_names)}

    general_submission = {'version': 'rc2', 'metric': 'recall'}
    subset_submission = {'version': 'rc2', 'metric': 'recall_subset'}

    general_submission.update(pairid_to_gengeral_pred)
    subset_submission.update(pairid_to_subset_pred)

    print('save cirr test result')
    with open(os.path.join(save_dir, 'test1_pred_ranks_recall.json'), 'w+') as f:
        json.dump(general_submission, f, sort_keys=True)
        
    with open(os.path.join(save_dir, 'test1_pred_ranks_recall_subset.json'), 'w+') as f:
        json.dump(subset_submission, f, sort_keys=True)

if __name__ == '__main__':
    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13B_b90k')
    testset = datasets.CIRR(path='[CIRR dataset path]',transform=[preprocess_train, preprocess_val])
    model_dir = '[saved cirr checkpoint path]'
    # load model
    model = torch.load(os.path.join(model_dir, '[No.]_cirr_best_model.pt'))
    # generate submission json file
    test_cirr_submit_result(model, save_dir=model_dir, testset=testset, batch_size=128)