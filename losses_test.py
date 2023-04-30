import torch
#import pytest

from model.losses import dice_coefficient, LossComputer, mask_to_bbox, get_neighbor_images_patch_color_similarity, topk_mask, compute_project_term, unfold_patches
from model.losses import get_neighbor_similarity
from util.configuration import Configuration

import torchvision.transforms as transforms
from skimage import color
import random
from os import path
from torch.utils.data import DataLoader
from dataset.vos_dataset import VOSDataset
from util.load_subset import load_sub_davis, load_sub_yv
import cv2
import numpy as np


np.random.seed(50)
inv_im_trans = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])



def test_dice_coefficient():
    # Test case 1: Perfect prediction
    x = torch.ones(1, 4, 4)
    target = torch.ones(1, 4, 4)
    loss = compute_project_term(x, target)
    assert loss == 0.0

    # Test case 2: Perfect negative prediction
    x = torch.zeros(1, 4, 4)
    target = torch.zeros(1, 4, 4)
    loss = compute_project_term(x, target)
    assert loss == 0.0

    # Test case 3: Half of the prediction is correct
    x = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float)
    target = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]], dtype=torch.float)
    loss = compute_project_term(x, target)
    assert loss==0.5

    # Test case 4: Completely wrong prediction
#    x = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float)
#    target = torch.tensor([[[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1,

#test_dice_coefficient()

def test_proj_loss():
    bboxes = bboxes.flatten(0,1)[:, None]
    gt_masks = gt_masks.flatten(0,1)[:, None]
    gt_logits = gt_masks.logit(eps=1e-7)
    print(f'gt_logits: {gt_masks.shape}')

    test1 = torch.zeros((5,1,384,384))
    test1[:2] = 0.
#        test2 = torch.ones((5,1,384,384))

    #gt_masks = (~(bboxes.bool())).float()
    gt_masks = torch.zeros_like(bboxes)
    
    res = compute_project_term(gt_masks, bboxes)

    print(f'proj loss {res}')

def sample_data(batch_size, num_samples=24, first_rgb_frame=False, yt=False):
    assert (num_samples % batch_size) == 0
    max_skip=5
    davis_root = path.join(path.expanduser('../DAVIS'), '2017', 'trainval')
    yv_root = path.join(path.expanduser('../YouTube'), 'train_480p')
    subset = ['camel', 'cows','bear', 'bmx-bumps', 'boat', 'breakdance','car-roundabout', 'car-shadow', 
             'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high'][:num_samples] if num_samples<=25 else None
    
    if num_samples > 25:
        subset = load_sub_davis()

    davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                        path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=subset,
                        num_frames=6, finetune=False, first_frame_bbox=False)

    if yt:
        yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                        path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=list(load_sub_yv())[:num_samples],
                        num_frames=6, finetune=False, first_frame_bbox=False)
        train_loader = DataLoader(yv_dataset, batch_size, shuffle=False)
    else:
        train_loader = DataLoader(davis_dataset, batch_size, shuffle=False)

    for sample in train_loader:
        B, T = sample['rgb'].shape[:2]
        rgb = sample['rgb']
        rgb = inv_im_trans(rgb)
        gt = sample['cls_gt']
        if not first_rgb_frame:
            rgb = rgb[:, 1:]
            gt = gt[:, 1:]
            T -= 1
        bboxes, gt_masks = mask_to_bbox(gt, 3)

        yield rgb, B, T, bboxes, gt_masks

def test_temporal():
    accs = []
    for sample in sample_data(yt=True, batch_size=2, num_samples=24):
        rgb, B, T, bboxes, gt_masks = sample
#        for i in range(bboxes.shape[1]):
#            bboxes[:,i]= bboxes[:, 0]
#        for i in range(rgb.shape[1]):
#            rgb[:,i]=.03+rgb[:,max(0,i-1)]
#            if i==rgb.shape[1]-1:
#                rgb[:,i]=rgb[:,0]
#        print(torch.all(rgb[:,0]==rgb[:,-1]))
#        rgb = rgb.clamp(min=0., max=1.)

        # torch.Size([2, 5, 3, 384, 384])
#        rgb = torch.ones_like(rgb)
#        rgb[...,100:150, 200:350] = 1.


        images_rgb = 255.*rgb#*inv_im_trans(sample['rgb'])[:,1:] # B x T x 3 x H x W
        images_rgb = images_rgb[:, :, [2,1,0]]
        images_lab = torch.stack([torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)
                        for rgb_img in images_rgb.flatten(0,1)])

        images_lab = images_lab.reshape(images_rgb.shape)


        # REMOVE"!
#        images_lab = images_rgb
#        masks_input = gt_masks[[0, 3]].unsqueeze(2).repeat(1, 1, 3, 1, 1)
#        rgb = rand_with_object(rgb, ~(masks_input.bool()))
#        rgb = rand_with_object(rgb, masks_input)
#        images_lab = rgb

        sims, offset = get_neighbor_similarity(
            images_lab, B, T, num_obj=3,
            tube_len=5, theta=0.3, topk=5,
        )

        # T x B*n_obj x K2 x H x W
        sims = torch.stack(sims, dim=1)

        sims = (sims > 0.05).float()
        mean = sims.mean(dim=(1,2,3,4))
        print(mean)
        print(sims.sum()/sims.numel())

        acc_flag, acc = check_accuracy(sims, gt_masks)
#        accs.extend(acc)
#
#        if acc_flag:


        with torch.no_grad():
            rows = []
            for obj_idx in range(3):
                rows.append(
                    torch.cat([sims[0, 0, 3*obj_idx+i] for i in range(3)], dim=1)
                )
            img = torch.cat(rows, dim=0)
            cv2.imwrite(f'vis_mask_check/simall.png',img.repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)

        #rgb = rgb[:, :, [2,1,0]]
        rgb_img = rgb.repeat_interleave(3, dim=0)
        #rgb_img = inv_im_trans(rgb_img)
        rgb_img = rgb_img[:, :, [2,1,0]]

        save_image(rgb_img, 5, 6, 'rgb')
        save_image(bboxes, 5, 6, 'bboxes')
        save_image(sims[:,:,4], 5, 6, 'sims')
        
       # input('press y')


def save_image(img, x, y, name):
    if img.shape[-3] != 3:
        img = img.unsqueeze(-3)

    rows = []
    for obj_idx in range(y):
        rows.append(
            torch.cat([img[obj_idx,tube_idx] for tube_idx in range(x)], dim=2)
        )
    img = torch.cat(rows, dim=1)
    if img.shape[-3] != 3:
        img = img.repeat(3,1,1)
        
    cv2.imwrite(f'vis_mask_check/{name}.png',img.permute(1,2,0).float().cpu().numpy() * 255)
 
def check_accuracy(sims, masks):
    # sims =  B*n_obj x T x K2 x H x W
    # masks = B*n_obj x T x H x W
    flag = False
    # B*n_obj x T x K2 x H x W
    unf_masks = unfold_patches(masks, 3, 3, remove_center=False)
    # B*n_obj x K2 x H x W
    accs = []
    for i in range(sims.shape[1]):
        gt_sim = unf_masks[:, i] == unf_masks[:, (i+1)%sims.shape[1]]
        pred_sim = sims[:, i]>=0.01#0.00001
        # when similarity is 1, what is the accuracy?
        correct_pred = pred_sim & gt_sim 
        acc = correct_pred.sum() / pred_sim.sum()
        accs.append(acc)
#        if acc < 0.94:
        cv2.imwrite(f'vis_mask_check/debug.png',correct_pred[0,0].float().repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)
        print(acc) 
        flag = True
    print()
    return flag, accs


def uniform_img(value):
    ...

def rand_with_square(rgb):
    img = torch.randn_like(rgb)

    val = float(0)
    img[...,100:150, 200:350] = val
    img[...,50:100, 200:220] = val
    img[...,50:55, 200:350] = val

    return img


def rand_with_object(img, mask):
    bg_mask = ~(mask.bool())
    rand_img = torch.randn_like(img)

    # add random noise to background
    img[bg_mask] = rand_img[bg_mask]
    return img

def track_bg(img, mask):
    fg_mask = mask.bool()
    rand_img = torch.randn_like(img)

    # add random noise to background
    img[fg_mask] = rand_img[fg_mask]
    return img

if __name__=='__main__':
    test_temporal()