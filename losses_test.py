import torch
#import pytest

from model.losses import dice_coefficient, LossComputer, mask_to_bbox, get_neighbor_images_patch_color_similarity, topk_mask, compute_project_term, unfold_patches, calculate_temporal_loss, get_self_similarity
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
import matplotlib.pyplot as plt
import tqdm


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

def sample_data(batch_size, num_samples=24, first_rgb_frame=False, yt=False, prog_bar=False):
    assert (num_samples % batch_size) == 0
    max_skip=5
    

    if yt:
        yv_root = path.join(path.expanduser('../YouTube'), 'train_480p')
        yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                        path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=list(load_sub_yv())[:num_samples],
                        num_frames=6, finetune=False, first_frame_bbox=False)
        train_loader = DataLoader(yv_dataset, batch_size, shuffle=False)
    else:
        if num_samples > 25:
            subset = load_sub_davis()
        davis_root = path.join(path.expanduser('../DAVIS'), '2017', 'trainval')
        subset = ['camel', 'cows','bear', 'bmx-bumps', 'boat', 'breakdance','car-roundabout', 'car-shadow', 
                'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high'][:num_samples] if num_samples<=25 else None
        davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                        path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=subset,
                        num_frames=6, finetune=False, first_frame_bbox=False)
        train_loader = DataLoader(davis_dataset, batch_size, shuffle=False)
    # device 
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
#    dev = torch.device('cpu')
    if prog_bar:
        train_loader = tqdm.tqdm(train_loader)
    for sample in train_loader:
        B, T = sample['rgb'].shape[:2]
        rgb = sample['rgb'].to(dev)
        rgb = inv_im_trans(rgb)
        gt = sample['cls_gt'].to(dev)
        if not first_rgb_frame:
            rgb = rgb[:, 1:]
            gt = gt[:, 1:]
            T -= 1
        bboxes, gt_masks = mask_to_bbox(gt, 3)

        yield rgb, B, T, bboxes, gt_masks, sample

def test_temporal():
    accs = []
    losses1, losses2, losses3 = [], [], []
    for sample in sample_data(yt=True, batch_size=1, num_samples=500):
        rgb, B, T, bboxes, gt_masks, data = sample
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

        sims_T_dim_second = torch.stack(sims, dim=0)
        # T x B*n_obj x K2 x H x W
        sims = torch.stack(sims, dim=1)

        sims = (sims > 0.05).float()
        mean = sims.mean(dim=(1,2,3,4))
        print(mean)
        print(sims.sum()/sims.numel())

#        acc_flag, acc = check_accuracy(sims, gt_masks)
#        accs.extend(acc)
#
#        if acc_flag:


#        with torch.no_grad():
#            rows = []
#            for obj_idx in range(3):
#                rows.append(
#                    torch.cat([sims[0, 0, 3*obj_idx+i] for i in range(3)], dim=1)
#                )
#            img = torch.cat(rows, dim=0)
#            cv2.imwrite(f'vis_mask_check/simall.png',img.repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)
#
#        #rgb = rgb[:, :, [2,1,0]]
#        rgb_img = rgb.repeat_interleave(3, dim=0)
#        #rgb_img = inv_im_trans(rgb_img)
#        rgb_img = rgb_img[:, :, [2,1,0]]
#
#        save_image(rgb_img, 5, 6, 'rgb')
#        save_image(bboxes, 5, 6, 'bboxes')
#        save_image(sims[:,:,4], 5, 6, 'sims')
        
       # input('press y')

@torch.no_grad()
def save_image(img, x, y, name):
    """
        Save image of shape B x T x 3 x H x W
        or B x T x H x W
    """
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


def check_pairwise_similarity():
    for sample in sample_data(yt=True, batch_size=2, num_samples=6):
        rgb, B, T, bboxes, gt_masks, data = sample

        images_rgb = 255.*rgb
        images_rgb = images_rgb[:, :, [2,1,0]]
        images_lab = torch.stack([torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)
                        for rgb_img in images_rgb.flatten(0,1)])

        images_lab = images_lab.reshape(images_rgb.shape)
        images_lab_sim = get_self_similarity(images_lab, B, T, num_obj=3)
        images_lab_sim = images_lab_sim >= 0.3
        save_image(images_lab_sim.reshape(6,5,8,384,384)[:,:,0], 5, 6, 'self_sim')
        rgb_img = rgb.repeat_interleave(3, dim=0)
        rgb_img = rgb_img[:, :, [2,1,0]]
        save_image(rgb_img, 5, 6, 'rgb')

        print()


def test_accuracy():
    accs = []
    for sample in sample_data(yt=True, batch_size=10, num_samples=4000, prog_bar=True):
        rgb, B, T, bboxes, gt_masks, data = sample

        images_rgb = 255.*rgb#*inv_im_trans(sample['rgb'])[:,1:] # B x T x 3 x H x W
        images_rgb = images_rgb[:, :, [2,1,0]]
        images_lab = torch.stack([torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)
                        for rgb_img in images_rgb.flatten(0,1)])

        images_lab = images_lab.reshape(images_rgb.shape)

        sims, offset = get_neighbor_similarity(
            images_lab, B, T, num_obj=3,
            tube_len=5, theta=0.3, patch_size=3, search_size=3
        )

        # B*n_obj x T x K2 x H x W
        sims = torch.stack(sims, dim=1)

        accs_t = []
        for t in [0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            sims_threshold = sims >= t
#            mean = sims_threshold.mean(dim=(1,2,3,4))
#            print(mean)
#            print(sims_threshold.sum()/sims_threshold.numel())

            acc = check_accuracy(sims_threshold, gt_masks)
            accs_t.append(acc)
        accs.append(accs_t)

    accs = np.array(accs)
    print(np.mean(accs, axis=0))
    print('-'*50)
    print(np.nanmean(accs, axis=0))
    
 
def check_accuracy(sims, masks):
    """
        TODO: visualize which pixels are wrong
                - count # pixels / img wrong
                - percentage of pixels wrong


        BUG: should not have n_obj since similarities are same for all objects
    """

    # sims =  B*n_obj x T x K2 x H x W
    # masks = B*n_obj x T x H x W
    # B*n_obj x T x K2 x H x W
    unf_masks = unfold_patches(masks, 3, 3, remove_center=False)
    # B*n_obj x K2 x H x W
    accs = []
    num = 0
    den = 0
    for i in range(sims.shape[1]):
        gt_sim = unf_masks[:, i] == unf_masks[:, (i+1)%sims.shape[1]]
        pred_sim = sims[:, i]
        # when similarity is 1, what is the accuracy?
        correct_pred = pred_sim & gt_sim 
#        num_wrong_guess = (pred_sim.sum()-correct_pred.sum())/(sims.shape[0]*sims.shape[2]) #/ pred_sim.sum()
        prec = correct_pred.sum() / pred_sim.sum()
#        num_ones = pred_sim.sum()
        #acc = np.array([prec, num_wrong_guess, num_ones])
        num += correct_pred.sum()
        den += pred_sim.sum()
#        accs.append(prec)
#        if acc < 0.94:
#        cv2.imwrite(f'vis_mask_check/debug.png',correct_pred[0,0].float().repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)
        #print(acc) 
    return den.item()
    return (num/den).item()
    accs = np.array(accs)
    return accs.mean(axis=0)


    #[0.96964358 0.99405529 0.99471485 0.99619525 0.99677592 0.99724341 0.99738567 0.9971882 ]


def test_loss_decrease():
    losses1, losses2, losses3 = [], [], []
    for sample in sample_data(yt=True, batch_size=1, num_samples=500):
        rgb, B, T, bboxes, gt_masks, data = sample

        images_rgb = 255.*rgb
        images_rgb = images_rgb[:, :, [2,1,0]]
        images_lab = torch.stack([torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)
                        for rgb_img in images_rgb.flatten(0,1)])

        images_lab = images_lab.reshape(images_rgb.shape)

        sims, offset = get_neighbor_similarity(
            images_lab, B, T, num_obj=3,
            tube_len=5, theta=0.3, topk=5,
        )
        sims_T_dim_second = torch.stack(sims, dim=0)

        l1, l2, l3=make_sure_loss_decrease(data, gt_masks, bboxes, sims_T_dim_second, offset)
        losses1.append(l1)
        losses2.append(l2)
        losses3.append(l3)
        print('-'*20)
        
    losses1, losses2, losses3 = np.array(losses1), np.array(losses2), np.array(losses3)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the histograms on each subplot
    axs[0].hist(losses1, bins=20)
    axs[0].set_title('Exact mask logits')

    axs[1].hist(losses2, bins=20)
    axs[1].set_title('Bbox logits')

    axs[2].hist(losses3, bins=20)
    axs[2].set_title('Random logits')

    plt.savefig('figures/histogram.png')
    

def make_sure_loss_decrease(data, gt_masks, bboxes, sims, offset):
    """
        Loss from guessing perfect mask should be lower than for guessing bbox mask
        and random mask 

        perfect < bbox < random

        Is loss between [0, inf)?
    """
    params = {
        'color_threshold':0.05,
        'kernel_size':3, 'dilation_size':3,
        'offset':offset, 'tube_len':5
    }

    # 1. Create logits based on perfect segmentation mask
    gt_logits = gt_masks.logit(eps=1e-7)
    loss_gt = calculate_temporal_loss(gt_logits, bboxes, sims, **params) 
    loss_gt_mean = sum(loss_gt.values())/len(loss_gt)
    print(f'mask logits: {loss_gt_mean}')

    # 2. Create logits based on bbox segmentation mask
    bbox_logits = bboxes.logit(eps=1e-7)
    loss_bbox = calculate_temporal_loss(bbox_logits, bboxes, sims, **params) 
    loss_bbox_mean = sum(loss_bbox.values())/len(loss_bbox)
    print(f'bbox logits: {loss_bbox_mean}')

    # 3. Create logits based on random  mask
    #rand_mask = (torch.randn_like(bboxes) > 0.5).float()
    rand_mask = torch.randn_like(bboxes)
    #rand_mask = 0.2*torch.ones_like(bboxes)
    rand_logits = rand_mask.logit(eps=1e-7)
    loss_rand = calculate_temporal_loss(rand_logits, bboxes, sims, **params) 
    loss_rand_mean = sum(loss_rand.values())/len(loss_rand)
    print(f'rand logits: {loss_rand_mean}')

    return loss_gt_mean, loss_bbox_mean, loss_rand_mean


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

def kornia_test():
    import kornia.color as kol

    import time
#    start = time.time()
#    for _ in range(10):
#        rgb_img = torch.rand((3,384,384)).cuda()
#        res_a = torch.as_tensor(color.rgb2lab((255.*rgb_img).byte().permute(1, 2, 0).cpu().numpy()), device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)
#       # res_b = kol.rgb_to_lab(rgb_img)
#        print(torch.abs(res_a-res_b).max())
#        print(res_a.max())
#    end = time.time()
    
    rgb_img = 3*torch.rand((5,3,384,384)).cuda()

    res_b = kol.rgb_to_lab(rgb_img)

    print(res_b.max())


    print(f'kornia time: {end-start}')



if __name__=='__main__':
    #test_temporal()
    test_accuracy()