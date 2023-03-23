import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import masks_to_boxes
import torchvision.transforms as transforms

from collections import defaultdict

from skimage import color
import cv2
import numpy as np

import random


def compute_pairwise_term_neighbor(mask_logits, mask_logits_neighbor, pairwise_size, pairwise_dilation):
	# N x T x H x W
    assert mask_logits.dim() == 4

    log_fg_prob_neigh = F.logsigmoid(mask_logits_neighbor)
    log_bg_prob_neigh = F.logsigmoid(-mask_logits_neighbor)

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    # when we compare neighboring images, center pixels arent always equal
    # hence we also include center pixel 
    log_fg_prob_unfold = unfold_patches(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation,
        remove_center=False 
    )
    log_bg_prob_unfold = unfold_patches(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation,
        remove_center=False
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob_neigh[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob_neigh[:, :, None] + log_bg_prob_unfold

    # loss = -log(prob)
    return -torch.logaddexp(log_same_fg_prob, log_same_bg_prob)[:,0]

def compute_pairwise_term(mask_logits, pairwise_size=3, pairwise_dilation=2):
    """
        Calc probabilities Pr(y_e=1)=Pr(pixels are the same)
        using input_mask (networks predictions).

        Then calculate y_e with using color similarities.
    """
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_patches(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_patches(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    # factor out the e^max so we dont get overflow. Smart!
    # when we take it out= log(e^max)+log(e^x1+e^x2)
    # This is implemented in torch.logsumexp()
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def dice_coefficient(x, target):
    #assert (x<=1.).all() and (x>=0.).all(), 'probability not in [0,1]'
    #assert (target<=1.).all() and (target>=0.).all(), 'probability not in [0,1]'
    #assert ((target == 1.) | (target == 0.)).all(), 'mask not 0,1'

    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_project_term(mask_scores, gt_bitmasks):
    # mask_scores [0,1] , gt_bitmasks = {0,1}
    # B*n_obj x 1 x H x W
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.color_threshold = 0.3
        self.col_thresh_neigh = 0.01

        self.kernel_size_neigh = 3
        self.dilation_size_neigh = 3

        # TODO: use instead of hardcoded. not used right now
        self.kernel_size = 3
        self.pairwise_dilation = 2
        self.iter = 0
        self.warmup_iters = 10_000

        # TODO: set to t
        self.tube_len = 5

        self.inv_im_trans = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225])
    
    def knn_loss(self, data, bboxes, b, t, train_iter):
        """
            Calculate the 

            TODO: send in bboxes as B*n_obj x T x H x W
        """
        # B*n_obj x T x H x W  (n_obj excludes bg)
        logits = torch.stack([data[f'logits_{i+1}'][:,1:].flatten(0,1) for i in range(t)], dim=1)

        # Remove first image frame since no mask-prediction for it will be made
        images_rgb = 255.*self.inv_im_trans(data['rgb'])[:,1:].flatten(0,1) # B*T x 3 x H x W
        images_lab = [torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)[None]
                        for rgb_img in images_rgb]
        
        # Self LAB similarities used for pairwise loss    B*T x K^2-1 x H x W
        images_lab_sim = torch.cat([get_images_color_similarity(img_lab) for img_lab in images_lab])
        # reshape into B x T x K2-1 x H x W
        images_lab_sim = images_lab_sim.reshape(b, t, *images_lab_sim.shape[-3:]).repeat_interleave(3, dim=0).flatten(0,1)

        # TODO: remove hardcoded 3, should be num_obj
        # N*T x K2-1 x H x W
       # images_lab_sim = images_lab_sim.repeat_interleave(3, dim=0).flatten(0,1)

        if not self.config['no_temporal_loss']: 
            # list of len t, B*n_obj x K^2 x H x W
            images_lab_sim_neighs = []
            # rand between [0, t-self.tube_len]
            offset = random.randint(0,t-self.tube_len)
            for i in range(self.tube_len):
                # Add cyclic neighbors between consecutive frames ii and ii+1. frame t reconnects with frame 0. 
                neigh_lab_sim = torch.cat(
                    [get_neighbor_images_patch_color_similarity(images_lab[ii+i+offset], images_lab[ii+offset+(i+1)%self.tube_len], 3, 3) 
                                for ii in range(0, len(images_lab), t)]
                )#.unsqueeze(1)

                # for every img/obj I want top k correspondences in other image
                neigh_lab_sim_top = topk_mask(neigh_lab_sim, k=5)

                # TODO: remove hardcoded 3
                # B*n_obj x K2 x H x W
                neigh_lab_sim_top = neigh_lab_sim_top.repeat_interleave(3, dim=0)
                images_lab_sim_neighs.append(neigh_lab_sim_top)

            # Calculate neighboring pairwise log probabilities
            pairwise_logprob_neighbor = []
            for i in range(self.tube_len):
                pairwise_logprob_neighbor.append(
                    compute_pairwise_term_neighbor(
                        logits[:,i+offset:i+offset+1], logits[:,offset+(i+1)%self.tube_len:offset+1+(i+1)%self.tube_len],
                        self.kernel_size_neigh, self.dilation_size_neigh 
                    ) 
                )
            
            # N x T x H x W -> N x 1 x H x W (since we sum over T dimension and keepdim)
            bbox_time_sum = (bboxes.sum(dim=1, keepdim=True) >= 1.).float()

        # Finally, compute neighbor-independent pairwise loss and projection loss
        # logits: NT x 1 x H x W
        logits = logits.flatten(0,1)[:, None]
        # first (n_obj+B) pictures are batch 1: obj 1 obj 2 obj n batch 2: obj 1 obj 2 obj n 
        bboxes = bboxes.flatten(0,1)[:, None]

        # Calculate the time-independent pairwise and projection loss
        loss_projection = compute_project_term(logits.sigmoid(), bboxes)  

        if not self.config['no_pairwise_loss']: 
            # N * 3 x H x W -> N*T x 3 x H x W
            pairwise_logprobs = compute_pairwise_term(logits, self.kernel_size, self.pairwise_dilation)
        
            weights = (images_lab_sim >= self.color_threshold).float() * bboxes.float()
            loss_pairwise = (pairwise_logprobs * weights).sum() / weights.sum().clamp(min=1.0) 

        #with torch.no_grad():
        #    rows = []
        #    for ti in range(t):
        #        rows.append(
        #            torch.cat([bboxes[6*ti+n,0] for n in range(6)], dim=1)
        #        )
        #    img = torch.cat(rows, dim=0)
        #    cv2.imwrite(f'vis_mask_check/bboxes.png',img.repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)

        #print('drop top rain drops')
#        with torch.no_grad():
#            print('saving images on!')
#            pred_mask = data[f'masks_{1}'][0,0]
#            # TODO: make sure these masks line up with gt and bbox, they do not RN
#            gt_mask = data['cls_gt'][0, 1, 0]
#            gt_box = bboxes[0,0]
#            img = torch.cat([pred_mask, gt_mask, gt_box],dim=0)
#            # now check the lab similarities
#            img_lower = torch.cat([images_lab_sim[0,0], images_lab_sim_neighs[0][0,0], images_lab_sim_neighs[0][0,4]],dim=0)
#            
#            img = torch.cat([img, img_lower], dim=1)
#            cv2.imwrite(f'vis_mask_check/pred_mask.png',img.repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)
#
#            # B*T x 3 x H x W
#            img_seq_t = torch.cat([images_rgb[i] for i in range(t)],dim=1).permute(1,2,0)
#
#            row = []
#            for i in range(t):
#                # list of len t, B*n_obj x K^2 x H x W. Batch 0, obj 0, upper left pixel
#                row.append(
#                    torch.cat([images_lab_sim_neighs[i][n,0]>=self.col_thresh_neigh for n in range(6)],dim=1).repeat(3,1,1).permute(1,2,0)*255
#                )
#            img_seq_b = torch.cat(row)
#
#            img2 = torch.cat([img_seq_t, img_seq_b], dim=1).float().cpu().numpy()
#
#            cv2.imwrite(f'vis_mask_check/sequence2.png',img2)
#
#            img3 = torch.cat([bbox_time_sum[i] for i in range(6)],dim=1).repeat(3,1,1).permute(1,2,0).float().cpu().numpy()*255
#            cv2.imwrite(f'vis_mask_check/bbox_time_union.png',img3)
#

        losses = defaultdict(int)
        if not self.config['no_temporal_loss']: 
            # Calculate weights for neighbors, and then losses using weights.
            for i in range(self.tube_len):
                weight_neigh = (images_lab_sim_neighs[i] >= self.col_thresh_neigh).float() * bbox_time_sum
                losses[f'neigh_loss_{i}'] = (pairwise_logprob_neighbor[i] * weight_neigh).sum() / weight_neigh.sum().clamp(min=1.0)

        warmup_factor = min(1., train_iter / self.warmup_iters)
        losses['proj_loss'] = loss_projection * self.tube_len
        losses['total_loss'] += losses['proj_loss']
        if not self.config['no_pairwise_loss']: 
            losses['pair_loss'] = loss_pairwise * self.tube_len
            losses['total_loss'] += warmup_factor * losses['pair_loss']
        if not self.config['no_temporal_loss']: 
            lambda_ = 0.125
            losses['neigh_mean'] = sum(losses[f'neigh_loss_{i}'] for i in range(self.tube_len)) * 1./ self.tube_len
            losses['total_loss'] += lambda_ * warmup_factor * losses['neigh_mean']

        return losses


    def compute(self, data, num_objects, it):
        """
            it = the # time this function is called
            num_objects[i] = # objects in frame[i]. i=0...Batch_size-1

            data['rgb'] = (B, 8, 3, H, W) what is 8?
            data['masks_i'] =   B * n_obj * H * W 
            data['logits_i'] =  B * n_obj+1 * H * W (+1 because we treat bg as object)

            data['cls_gt'] B x 8 x 1 x H x W
            data['info]['num_objects'] = tensor of len B (Just the num objects per batch)

            masks [0,1] are the sigmoid of logits (-inf, inf) with bg object removed
        """
        self.iter += 1
        losses = defaultdict(int)

        # get batch b, and num_frames t
        b, t = data['rgb'].shape[:2]

        no = data['logits_1'][:, 1:].shape[1]
        bboxes,gt_masks = mask_to_bbox(data['cls_gt'][:, 1:], no)
        losses = self.knn_loss(data, bboxes, b, t-1, it)

        # calculate IoU using gt and predictions
#        if (it%5)==0:
        masks = torch.stack([data[f'masks_{i}'].flatten(0,1) for i in range(1,t)], dim=1)
        losses['IoU']=dice_coefficient(masks.flatten(0,1), gt_masks.flatten(0,1))

        return losses

def unfold_patches(x, kernel_size, dilation, remove_center=True):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    if remove_center:
        size = kernel_size ** 2
        unfolded_x = torch.cat((
            unfolded_x[:, :, :size // 2],
            unfolded_x[:, :, size // 2 + 1:]
        ), dim=2)

    return unfolded_x

def get_images_color_similarity(images, kernel_size=3, dilation=2):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_patches(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    return similarity 


def get_neighbor_images_color_similarity(images, images_neighbor, kernel_size, dilation):
    """
        TODO: merge this with other image color similarity
        TODO: Take into account matching radius R?
    """
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_patches(
        images, kernel_size=kernel_size,
        dilation=dilation, remove_center=False
    )

    diff = images_neighbor[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    return similarity

def get_neighbor_images_patch_color_similarity(images, images_neighbor, kernel_size, dilation):
    # images: 1 x C x H x W
    assert images.dim() == 4
    assert images.size(0) == 1

    # TODO: why dilation=1 here, and also why always 3,3 below?
    unfolded_images = unfold_patches(
        images, kernel_size=kernel_size, dilation=1, remove_center=False
    )
    unfolded_images_neighbor = unfold_patches(
        images_neighbor, kernel_size=kernel_size, dilation=1, remove_center=False
    )
    unfolded_images = unfolded_images.flatten(1,2)
    unfolded_images_neighbor = unfolded_images_neighbor.flatten(1,2)
    
    similarity = get_neighbor_images_color_similarity(unfolded_images, unfolded_images_neighbor, 3, 3)
    return similarity

def topk_mask(images_lab_sim, k):
    """
        Return images_lab_sim with everything zerod out
        except for the top k entries in dimension 1.
        [[0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]]
        ----------------------->
        [[0.,  0.,  2.,  3.],
        [ 0.,  0.,  6.,  7.],
        [ 0.,  0., 10., 11.]]
    """
    images_lab_sim_mask = torch.zeros_like(images_lab_sim)
    topk, indices = torch.topk(images_lab_sim, k, dim=1) # 1, 3, 5, 7
    images_lab_sim_mask = images_lab_sim_mask.scatter(1, indices, topk)
    return images_lab_sim_mask

def mask_to_bbox(gt_mask, num_objects):
    # num_objects is just the shape of logits
    # mask: B x 8 x 1 x H x W
    # return bboxes as B*n_obj x T x H x W
    assert gt_mask.dim() == 5
    assert gt_mask.size(2) == 1

    # mask: 8 x B x 1 x H x W
    #gt_mask = gt_mask.permute(1,0,2,3,4)

    #obj_exist_mask = []
    masks = []
    bboxes = []
    for bn, m_b in enumerate(gt_mask):
        for i in range(num_objects): 
            for t, m in enumerate(m_b):
                mask = (m==(i+1)).float()
                bbox = torch.zeros_like(mask)

                if mask.any():
                    x1,y1,x2,y2 = masks_to_boxes(mask).long()[0]
                    bbox[0, y1:y2+1, x1:x2+1] = 1.
    #               obj_exist_mask.append(True)
                else:
                    bbox = mask
    #                obj_exist_mask.append(False)
                masks.append(mask)
                bboxes.append(bbox)
    
    # size B*n_obj. if there is an object in that entry
    # all-zero entries affect the dice loss for example 
    #obj_exist_mask = torch.tensor(obj_exist_mask)
    #target_masks = torch.stack(masks)

    target_bboxes = torch.stack(bboxes)
    target_bboxes = target_bboxes.reshape(-1, gt_mask.shape[1], gt_mask.shape[-2], gt_mask.shape[-1])

    gt_masks = torch.stack(masks)
    gt_masks = gt_masks.reshape(-1, gt_mask.shape[1], gt_mask.shape[-2], gt_mask.shape[-1])

    return target_bboxes, gt_masks
            
# test
def test_loss():
    #t1 = (torch.randn((12, 1, 300,300)) > 0.5).float()
    t_pred = torch.zeros((384,384), dtype=torch.float32)
    t_pred[100:150, 100:150] = 1.
    t_pred = t_pred[None, None]

    t_gt = torch.zeros_like(t_pred)
    t_gt[:,:,100:150, 100:150] = 1.

    loss = compute_project_term(t_pred, t_gt)
    print(loss)


    pair = compute_pairwise_term(t_pred)
    print(pair)

#test_loss()
