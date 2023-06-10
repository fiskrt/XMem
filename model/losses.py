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

from typing import List


def compute_pairwise_term_neighbor(mask_logits, mask_logits_neighbor, pairwise_size, pairwise_dilation):
    """
        The probability is 1 when the both pixels are either both 0 or both 1.
        
        We return the -log(prob) which has range [0, inf]
    """
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

def dice_coefficient(x: torch.tensor, target: torch.tensor, numerator_eps=True):
    """
        x, target any shape D1 x D2 x ...  
        returns tensor of shape D1
        dice = 1 - (2*TP+eps)/(2*TP+FP+FN+eps)

        But usually the shape is: 
        x=target: N x 1 x 1 x (W or H)
        where N = B*n_obj*T and last dim is either width or height
        depending on which projection is called
    """
    #assert (x<=1.).all() and (x>=0.).all(), 'probability not in [0,1]'
    #assert (target<=1.).all() and (target>=0.).all(), 'probability not in [0,1]'
    #assert ((target == 1.) | (target == 0.)).all(), 'mask not 0,1'

#    eps = 1e-5
    eps_den = 1.
    eps_num = eps_den if numerator_eps else 0.

    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1) 
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1)
    loss = 1. - ((2 * intersection + eps_num) / (union + eps_den))
    return loss

def compute_project_term(mask_scores, gt_bitmasks, numerator_eps=True):
    """
    # mask_scores: B*n_obj(*T) x 1 x H x W
    # gt_bitmasks: B*n_obj(*T) x 1 x H x W
    # mask_scores [0,1] , gt_bitmasks = {0,1}

    """
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0],
        numerator_eps=numerator_eps
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0],
        numerator_eps=numerator_eps
    )
    return (mask_losses_x + mask_losses_y).mean()


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.color_threshold = self.config['pairwise_color_threshold']
        self.col_thresh_neigh = self.config['temporal_color_threshold']

        self.kernel_size_neigh = 3
        self.dilation_size_neigh = 3
        self.theta_neigh = self.config['temporal_theta']

        # TODO: use instead of hardcoded. not used right now
        self.kernel_size = 3
        self.pairwise_dilation = 2
        self.iter = 0
        self.warmup_iters = self.config['pairwise_warmup_steps']

        # TODO: set to t
        self.tube_len = 5

        self.inv_im_trans = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225])

    def first_frame_gt_loss(self, data):
        """
            x  o  o  o  o  o  o  o  o
            |__|__|__|
            Here x is the first frame with gt available

            TODO: can cut off logits/gt_masks to only 3 in temporal dim?
            but maybe this is useless?

            images_lab is a list of length B*t where each element is the LAB converted rgb image.
            t here is INCLUDES the first frame! This is not the case in other loss functions.

        """
        b, t = data['rgb'].shape[:2]
        num_objects = data['logits_1'].shape[1]-1
        num_loss_frames = self.config['num_loss_frames']

        # B * n_obj x T x H x W  (n_obj excludes bg)
        logits = torch.stack([data[f'logits_{i}'][:,1:].flatten(0,1) for i in range(1,t)], dim=1)

        # B x T=1 x n_obj x H x W
        first_frame_gt = data['first_frame_gt'].float()
        assert torch.bitwise_or(first_frame_gt == 1., first_frame_gt == 0.).all()
        # B*n_obj x T=1 x H x W
        first_frame_gt = first_frame_gt.reshape(b*num_objects, 1, *first_frame_gt.shape[-2:])

        # Remove first image frame since no mask-prediction for it will be made
        # B*T x 3 x H x W
        images_rgb = 255.*self.inv_im_trans(data['rgb']).flatten(0,1)
        # Turn rgb into L*a*b* with ranges L*: [0, 100], a*: [-127, 128] b*: [-128,127]
        images_lab = [torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)[None]
                        for rgb_img in images_rgb]
        
        # get col sim between first frame and the i'th frame
        lab_sims = [] 
        for i in range(num_loss_frames):
            # B x K^2 x H x W. The LAB similarity between frame 0 and i
            # TODO: remove hardcoded 3,2
            first_frame_i_lab_sim = torch.cat(
                [get_neighbor_images_color_similarity(images_lab[ii+i+1], images_lab[0+ii], 3, 2) 
                            for ii in range(0, len(images_lab), t)]
            )

            lab_sims.append(first_frame_i_lab_sim)
        
        # num_loss_frames is <= T
        # B x num_loss_frames x K^2 x H x W  
        lab_sims = torch.stack(lab_sims, dim=1)
        # B*n_obj x num_loss_frames x K^2 x H x W  
        lab_sims = lab_sims.repeat_interleave(num_objects, dim=0)
        lab_sims = (lab_sims > 0.4).float()

        # since we only use the first num_loss_frames in the loss function
        # we truncate logits/gt aswell since they are not supervised in any way.
        # B*n_obj x num_loss_frames x H x W
        logits = logits[:,:num_loss_frames]
        # B*n_obj x num_loss_frames x H x W
        first_frame_gt = torch.cat(num_loss_frames*[first_frame_gt], dim=1)

        # B*n_obj x num_loss_frames x K^2 x H x W
        logits_unfold = unfold_patches(
            logits, kernel_size=self.kernel_size,
            dilation=self.pairwise_dilation,
            remove_center=False 
        )

        # Since F.cross_ent does not broadcast we have to add dimension to first_frame_gt 
        # B*n_obj x num_loss_frames x K^2 x H x W
        first_frame_gt_broadcast = torch.stack(logits_unfold.shape[-3]*[first_frame_gt], dim=2)

#        gt_unfold = unfold_patches(
#            first_frame_gt, kernel_size=self.kernel_size,
#            dilation=self.pairwise_dilation,
#            remove_center=False 
#        )

        loss = F.binary_cross_entropy_with_logits(logits_unfold, first_frame_gt_broadcast, reduction='none')
        #loss2 = (loss * lab_sims).mean()
        loss3 = (loss*lab_sims).sum() / lab_sims.sum().clamp(min=1.)
        losses = defaultdict(int)
        losses['frame0gt_propagation_loss'] = loss3
        losses['total_loss'] += losses['frame0gt_propagation_loss']
        return losses
        
    
    def knn_loss(self, data, bboxes, b, t, train_iter):
        """
            Calculate the 

            TODO: send in bboxes as B*n_obj x T x H x W
        """
        # B*n_obj x T x H x W  (n_obj excludes bg)
        logits = torch.stack([data[f'logits_{i+1}'][:,1:].flatten(0,1) for i in range(t)], dim=1)

        # Remove first image frame since no mask-prediction for it will be made
        images_rgb = 255.*self.inv_im_trans(data['rgb'])[:,1:] # B x T x 3 x H x W
        images_lab = torch.stack([torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                        device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)
                                         for rgb_img in images_rgb.flatten(0,1)]
                    )

        images_lab = images_lab.reshape(images_rgb.shape)

        images_lab_sim = get_self_similarity(images_lab, b, t, num_obj=3)
        images_lab_sim_neighs, offset = get_neighbor_similarity(
                                                                images_lab, b, t,
                                                                num_obj=3,
                                                                tube_len=self.tube_len,
                                                                theta = self.theta_neigh
                                        )
        
        params = [logits, bboxes, images_lab_sim_neighs,
                    self.col_thresh_neigh, self.kernel_size_neigh,
                    self.dilation_size_neigh, offset, self.tube_len
        ]

        if self.config['detach_temporal_loss']:
            loss_temporal1 = calculate_temporal_loss(*params, detach=1) 
            loss_temporal2 = calculate_temporal_loss(*params, detach=2) 
            loss_temporal = {k:0.5*(loss_temporal1[k] + loss_temporal2[k]) for k in loss_temporal1}
        else:
            loss_temporal = calculate_temporal_loss(*params) 


        # Finally, compute neighbor-independent pairwise loss and projection loss
        # logits: NT x 1 x H x W
        logits = logits.flatten(0,1)[:, None]
        # first (n_obj+B) pictures are batch 1: obj 1 obj 2 obj n batch 2: obj 1 obj 2 obj n 
        bboxes = bboxes.flatten(0,1)[:, None]

        losses = defaultdict(float)

        ratio = (logits.sigmoid()*bboxes).sum() / bboxes.sum().clamp(min=1.0)
        losses['ratio'] = ratio
        ratio_threshold = self.config['ratio_loss_threshold']
        if self.config['use_ratio_loss']:
            losses['total_loss'] += max(0., ratio_threshold-ratio) # when 0<ratio<0.2, the  

#        if ratio < 0.2:
#            print(f'ratio is {ratio} at iteration {train_iter}')

        # Calculate the time-independent pairwise and projection loss
        loss_projection = compute_project_term(logits.sigmoid(),
                                                bboxes,
                                                numerator_eps=self.config['dice_numerator_smoothing']
                        )

        # N * 3 x H x W -> N*T x 3 x H x W
        pairwise_logprobs = compute_pairwise_term(logits, self.kernel_size, self.pairwise_dilation)
    
        weights = (images_lab_sim >= self.color_threshold).float() * bboxes.float()
        loss_pairwise = (pairwise_logprobs * weights).sum() / weights.sum().clamp(min=1.0) 


        warmup_factor = min(1., train_iter / self.warmup_iters)
        losses['proj_loss'] = loss_projection
        if not self.config['no_projection_loss']: 
            alpha_ = self.config['projection_loss_scale']
            losses['proj_loss_scaled'] = alpha_ * losses['proj_loss']
            losses['total_loss'] += losses['proj_loss_scaled']

        losses['pair_loss'] = loss_pairwise
        if not self.config['no_pairwise_loss']: 
            gamma_ = self.config['pairwise_loss_scale'] 
            losses['pair_loss_scaled'] = gamma_ * warmup_factor * losses['pair_loss']
            losses['total_loss'] += losses['pair_loss_scaled']

        losses.update(loss_temporal)
        losses['neigh_mean'] = sum(losses[f'neigh_loss_{i}'] for i in range(self.tube_len)) * 1. / self.tube_len
        if not self.config['no_temporal_loss']: 
            lambda_ = self.config['temporal_loss_scale']
            losses['neigh_mean_scaled'] = lambda_ * warmup_factor * losses['neigh_mean']
            losses['total_loss'] += losses['neigh_mean_scaled']

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
        # get batch b, and num_frames t
        b, t = data['rgb'].shape[:2]
        no = data['logits_1'][:, 1:].shape[1]
        bboxes,gt_masks = mask_to_bbox(data['cls_gt'][:, 1:], no)

        #GT_PRED = gt_masks.logit(eps=1e-7)
        #data['GT_PRED'] = GT_PRED
        losses = self.knn_loss(data, bboxes, b, t-1, it)

        # calculate IoU using gt and predictions
        masks = torch.stack([data[f'masks_{i}'].flatten(0,1) for i in range(1,t)], dim=1)
        losses['IoU']=dice_coefficient(masks.flatten(0,1), gt_masks.flatten(0,1))

        return losses


def get_self_similarity(images_lab, b, t, num_obj):
    """
        For every frame in video, create similarity map of pixels in the same frame.
        Used in pairwise loss.

        images_lab: # B x T x 3 x H x W

        Return tensor: B*n_obj*T x K2-1 x H x W

        TODO: fix stupid order to avoid reshape-repeat-flatten
    """
    assert images_lab.dim() == 5 
    assert images_lab.shape[0] == b and images_lab.shape[1] == t and images_lab.shape[2] == 3

    # get_images_color_similarity assumes that we have a batch dimension of 1
    images_lab = images_lab.unsqueeze(2)
    # Self LAB similarities used for pairwise loss    B*T x K^2-1 x H x W
    images_lab_sim = torch.cat([get_images_color_similarity(img_lab) for img_lab in images_lab.flatten(0,1)])
    # reshape into B x T x K2-1 x H x W
    images_lab_sim = images_lab_sim.reshape(b, t, *images_lab_sim.shape[-3:])

    # B*n_obj*T x K2-1 x H x W
    return images_lab_sim.repeat_interleave(num_obj, dim=0).flatten(0,1)

def calculate_temporal_loss(logits: torch.FloatTensor, # B*n_obj x T x H x W
                            bboxes: torch.FloatTensor, # B*n_obj x T x H x W
                            images_lab_sim_neighs: List[torch.FloatTensor], # [i] = B*n_obj x K2 x H x W 
                            color_threshold: float,
                            kernel_size: int,
                            dilation_size: int,
                            offset: int,
                            tube_len: int,
                            detach: int=-1):
    """
        Calculate the temporal loss between frames.
    """

    # Calculate neighboring pairwise log probabilities
    pairwise_logprob_neighbor = []
    for i in range(tube_len):
        pred1 = logits[:,i+offset:i+offset+1]
        pred2 = logits[:,offset+(i+1)%tube_len:offset+1+(i+1)%tube_len]
        if detach == 1:
            pred1 = pred1.detach()
        elif detach == 2:
            pred2 = pred2.detach()

        pairwise_logprob_neighbor.append(
            compute_pairwise_term_neighbor(
                pred1, 
                pred2,
                kernel_size,
                dilation_size
            ) 
        )
    
    # N x T x H x W -> N x 1 x H x W (since we sum over T dimension and keepdim)
    # TODO: only sum over tube_len and discard rest?
    bbox_time_sum = (bboxes.sum(dim=1, keepdim=True) >= 1.).float()

    losses = defaultdict(float)
    for i in range(tube_len):
        weight_neigh = (images_lab_sim_neighs[i] >= color_threshold).float() * bbox_time_sum
        losses[f'neigh_loss_{i}'] = (pairwise_logprob_neighbor[i] * weight_neigh).sum() / weight_neigh.sum().clamp(min=1.0)
    return losses

def get_neighbor_similarity(images_lab, b, t, num_obj, tube_len, theta=0.5, topk=5):
    """
        Takes 'images_lab' and turns returns the similarity with neighbors.
        Used in temporal loss.

        images_lab: # B x T x 3 x H x W

        Return list of length tube_len with elements B*n_obj x K2 x H x W
    """
    assert images_lab.dim() == 5 
    assert images_lab.shape[0] == b and images_lab.shape[1] == t and images_lab.shape[2] == 3

    images_lab = images_lab.unsqueeze(2)

    images_lab_sim_neighs = []
    offset = random.randint(0,t-tube_len) # rand between [0, t-self.tube_len]
    for i in range(tube_len):
        # Add similarities between consecutive frames i and i+1. frame t reconnects with frame 0. 
        neigh_lab_sim = torch.cat([
            get_neighbor_images_patch_color_similarity(
                images_lab[b_i, i+offset],
                images_lab[b_i, offset+((i+1)%tube_len)],
                3, 3,  # TODO: dilation is never used, very fragile hardcoded and has to match in other functions
                theta=theta
            ) 
            for b_i in range(b)]
        )

        # for every img/obj I want top k correspondences in other image
        neigh_lab_sim_top = topk_mask(neigh_lab_sim, k=topk)

        # B*n_obj x K2 x H x W
        neigh_lab_sim_top = neigh_lab_sim_top.repeat_interleave(num_obj, dim=0)
        images_lab_sim_neighs.append(neigh_lab_sim_top)
    
    return images_lab_sim_neighs, offset


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
    # This uses the Frobenius norm
    return similarity 


def get_neighbor_images_color_similarity(images, images_neighbor, kernel_size, dilation, theta=0.5):
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
    similarity = torch.exp(-torch.norm(diff, dim=1) * theta)

    return similarity

def get_neighbor_images_patch_color_similarity(images, images_neighbor, kernel_size, dilation, theta):
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
    
    return get_neighbor_images_color_similarity(
        unfolded_images, 
        unfolded_images_neighbor,
        3, 3, theta=theta
        )

def topk_mask(images_lab_sim, k):
    """
        Return images_lab_sim with everything zerod out
        except for the top k entries in dimension 1.
        [[0.,  4.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 28., 9., 10., 1.]]
        ----------------------->
        [[0.,  4.,  0.,  3.],
        [ 0.,  0.,  6.,  7.],
        [ 28.,  0., 10., 0.]]
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
