import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import masks_to_boxes
import torchvision.transforms as transforms

from collections import defaultdict

from skimage import color
import cv2
import numpy as np


def dice_loss(input_mask, cls_gt):
    """
        input_mask: 4x3x384x384 (B x num_obj x H x W)
        cls_gt:     4x384x384

        dice_loss is calculated per class, each loss is also
        over batch dimension. 
        Finally, the losses for each class is averaged.
    """
    #return projected_loss(input_mask, cls_gt)
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()

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

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    assert torch.isclose(-torch.logexpsum(log_same_fg_prob, log_same_bg_prob)[:,0], -log_same_prob[:,0])
    # loss = -log(prob)
    return -log_same_prob[:, 0]

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
    assert (x<=1.).all() and (x>=0.).all(), 'probability not in [0,1]'
    assert (target<=1.).all() and (target>=0.).all(), 'probability not in [0,1]'
    assert ((target == 1.) | (target == 0.)).all(), 'mask not 0,1'

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


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p



class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])

        self.color_threshold = 0.3
        self.col_thresh_neigh = 0.05
        
        # TODO: use instead of hardcoded. not used right now
        self.kernel_size = 3
        self.pairwise_dilation = 2
        self.iter = 0
        self.warmup_iters = 10_000

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
        images_lab_sim = images_lab_sim.reshape(b, t, *images_lab_sim.shape[-3:])
        # TODO: remove hardcoded 3, should be num_obj
        # N*T x K2-1 x H x W
        images_lab_sim = images_lab_sim.repeat_interleave(3, dim=0).flatten(0,1)

        
        # list of len t, B x 1 x K^2 x H x W
        images_lab_sim_neighs = []
        for i in range(t):
            # Add cyclic neighbors between consecutive frames ii and ii+1. frame t reconnects with frame 0. 
            neigh_lab_sim = torch.cat(
                [get_neighbor_images_patch_color_similarity(images_lab[ii+i], images_lab[ii+(i+1)%t], 3, 3) 
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
        for i in range(t):
            pairwise_logprob_neighbor.append(
                compute_pairwise_term_neighbor(
                    logits[:,i:i+1], logits[:,(i+1)%t:1+(i+1)%t], k_size, 3
                ) 
            )
        
        # N x T x H x W -> N x 1 x H x W (since we sum over T dimension and keepdim)
        bbox_time_sum = (bboxes.sum(dim=1, keepdim=True) >= 1.).float()

        # Finally, compute neighbor-independent pairwise loss and projection loss
        # logits: NT x 1 x H x W
        logits = logits.flatten(0,1)[:, None]
        bboxes = bboxes.flatten(0,1)[:, None]

        # Calculate the time-independent pairwise and projection loss
        loss_projection = compute_project_term(logits.sigmoid(), bboxes)  
        pairwise_logprobs = compute_pairwise_term(logits, self.kernel_size, 2)
       
        weights = (images_lab_sim >= self.color_threshold).float() * target_masks.float()
        loss_pairwise = (pairwise_logprobs* weights).sum() / weights.sum().clamp(min=1.0) 

        # Calculate weights for neighbors, and then losses using weights.
        for i in range(t):
            weight_neigh = (images_lab_sim_neighs[i] >= self.col_thresh_neigh).float() * bbox_time_sum
            losses[f'neigh_loss_{i}'] = (pairwise_logprob_neighbor[i] * weight_neigh).sum() / weight_neigh.sum().clamp(min=1.0)

        lambda_ = 1.
        warmup_factor = min(1., train_iter / self.warmup_iters)
        losses[f'proj_loss'] = loss_projection 
        losses[f'pair_loss'] = loss_pairwise
        losses['total_loss'] += (losses[f'proj_loss_{ti}'] + warmup_factor * losses[f'pair_loss_{ti}']
                                + lambda_ * warmup_factor * sum(losses[f'neigh_loss_{i}'] for i in range(t)))


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
        self.knn_loss(data, mask_to_bbox(data['cls_gt'][:, 1:], no), b, t-1, it)

        losses['total_loss'] = 0
        for ti in range(1, t):
            images_rgb = 255.*self.inv_im_trans(data['rgb'][:,ti]) # B x 3 x H x W
            logits = data[f'logits_{ti}'][:, 1:] # remove first which is bg
            # B x n_obj x H x W --> B*n_obj x 1 x H x W
            source_logits = logits.flatten(0,1)[:, None]
            num_objects = logits.shape[1] # (This is always 4 (max 3 fg objs and bg))
            target_masks = data['cls_gt'][:, ti, 0] # B x H x W

            # list of len B with (1 * C * H * W)
            images_lab = [torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                         device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)[None]
                            for rgb_img in images_rgb]

            # (B x K^2-1 x H x W)
            images_lab_similarity = torch.cat([get_images_color_similarity(img_lab) for img_lab in images_lab])
            # want images_lab_sim to be (B*num_obj x K^2-1 x H x W)
            images_lab_similarity = images_lab_similarity.repeat_interleave(num_objects, dim=0)

            # note that we count bg as a objected now
            # B*n_obj x 1 x H x W
            #target_masks = torch.stack([m==i for m in target_masks for i in range(num_objects)])[:, None]
            masks = []
            bboxes = []
            obj_exist_mask = []
            for bn, m in enumerate(target_masks):
                for i in range(num_objects): 
                    mask = (m==(i+1)).float()
                    bbox = torch.zeros_like(mask)

                    if mask.any():
                        x1,y1,x2,y2 = masks_to_boxes(mask[None]).long()[0]
                        bbox[y1:y2+1, x1:x2+1] = 1.
                        obj_exist_mask.append(True)
                    else:
                        bbox = mask
                        obj_exist_mask.append(False)
                    masks.append(mask)
                    bboxes.append(bbox)
            
            # size B*n_obj. if there is an object in that entry
            # all-zero entries affect the dice loss for example 
            obj_exist_mask = torch.tensor(obj_exist_mask)
            target_masks = torch.stack(masks)[:,None]
            target_bboxes = torch.stack(bboxes)[:,None]
            
            # B*n_obj x K^2-1 x H x W
            pairwise_loss = compute_pairwise_term(source_logits)

            # Use the lab_similarity to mask pixels that are similar
            # target_masks is the gt mask
            weights = (images_lab_similarity >= self.color_threshold).float() * target_bboxes.float()
            loss_pairwise = (pairwise_loss * weights).sum() / weights.sum().clamp(min=1.0)
            loss_projection = compute_project_term(source_logits.sigmoid(), target_bboxes)

#            assert loss_projection <= 2 and loss_projection >= 0, f'proj_loss is more than 2: {loss_projection}'
            losses[f'proj_loss_{ti}'] = loss_projection 
            losses[f'pair_loss_{ti}'] = loss_pairwise

            warmup_factor = min(1., self.iter / self.warmup_iters)
#            warmup_factor = 1
            losses['total_loss'] += losses[f'proj_loss_{ti}']# + warmup_factor * losses[f'pair_loss_{ti}'] 

#        with torch.no_grad():
#            if (self.iter % 10) == 0:
#                pred_mask = data[f'masks_{1}'][0,0]
#                gt_mask = data['cls_gt'][0, 1, 0]
#                gt_box = target_bboxes[0,0]
#                img = torch.cat([pred_mask, gt_mask, gt_box],dim=0)
#                cv2.imwrite(f'vis_mask_check/pred_mask.png',img.repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)

   #         b_save = 0
   #         # save the predicted mask for batch 'b_save'
   #         cv2.imwrite(f'vis_mask_check/pred_mask.png',data['masks_1'].amax(dim=1)[b_save][None]
   #                     .repeat(3,1,1).permute(1,2,0).float().detach().cpu().numpy()*255)
   #         cv2.imwrite(f'vis_mask_check/rgb.png',images_rgb[b_save].permute(1,2,0).float().cpu().numpy())
   #         cv2.imwrite(f'vis_mask_check/lab.png',images_lab[b_save][0].permute(1,2,0).float().cpu().numpy())
   #         for i, w in enumerate(weights[num_objects*b_save:num_objects*b_save+num_objects]):
   #             w = w[0].repeat(3,1,1) # only one of the 8 consisteny maps
   #             cv2.imwrite(f'vis_mask_check/{i}_image.png',w.permute(1,2,0).float().cpu().numpy() * 255)
   #             cv2.imwrite(f'vis_mask_check/{i}_mask.png',target_masks[num_objects*b_save+i].repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)
   #         print()

#            write_imgs(images_lab, data, ti, images_lab_similarity, weights)
     #       print()
            #losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:,ti,0])
            #losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses


def write_imgs(images, data, ti, images_sim, weights):
    for i, img in enumerate(images):
#                img2 = cv2.cvtColor(img.permute(1,2,0).float().cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
        img = img.squeeze(0)
        mask = data['cls_gt'][i, ti,0][None].repeat(3,1,1)
        sim = images_sim[4*i].repeat(3,1,1)
        w = weights[4*i].repeat(3,1,1)
        img_right = torch.cat([sim, w], dim=1)
        img_left = torch.cat([img, mask], dim=1)

        img = torch.cat([img_left, img_right], dim=2)

        print(f'concat img shape: {img.shape}')
        cv2.imwrite(f'vis_mask_check/{i}_image.png',img.permute(1,2,0).float().cpu().numpy() * 255)
#        cv2.imwrite(f'vis_mask_check/{i}_mask.png',mask.permute(1,2,0).float().cpu().numpy() * 255)
    
    exit(1)


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
    gt_mask = gt_mask.permute(1,0,2,3,4)

    obj_exist_mask = []
    masks = []
    bboxes = []
    for t, m_t in enumerate(gt_mask):
        for bn, m in enumerate(m_t):
            for i in range(num_objects): 
                mask = (m==(i+1)).float()
                bbox = torch.zeros_like(mask)

                if mask.any():
                    x1,y1,x2,y2 = masks_to_boxes(mask).long()[0]
                    bbox[y1:y2+1, x1:x2+1] = 1.
                    obj_exist_mask.append(True)
                else:
                    bbox = mask
                    obj_exist_mask.append(False)
                masks.append(mask)
                bboxes.append(bbox)
    
    # size B*n_obj. if there is an object in that entry
    # all-zero entries affect the dice loss for example 
    obj_exist_mask = torch.tensor(obj_exist_mask)
    target_masks = torch.stack(masks)
    target_bboxes = torch.stack(bboxes)

    return target_bboxes.reshape(-1, gt_mask.shape[0], gt_mask.shape[-2], gt_mask.shape[-1])
            
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

class LossComputer2:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])
        self.count = 0

    def compute(self, data, num_objects, it):
        self.count += 1
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        for ti in range(1, t):
            target_masks = data['cls_gt'][:,ti,0]
            masks = []
            bboxes = []
            for bn, m in enumerate(target_masks):
                for i in range(1): 
                    mask = (m==(i+1)).float()
                    bbox = torch.zeros_like(mask)

                    if mask.any():
                        x1,y1,x2,y2 = masks_to_boxes(mask[None]).long()[0]
                        bbox[y1:y2+1, x1:x2+1] = 1.
                    else:
                        bbox = mask
                    masks.append(mask)
                    bboxes.append(bbox)
            
            target_masks = torch.stack(masks)[:,None]
            target_bboxes = torch.stack(bboxes)[:,None]
            
            #losses[f'dice_loss_{ti}'] = compute_project_term(data[f'masks_{ti}'], target_bboxes)
            # using logits instead
            losses[f'dice_loss_{ti}'] = compute_project_term(data[f'logits_{ti}'][:,1:].sigmoid(), target_bboxes)
           # losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:,ti,0])
            losses['total_loss'] += losses[f'dice_loss_{ti}']

        if (self.count % 10) == 0:
            pred_mask = data[f'masks_{ti}'][0,0]
            gt_mask = data['cls_gt'][0, ti, 0]
            gt_box = target_bboxes[0,0]
            img = torch.cat([pred_mask, gt_mask, gt_box],dim=0)
            cv2.imwrite(f'vis_mask_check/pred_mask.png',img.repeat(3,1,1).permute(1,2,0).float().cpu().detach().numpy() * 255)
        return losses