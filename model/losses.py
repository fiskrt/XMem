import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import masks_to_boxes

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


def pairwise_loss(input_mask, cls_gt):
    """
        Only have to do this for edges e which have atleast one pixel within
        the bounding box.

        TODO: Mask predicted mask so we only sum over edges within bounding box?
        TODO: actual rgb images should be used: first convert to LAB, then do image similarity on every img

        Assume we are doing single object detection. Then just sum the losses over the objects.
    """

    # color thresh 0.3 by default in implementation
    pass


def compute_pairwise_term(mask_logits, pairwise_size=3, pairwise_dilation=2):
    """
        Calc probabilities Pr(y_e=1)=Pr(pixels are the same)
        using input_mask (networks predictions).

        Then calculate y_e with using color similarities.
    """
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    # TODO: why max?
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def projected_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i] # is now BxHxW
        # background not in mask, so we add one to cls_gt
        # extract gt for object 'i'
        gt = (cls_gt==(i+1)).float() # BxHxW

        loss_y= dice_coefficient(
            mask.max(dim=1)[0],
            gt.max(dim=1)[0]
        )

        loss_x= dice_coefficient(
            mask.max(dim=2)[0],
            gt.max(dim=2)[0]
        )
        losses.append(0.5*(loss_x+loss_y))
    return torch.cat(losses).mean()

def dice_coefficient(mask, gt):
    eps = 1
    numerator = 2 * (mask * gt).sum(-1)
    denominator = mask.sum(-1) + gt.sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss

def bs():
    t1 = 0.*torch.ones((4,300))
    t2 = 0.*torch.ones((4,300))
    res = dice_coefficient(t1,t2)
    print(res)
#bs()

def test_loss():
    m = torch.ones((4,1,5,5))
    gt = m[:,0] == 1
    m[1,0,3,4] = 1
    m[1,0,:,1] = 1
    m[:,0,2,:] = 1
    

    #gt[3,3,:] =0

    l = projected_loss(m, gt)
    print(l)

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()

#test_loss()

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

    def compute(self, data, num_objects, it):
        """
            it = the # time this function is called
            num_objects[i] = # objects in frame[i]. i=0...Batch_size-1

            data['rgb'] = (B, 8, 3, H, W) what is 8?
            data['masks_i'] =   B * n_obj * H * W 
            data['logits_i'] =  B * n_obj+1 * H * W (+1 because we treat bg as object)

            data['cls_gt'] B x 8 x 1 x H x W
            data['info]['num_objects'] = tensor of len B (Just the num objects per batch)

            masks [0,1] are the sigmoid of logits (-inf, inf)


        """
        losses = defaultdict(int)

        # get batch b, and num_frames t
        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        for ti in range(1, t):
           # for bi in range(b):
           #     loss, p = self.bce(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0], it)
           #     losses['p'] += p / b / (t-1)
           #     losses[f'ce_loss_{ti}'] += loss / b

           # losses['total_loss'] += losses['ce_loss_%d'%ti]
            images_rgb = data['rgb'][:,ti] # B x 3 x H x W
            logits = data[f'logits_{ti}']
            # B x n_obj x H x W --> B*n_obj x 1 x H x W
            source_logits = logits.flatten(0,1)[:, None]
            num_objects = logits.shape[1]
            target_masks = data['cls_gt'][:, ti, 0] # B x H x W

            # list of len B with (1 * C * H * W)
            images_lab = [torch.as_tensor(color.rgb2lab(rgb_img.byte().permute(1, 2, 0).cpu().numpy()),
                                         device=rgb_img.device, dtype=torch.float32).permute(2, 0, 1)[None]
                            for rgb_img in images_rgb]

            # (B x K^2-1 x H x W)
            images_lab_similarity = torch.cat([get_images_color_similarity(img_lab) for img_lab in images_lab])
            images_lab_similarity = images_lab_similarity.repeat(num_objects, 1, 1, 1)
            # want images_lab_sim to be (B*num_obj x H x W)
            # cat them together

            # note that we count bg as a objected now
            # B*n_obj x 1 x H x W
            #target_masks = torch.stack([m==i for m in target_masks for i in range(num_objects)])[:, None]
            masks = []
            bboxes = []
            images_lab_sim_all = []
            for bn, m in enumerate(target_masks):
                #for i in range(data['info']['num_objects'][bn]):
                for i in range(num_objects):
                    # create both img_lab_sim and target_bboxes here
                    # we can skip objects that does not exist in a batch
                    mask = (m==i).float()
                    bbox = torch.zeros_like(mask)

                    if mask.any():
                        x1,y1,x2,y2 = masks_to_boxes(mask[None]).long()[0]
                        bbox[y1:y2+1, x1:x2+1] = 1.
                    else:
                        bbox = mask
                    masks.append(mask)
                    bboxes.append(bbox)
                    #images_lab_sim_all.append(images_lab_similarity[bn])

            # TODO: insert dim 1?
            target_masks = torch.stack(masks)[:,None]
          #  images_lab_similarity = torch.cat(images_lab_sim_all)
            target_bboxes = torch.stack(bboxes)[:,None]
            
            # TODO: crashes when no object exist, 
            # Convert masks to bboxes
            #target_bboxes = torch.zeros_like(target_masks)
            #for i, (x1,y1,x2,y2) in enumerate(masks_to_boxes(target_masks[:,0]).long()):
            #    target_bboxes[i, 0, y1:y2+1, x1:x2+1] = 1.

            # B*n_obj x K^2-1 x H x W
            pairwise_loss = compute_pairwise_term(source_logits)


            # Use the lab_similarity to mask pixels that are similar
            # target_masks is the gt mask
            weights = (images_lab_similarity >= 0.3).float() * target_bboxes.float()
            loss_pairwise = (pairwise_loss * weights).sum() / weights.sum().clamp(min=1.0)
            
            loss_projection = compute_project_term(source_logits.sigmoid(), target_bboxes)

            losses[f'proj_loss_{ti}'] = loss_projection 
            losses[f'pair_loss_{ti}'] = loss_pairwise
            losses['total_loss'] += losses[f'pair_loss_{ti}'] + losses[f'proj_loss_{ti}']

            cv2.imwrite(f'vis_mask_check/mask.png',target_masks[5].repeat(3,1,1).permute(1,2,0).float().cpu().numpy() * 255)
            for i, w in enumerate(weights[5]):
                w = w.repeat(3,1,1)
                cv2.imwrite(f'vis_mask_check/{i}_image.png',w.permute(1,2,0).float().cpu().numpy() * 255)

#            write_imgs(images_lab, data, ti, images_lab_similarity, weights)

            #del images_lab_similarity
            #del images_lab
            #del source_logits
            del target_masks

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


def unfold_wo_center(x, kernel_size, dilation):
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
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def get_images_color_similarity(images, kernel_size=3, dilation=2):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    return similarity 