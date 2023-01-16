import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict


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

def projected_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i] # is now BxHxW
        # background not in mask, so we add one to cls_gt
        # extract gt for object 'i'
        gt = (cls_gt==(i+1)).float() # BxHxW
        #gt = cls_gt

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

def test_loss():
    m = torch.ones((4,1,5,5))
    gt = m[:,0] == 1
    m[1,0,3,4] = 1
    m[1,0,:,1] = 1
    m[:,0,2,:] = 1
    

    #gt[3,3,:] =0

    l = projected_loss(m, gt)
    print(l)

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
        """
        losses = defaultdict(int)

        # get batch b, and num_frames t
        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0], it)
                losses['p'] += p / b / (t-1)
                losses[f'ce_loss_{ti}'] += loss / b

            losses['total_loss'] += losses['ce_loss_%d'%ti]
            losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:,ti,0])
            losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses
