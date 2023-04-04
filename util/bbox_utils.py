import torch


def mask_to_box(masks):
    """
        Any shape with last two dimension H x W
        and turn into bounding box of same shape.
    """
    assert masks.dim() >= 3
    sh = masks.shape
    masks = masks.flatten(0,-3)

    bounding_boxes = torch.zeros_like(masks)

    for index, mask in enumerate(masks):
        if not mask.any():
            bounding_boxes[index] = mask
        else:
            y, x = torch.where(mask != 0)

            x1 = torch.min(x)
            y1 = torch.min(y)
            x2 = torch.max(x)
            y2 = torch.max(y)
            bounding_boxes[index, y1:y2+1, x1:x2+1] = 1.

    bounding_boxes = bounding_boxes.reshape(*sh)
    return bounding_boxes
