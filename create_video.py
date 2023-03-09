import cv2
import numpy as np
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
import os
from PIL import Image


counter = 0
compare = True
dim=0

for i, category in enumerate(['pigs']):
    result = 'd17_both_scratch_NOCROP_warmup_s3_50k'
    result2 = 'd17_bcedice_scratch_NOCROP_s3_50k'

    path_mask = f'./results/{result}/{category}/'
    path_mask2 = f'./results/{result2}/{category}/'
    path_img = f'../DAVIS/2017/trainval/JPEGImages/480p/{category}/'

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'avc1')
    if i == 0:
        with Image.open(path_mask+'00000.png') as m:
            size = m.size
            if compare:
                size = (2*size[0], size[1]) if dim==1 else (size[0], 2*size[1])
            # size is passed in the reverse way (WxH)
            writer = cv2.VideoWriter(f"videos/{result}-{category}-COMP0.avi",fourcc, 6, size)


    for mask in os.listdir(path_mask):
        counter += 1
        fname = mask.split('.')
        with Image.open(os.path.join(path_mask, mask)) as m, Image.open(os.path.join(path_img, fname[0]+'.jpg')) as im:
            frame = np.asarray(im)
            m = np.asarray(m)
            visualization = overlay_davis(frame, m) 

        if compare:
            with Image.open(os.path.join(path_mask2, mask)) as m2: 
                m2 = np.asarray(m2)
                visualization2 = overlay_davis(frame, m2) 
                visualization = np.concatenate([visualization, visualization2], axis=dim)

        writer.write(visualization.astype('uint8'))

print(f'Wrote video with {counter} frames')

writer.release()

# The codec we want is not in cv2 pip version due to licensing
#ffmpeg -i remote2.avi -c:v h264 out2.mp4
