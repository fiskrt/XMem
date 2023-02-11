import cv2
import numpy as np
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
import os
from PIL import Image



category = 'bmx-trees'
result = 'd17_onlydice'

path_mask = f'./results/{result}/{category}/'
path_img = f'../DAVIS/2017/trainval/JPEGImages/480p/{category}/'

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'avc1')
with Image.open(path_mask+'00000.png') as m:
    writer = cv2.VideoWriter(f"videos/{result}-{category}.avi",fourcc, 6, m.size)


for mask in os.listdir(path_mask):
    print(mask)
    fname = mask.split('.')
    with Image.open(os.path.join(path_mask, mask)) as m, Image.open(os.path.join(path_img, fname[0]+'.jpg')) as im:
        frame = np.asarray(im)
        m = np.asarray(m)
        visualization = overlay_davis(frame, m) 
        writer.write(visualization.astype('uint8'))

writer.release()

# The codec we want is not in cv2 pip version due to licensing
#ffmpeg -i remote2.avi -c:v h264 out2.mp4
