import cv2
import numpy as np
from inference.interact.interactive_utils import overlay_davis
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import pandas as pd

def video(result, result2, mp4_out_name, compare=True, dim=0, topk=5, have_text=False):
    """
        dim=0 means concat along y axis
        compare=True means to compare result and result2
    """
    counter = 0
    categories,scores = get_diff_classes(result, result2, topk=topk)
    print(categories)
    size = (854, 480)
    #categories = ['pigs', 'camel', 'blackswan']
    for i, category in enumerate(categories):
        # when compare==True result will be on top/left.
        J, F, J2, F2 = scores[i]
        path_mask = f'./results/{result}/{category}/'
        path_mask2 = f'./results/{result2}/{category}/'
        path_img = f'../DAVIS/2017/trainval/JPEGImages/480p/{category}/'

        if i == 0:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #fourcc = cv2.VideoWriter_fourcc(*'avc1')
            with Image.open(path_mask+'00000.png') as m:
                if compare:
                    full_size = (2*size[0], size[1]) if dim==1 else (size[0], 2*size[1])
                # size is passed in the reverse way (WxH)
                writer = cv2.VideoWriter(f"videos/{result}.avi",fourcc, 7, full_size)

#        font = ImageFont.truetype("FreeSans.ttf", 22)
        for mask in sorted(os.listdir(path_mask)):
            fname = mask.split('.')
            with Image.open(os.path.join(path_mask, mask)) as m, Image.open(os.path.join(path_img, fname[0]+'.jpg')) as im:
                if m.size != size:
                    m = m.resize(size)
                    im = im.resize(size)
                im2 = im.copy()
                draw = ImageDraw.Draw(im)
                #draw.text((0, 0), f'{result[-25:]} J&F:{0.5*(J+F):0.3f}',(255,255,255),font=font)
                frame = np.asarray(im)
                m = np.asarray(m)
                visualization = overlay_davis(frame, m) 

            if compare:
                with Image.open(os.path.join(path_mask2, mask)) as m2: 
                    draw = ImageDraw.Draw(im2)
#                    draw.text((0, 0), f'{result2[-25:]} J&F:{0.5*(J2+F2):0.3f}',(255,255,255),font=font)
                    frame2 = np.asarray(im2)

                    if m2.size != size:
                        m2 = m2.resize(size)
                    m2 = np.asarray(m2)
                    visualization2 = overlay_davis(frame2, m2) 

                    visualization = np.concatenate([visualization, visualization2], axis=dim)
                    # convert from BGR to RGB
                    visualization = visualization[..., ::-1].copy()

            counter += 1
            writer.write(visualization.astype('uint8'))

    print(f'Wrote video with {counter} frames')

    writer.release()
    #os.system(f'/usr/bin/ffmpeg -i videos/{result}.avi -c:v h264 videos/{mp4_out_name}.mp4  -hide_banner -loglevel error')
    #os.system(f'rm videos/{result}.avi')

# The codec we want is not in cv2 pip version due to licensing
# conda's ffmpeg uses libopenh264(does not work) instead of libx264
# /usr/bin/ffmpeg -i in.avi -c:v h264 out.mp4

def get_diff_classes(base, comp, topk=10):
    """
        Gets the categories in which the scores vary the most
        between models base and comp.
    """
    base_path = f'results/{base}/per-sequence_results-val.csv'
    comp_path = f'results/{comp}/per-sequence_results-val.csv'
    base_df = pd.read_csv(base_path)
    comp_df = pd.read_csv(comp_path)

    base_df['J-Diff'] = base_df['J-Mean']-comp_df['J-Mean']
    base_df['F-Diff'] = base_df['F-Mean']-comp_df['F-Mean']
    base_df['Diff-Mean'] = 0.5*(base_df['J-Diff']+base_df['F-Diff'])
    base_df['Diff-Mean-Abs'] = abs(base_df['Diff-Mean'])
    base_df['J-Mean2'] = comp_df['J-Mean']
    base_df['F-Mean2'] = comp_df['F-Mean']

    sort_key = 'Diff-Mean'
    sorted_df = base_df.sort_values(by=[sort_key], ascending=False)
    print(sorted_df.head(10))
    categories = sorted_df[['Sequence', 'J-Mean', 'F-Mean', 'J-Mean2', 'F-Mean2']]
    cleaned_names = []
    scores = []
    for _, c, j, f, j2,f2 in categories.itertuples():
        scores.append((j,f, j2, f2))
        if c[-2] == '_':
            if c[:-2] not in cleaned_names:
                cleaned_names.append(c[:-2])
        else:
            if c not in cleaned_names:
                cleaned_names.append(c)
    return cleaned_names[:topk], scores[:topk]


if __name__=='__main__':
    print('Running as script')
    mp4_out_name = 'all_nosmooth_vs_all_smooth10'
    
#    mp4_out_name = 'onlyproj_vs_all_smooth10'
#    result = 'd17/Jun26_00.28.01_only_proj_kornia_s3_110000'
    result = 'd17/Jun28_17.08.08_all_nosmooth_detach_thres001_s3_110000'
    result2 = 'd17/Jul01_03.04.12_all_thres001_smoothing_s3_110000'

    video(result, result2, mp4_out_name, topk=10)


