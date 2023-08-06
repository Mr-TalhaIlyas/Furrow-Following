# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:54:50 2023

@author: talha
"""

from fmutils import fmutils as fmu 
import cv2
import numpy as np
from forrows2midline import get_median_line
from pathlib import Path
from tqdm import trange
from gray2color import gray2color

g2c = lambda x: gray2color(x, use_pallet='pannuke')
split = 'train'

data_dir = 'C:/Users/talha/Desktop/ibrahim/data/'
op_dir = data_dir.replace('data', 'data_new')
viz_dir = 'C:/Users/talha/Desktop/ibrahim/data_v3/viz/'

depth = Path(data_dir, 'depth', split)
rgb = Path(data_dir, 'rgb', split)
gt = Path(data_dir, 'gt', split)
seg = Path(data_dir, 'seg', split)

op_detph = str(depth).replace('data','data_v3')
op_rgb = str(rgb).replace('data','data_v3')
op_gt = str(gt).replace('data','data_v3')
op_seg = str(seg).replace('data','data_v3')

depth_files = fmu.get_all_files(depth)
rgb_files = fmu.get_all_files(rgb)
gt_files = fmu.get_all_files(gt)
seg_files = fmu.get_all_files(seg)

# i = 10
for i in trange(len(depth_files), total=len(depth_files)):
    
    dfile = cv2.imread(depth_files[i], -1)
    rfile = cv2.imread(rgb_files[i], -1)
    gtfile = cv2.imread(gt_files[i], 0)
    
    filename = str(Path(depth_files[i]).stem)
    
    # processing the gt furrow segments
    gtfile = gtfile[50:-50, 50:-50]
    gtfileo = get_median_line(gtfile, resizer=2, dilate=True)
    gtfileo = gtfileo[50:-50, 50:-50]
    gtc = g2c(gtfileo)
    
    dfile = dfile[100:-100, 100:-100]
    rfile = rfile[100:-100, 100:-100]
    gtfile = gtfile[50:-50, 50:-50] # 50 alread done
    
    overlay1 = cv2.addWeighted(rfile, 0.8, 
                               cv2.cvtColor(gtfile, cv2.COLOR_GRAY2RGB),
                               0.2, 0)
    overlay2 = cv2.addWeighted(overlay1, 0.5, gtc, 0.5, 0)
    
    
    cv2.imwrite(f'{op_rgb}/{filename}.png', rfile)
    cv2.imwrite(f'{op_detph}/{filename}.png', dfile)
    cv2.imwrite(f'{op_gt}/{filename}.png', (gtfileo*255).astype(np.uint8))
    cv2.imwrite(f'{op_seg}/{filename}.png', gtfile)
    cv2.imwrite(f'{viz_dir}/{filename}.png', overlay2)

    # break
