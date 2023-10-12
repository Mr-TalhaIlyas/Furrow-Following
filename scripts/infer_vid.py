# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:43:41 2023

@author: talha
"""

#%%
import os
# os.chdir(os.path.dirname(__file__))
os.chdir("D:/RV/Ibrahim/video/Furrow-Following/scripts/")
from pickletools import optimize
import yaml
with open('infer_config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
               config_include_keys=config.keys(), config=config)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tabulate import tabulate

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm
mpl.rcParams['figure.dpi'] = 300

from data.dataloader import GEN_DATA_LISTS, Furrow
from data.utils import collate, images_transform, torch_resizer, masks_transform

from core.model import FurrowDet

from metrics import ConfusionMatrix

from utils import ModelUtils
import torch.nn.functional as F
from fmutils import fmutils as fmu
from empatches import EMPatches
from data.utils import std_norm
from gray2color import gray2color

from skimage.measure import label as sml
import seaborn as sns
from collections import deque

def average_predictions(pred_queue, new_pred, n_frames):
    """
    Add new prediction to the queue, remove oldest if necessary.
    Return the average of predictions in the queue.
    """
    pred_queue.append(new_pred)
    if len(pred_queue) > n_frames:
        pred_queue.popleft()
    
    return np.mean(np.array(pred_queue), axis=0)

palette = sns.color_palette("tab10")
palette = np.array(palette)
palette = palette[np.newaxis, :, :]
palette = np.insert(palette, 0, [0, 0, 0], axis=1) # insert balck color for BG
g2s = lambda x : gray2color(x.astype(np.uint8), use_pallet='pannuke', custom_pallet=palette)
g2l = lambda x : gray2color(x.astype(np.uint8), use_pallet='pannuke', custom_pallet=palette[:,:,::-1])
def overlay_mask(image, mask):
    # Ensure the mask is binary such that the segmentation part is non-zero and the rest is zero
    binary_mask = np.all(mask == [0, 0, 0], axis=-1)

    # Create a new mask where the segmented part is the mask and the rest is the image
    overlayed_image = np.where(binary_mask[..., None], image, mask)

    return overlayed_image

normalize = lambda x, alpha, beta : (((beta-alpha) * (x-np.min(x))) / (np.max(x)-np.min(x))) + alpha
standardize = lambda x : (x - np.mean(x)) / np.std(x)

def std_norm(img, norm=True, alpha=0, beta=1):
    '''
    Standardize and Normalizae data sample wise
    alpha -> -1 or 0 lower bound
    beta -> 1 upper bound
    '''
    img = standardize(img)
    if norm:
        img = normalize(img, alpha, beta)
        
    return img
from torchvision import transforms
transformer = transforms.Compose([
                                 # this transfrom converts BHWC -> BCHW and 
                                 # also divides the image by 255 by default if values are in range 0..255.
                                 transforms.ToTensor(),
                                ])
#%%
model = FurrowDet(num_classes=config['num_classes'])
                
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
# mu.load_chkpt(model)
mu.load_pretrained_chkpt(model, 'D:/RV/Ibrahim/video/Furrow-Following/bagfile_model_weight/Exp_6_talha_epoch99.pth')

#%%

rgb_vid = "D:/RV/Ibrahim/video/Furrow-Following/output.mp4"
depth_vid = 'D:/RV/Ibrahim/video/Furrow-Following/doutput.mp4'
rcap = cv2.VideoCapture(rgb_vid)
dcap = cv2.VideoCapture(depth_vid)


# Get video properties to set up the video writer
fps = int(rcap.get(cv2.CAP_PROP_FPS))
width = int(rcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(rcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/RV/Ibrahim/video/Furrow-Following/pred.avi', fourcc, fps, (width, height))
i=0
pred_queue = deque()
# Loop through the video frames
while rcap.isOpened() and dcap.isOpened:
    # Read a frame from the video
    rsuccess, rframe = rcap.read()
    dsuccess, dframe = dcap.read()

    if rsuccess and dsuccess:
        rframe = cv2.cvtColor(rframe, cv2.COLOR_BGR2RGB)
        rframe = cv2.resize(rframe, (768, 384), interpolation=cv2.INTER_LINEAR)
        img = std_norm(rframe)
        dframe = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
        dframe = cv2.resize(dframe, (768, 384), interpolation=cv2.INTER_LINEAR)
        dframe = dframe / 255
        rframe = transformer(rframe).float().to('cuda' if torch.cuda.is_available() else 'cpu')[None, ...]
        dframe = transformer(dframe).float().to('cuda' if torch.cuda.is_available() else 'cpu')[None, ...]
        
        model.eval() # <-set mode important
        with torch.no_grad():
            _, preds = model.forward([rframe, dframe])
        
        preds_seg = preds['out_seg'].argmax(1)
        preds_seg = preds_seg.cpu().numpy().squeeze()
    
        preds_line = preds['out_line'].argmax(1)
        preds_line = preds_line.cpu().numpy().squeeze()
        
        # preds_line = sml(preds_line)
        
        # Average the line predictions over the last n frames
        averaged_line = average_predictions(pred_queue, preds_line, 10)
        # Display the annotated frame
        cv2.imshow("oking", averaged_line)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        out.write((averaged_line*255).astype(np.uint8))
    i+=1
    print(i)



rcap.release()
dcap.release()
out.release()
cv2.destroyAllWindows()

#%%
   