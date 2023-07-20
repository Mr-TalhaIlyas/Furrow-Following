#%%
import os
# os.chdir(os.path.dirname(__file__))
os.chdir("/home/user01/data/furrow_line/scripts/")
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
#%%
model = FurrowDet(num_classes=config['num_classes'])
                
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
# mu.load_chkpt(model)
mu.load_pretrained_chkpt(model, f'{config["checkpoint_path"]}/{config["experiment_name"]}.pth')

#%%
data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
_, test_paths = data_lists.get_splits()
test_data = Furrow(test_paths[0], test_paths[1],  test_paths[2], test_paths[3],
                       config['img_height'], config['img_width'],
                       False, config['Normalize_data'])

test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=3, persistent_workers=True)
try:
    os.mkdir(f'{config["predictions_dir"]}{config["experiment_name"]}')
    print(f'Made directory for preds {config["predictions_dir"]}{config["experiment_name"]}')
except FileExistsError:
    pass

pbar = tqdm(test_loader)
ts, tl, tloss = [], [], []
for step, data_batch in enumerate(pbar):

    img_batch = images_transform(data_batch['img'])
    depth_batch = images_transform(data_batch['depth'])

    seg_batch = data_batch['f_seg']#torch_resizer(masks_transform(data_batch['f_seg']))
    line_batch = data_batch['f_line']#torch_resizer(masks_transform(data_batch['f_line']))
    
    model.eval() # <-set mode important
    with torch.no_grad():
        _, preds = model.forward([img_batch, depth_batch])
    
    preds_seg = preds['out_seg'].argmax(1)
    preds_seg = preds_seg.cpu().numpy().squeeze()

    preds_line = preds['out_line'].argmax(1)
    preds_line = preds_line.cpu().numpy().squeeze()

    # Eroding lines
    # preds_line = cv2.erode(preds_line.astype(np.uint8),
    #               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
    #             #   np.ones((3,3),np.uint8),
    #               iterations = 1)

    preds_line = sml(preds_line)
    preds_seg = sml(preds_seg)

    try:
        pred_seg = cv2.resize(g2s(preds_seg), (config['img_width'],config['img_height']), interpolation=cv2.INTER_NEAREST)
        pred_line = cv2.resize(g2l(preds_line), (config['img_width'],config['img_height']), interpolation=cv2.INTER_NEAREST)
    except:
        pred_seg = cv2.resize((preds_seg*255).astype(np.uint8), (config['img_width'],config['img_height']), interpolation=cv2.INTER_NEAREST)
        pred_line = cv2.resize((preds_line*255).astype(np.uint8), (config['img_width'],config['img_height']), interpolation=cv2.INTER_NEAREST)
        pred_seg = cv2.cvtColor(pred_seg, cv2.COLOR_GRAY2BGR)
        pred_line = cv2.cvtColor(pred_line, cv2.COLOR_GRAY2BGR)

    # overlay1 = cv2.addWeighted((data_batch['img'][0]*255).astype(np.uint8), 0.5,
    #                            preds_seg, 0.5,0)
    # overlay = cv2.addWeighted(overlay1, 0.5, preds_line, 0.5,0)
    overlay1 = overlay_mask((data_batch['img'][0]*255).astype(np.uint8), pred_seg)# pred_seg here
    overlay = overlay_mask(overlay1, pred_line)

    # tiles = imgviz.tile([], shape=(1,2), border=(255,0,0))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{config["predictions_dir"]}{config["experiment_name"]}/{step}.png', overlay)
    # break
