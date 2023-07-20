#%%
import os
# os.chdir(os.path.dirname(__file__))
os.chdir("/home/user01/data/furrow_line/scripts/")

import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')
    
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300


from data.dataloader import GEN_DATA_LISTS, Furrow
from data.utils import collate

from core.model import FurrowDet
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import Trainer, Evaluator, ModelUtils, eval_wrapper
import torch.nn.functional as F

from skimage.measure import label as sml
import matplotlib.cm as cm
import seaborn as sns
from gray2color import gray2color
# Get the "tab10" palette
palette = sns.color_palette("tab10")
palette = np.array(palette)
palette = palette[np.newaxis, :, :]
palette = np.insert(palette, 0, [0, 0, 0], axis=1) # insert balck color for BG
g2c = lambda x : gray2color(x.astype(np.uint8), use_pallet='pannuke', custom_pallet=palette)

data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
train_paths, test_paths = data_lists.get_splits()
classes = data_lists.get_classes()
data_lists.get_filecounts()


train_data = Furrow(train_paths[0], train_paths[1], train_paths[2], train_paths[3],
                    config['img_height'], config['img_width'],
                    config['Augment_data'], config['Normalize_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'], drop_last=True, # important for adaptive augmentation to work properly.
                          collate_fn=collate, pin_memory=config['pin_memory'],
                          prefetch_factor=3, persistent_workers=True)

test_data = Furrow(test_paths[0], test_paths[1],  test_paths[2], test_paths[3],
                       config['img_height'], config['img_width'],
                       False, config['Normalize_data'])

test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=3, persistent_workers=True)

if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    s=255
    if config['batch_size'] > 1:
        img_ls = []
        [img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(2)]
        [img_ls.append((batch['depth'][i]*s).astype(np.uint8)) for i in range(2)]
        [img_ls.append(g2c(sml(batch['f_line'][i]))) for i in range(2)]
        [img_ls.append(g2c(sml(batch['f_seg'][i]))) for i in range(2)]
        plt.title('Sample Batch')
        plt.imshow(imgviz.tile(img_ls, shape=(2,4), border=(255,0,0)))
        plt.axis('off')
    else:
        plt.title('Sample Batch')
        plt.imshow(imgviz.tile([(batch['img'][0]*s).astype(np.uint8), g2c(batch['lbl'][0])]
                               , border=(255,0,0)))

#%%
model = FurrowDet(num_classes=config['num_classes'])
                
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # print(torch._dynamo.list_backends())
    model = torch.compile(model, mode="max-autotune")

optimizer = torch.optim.Adam([{'params': model.parameters(),
                               'lr':config['learning_rate']}],
                               weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

metric_seg = ConfusionMatrix(config['num_classes'])
metric_line = ConfusionMatrix(config['num_classes'])

mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
mu.get_model_profile(model, False)
# mu.load_chkpt(model, optimizer=None)
# mu.load_pretrained_chkpt(model, "/home/user01/data/talha/CWD26/pretrained/cityscape.pth")

trainer = Trainer(model, config['batch_size'], optimizer, metric_seg, metric_line)
evaluator = Evaluator(model, metric_seg, metric_line)

# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"val_SegmIOU": 0, "val_LinmIOU": 0,
               "LinemIOU": 0, "SegmIOU": 0,
               "loss": 10, "learning_rate": 0}, step=0)

#%%
start_epoch = 0
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_siou, total_avg_liou = [], []
for epoch in range(start_epoch, config['epochs']):
    epoch 
    pbar = tqdm(train_loader)
    model.train() # <-set mode important
    ts, tl, tloss = [], [], []
    for step, data_batch in enumerate(pbar):

        scheduler(optimizer, step, epoch)
        loss_value = trainer.training_step(data_batch)
        siou, liou = trainer.get_scores()
        trainer.reset_metric()
        
        tloss.append(loss_value)
        ts.append(siou['iou_mean'])
        tl.append(liou['iou_mean'])
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - SegmIOU {siou["iou_mean"]:.4f} - LinmIOU {liou["iou_mean"]:.4f}')
    print(f'=> Average loss: {np.nanmean(tloss):.4f}, Average SegIoU: {np.nanmean(ts):.4f}, Average LineIoU: {np.nanmean(tl):.4f}')
    g, n = data_batch['geo_augs'][0], data_batch['noise_augs'][0]

    if (epoch + 1) % 2 == 0: # eval every 2 epoch
        curr_viou, avg_siou, avg_liou, total_avg_siou, total_avg_liou, tiled = eval_wrapper(evaluator, model, test_loader, total_avg_siou, total_avg_liou)
        
        cprint(f'=> Averaged SegIoU: {avg_siou:.4f}::Averaged LineIoU: {avg_liou:.4f}', 'magenta')

        if config['LOG_WANDB']:
            wandb.log({"val_SegmIOU": avg_siou, "val_LinmIOU":avg_liou}, step=epoch+1)
            wandb.log({'predictions': wandb.Image(tiled)}, step=epoch+1)

    if config['LOG_WANDB']:
        wandb.log({"loss": loss_value, 
                   "SegmIOU": np.nanmean(ts),"LinemIOU": np.nanmean(tl),
                   "learning_rate": optimizer.param_groups[0]['lr'],
                   'geo_augs': g, 'noise_augs': n}, step=epoch+1)
    
    if curr_viou > best_iou:
        best_iou = curr_viou
        mu.save_chkpt(model, optimizer, epoch, loss_value, best_iou)

mu.save_chkpt(model, optimizer, epoch, loss_value, best_iou)
if config['LOG_WANDB']:
    wandb.run.finish()
#%%


