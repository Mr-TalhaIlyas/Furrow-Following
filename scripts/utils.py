import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import math

import cv2, os, imgviz, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.utils import (images_transform, masks_transform, torch_imgresizer,
                        torch_resizer)
import seaborn as sns
from gray2color import gray2color
# Get the "tab10" palette
palette = sns.color_palette("tab10")
palette = np.array(palette)
palette = palette[np.newaxis, :, :]
palette = np.insert(palette, 0, [0, 0, 0], axis=1) # insert balck color for BG
g2c = lambda x : gray2color(x.astype(np.uint8), use_pallet='pannuke', custom_pallet=palette)

class ModelUtils(object):
    def __init__(self, num_classes, chkpt_pth, exp_name):
        self.num_classes = num_classes
        self.chkpt_pth = chkpt_pth
        self.exp_name = exp_name
    
    def save_chkpt(self, model, optimizer, epoch=0, loss=0, iou=0):
        cprint('-> Saving checkpoint', 'green')
        torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'iou': iou,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(self.chkpt_pth, f'{self.exp_name}_epoch{epoch}.pth'))

    def get_model_profile(self, model, summary=False):
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model total params: {total_params/10**6}')
        print(f'Model trainable params: {trainable_params/10**6}')
        if summary:
            from torchinfo import summary
            summary(model, input_size=(config['batch_size'],config['input_channels'], config['img_width'], config['img_height']), depth=2)

    def load_chkpt(self, model, optimizer=None):
        
        try:
            print('-> Loading checkpoint')
            chkpt = torch.load(os.path.join(self.chkpt_pth, f'{self.exp_name}.pth'),
                                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
            epoch = chkpt['epoch']
            loss = chkpt['loss']
            iou = chkpt['iou']
            model.load_state_dict(chkpt['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            print(f'[INFO] Loaded Model checkpoint: epoch={epoch} loss={loss} iou={iou}')
            print(f'loaded path: {os.path.join(self.chkpt_pth, f"{self.exp_name}.pth")}')
        except FileNotFoundError:
            print('[INFO] No checkpoint found')
        except RuntimeError:
            print('[Error] Pretrained dict dont match')
    
    def load_pretrained_chkpt(self, model, pretrained_path=None):
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path,
                               map_location='cuda' if torch.cuda.is_available() else 'cpu')
            try:
                # load pretrained
                pretrained_dict = chkpt['model_state_dict']
                # load model state dict
                state = model.state_dict()
                # loop over both dicts and make a new dict where name and the shape of new state match
                # with the pretrained state dict.
                matched, unmatched = [], []
                new_dict = {}
                for i, j in zip(pretrained_dict.items(), state.items()):
                    pk, pv = i # pretrained state dictionary
                    nk, nv = j # new state dictionary
                    # if name and weight shape are same
                    if pk.strip('_orig_mod.module.') == nk.strip('_orig_mod.module.') and pv.shape == nv.shape:
                        new_dict[nk] = pv
                        matched.append(pk)
                    else:
                        unmatched.append(pk)

                state.update(new_dict)
                model.load_state_dict(state)
                print('Pre-trained state loaded successfully...')
                print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
                print(f'loaded path: {os.path.join(self.chkpt_pth, f"{self.exp_name}.pth")}')
            except:
                print(f'ERROR in pretrained_dict @ {pretrained_path}')
        else:
            print('Enter pretrained_dict path.')

class Trainer(object):
    def __init__(self, model, batch, optimizer, metric_seg, metric_line):
        self.model = model
        self.batch = batch
        self.optimizer = optimizer
        self.metric_seg = metric_seg
        self.metric_line = metric_line
    
    def get_scores(self):
        return self.metric_seg.get_scores(), self.metric_line.get_scores()

    def reset_metric(self):
        self.metric_seg.reset(), self.metric_line.reset()
    
    def training_step(self, batched_data):
        img_batch = images_transform(batched_data['img'])
        depth_batch = images_transform(batched_data['depth'])
        # print(img_batch.shape, depth_batch.shape)
        seg_batch = torch_resizer(masks_transform(batched_data['f_seg']))
        line_batch = torch_resizer(masks_transform(batched_data['f_line']))
        # print(seg_batch.shape, line_batch.shape)
        # self.optimizer.zero_grad()
        self.model.zero_grad()

        loss, preds = self.model.forward([img_batch, depth_batch], target=[seg_batch, line_batch])

        if config['use_ocr']: # because only OCR has aux_loss.
            loss = loss['loss']

        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            loss = loss.mean()
        
        loss.backward()
        self.optimizer.step()

        preds_seg = preds['out_seg'].argmax(1)#(preds['out_seg'].squeeze(1) > 0.5).int()
        preds_seg = preds_seg.cpu().numpy()
        seg_batch = seg_batch.cpu().numpy()
        self.metric_seg.update(seg_batch, preds_seg)

        preds_line = preds['out_line'].argmax(1)#(preds['out_line'].squeeze(1) > 0.5).int()
        preds_line = preds_line.cpu().numpy()
        line_batch = line_batch.cpu().numpy()
        self.metric_line.update(line_batch, preds_line)

        return loss.item()

class Evaluator(object):
    def __init__(self, model, metric_seg, metric_line):
        self.model = model
        self.metric_seg = metric_seg
        self.metric_line = metric_line
    
    def get_scores(self):
        return self.metric_seg.get_scores(), self.metric_line.get_scores()

    def reset_metric(self):
        self.metric_seg.reset()
        self.metric_line.reset()
    
    def eval_step(self, data_batch):
        self.img_batch = images_transform(data_batch['img'])
        self.depth_batch = images_transform(data_batch['depth'])

        seg_batch = torch_resizer(masks_transform(data_batch['f_seg']))
        line_batch = torch_resizer(masks_transform(data_batch['f_line']))
        
        with torch.no_grad():
            _, preds = self.model.forward([self.img_batch, self.depth_batch]) 

        preds_seg = preds['out_seg'].argmax(1)#(preds['out_seg'].squeeze(1) > 0.5).int()
        self.preds_seg = preds_seg.cpu().numpy()
        self.seg_batch = seg_batch.cpu().numpy()
        self.metric_seg.update(self.seg_batch, self.preds_seg)

        preds_line = preds['out_line'].argmax(1)#(preds['out_line'].squeeze(1) > 0.5).int()
        self.preds_line = preds_line.cpu().numpy()
        self.line_batch = line_batch.cpu().numpy()
        self.metric_line.update(self.line_batch, self.preds_line)
        
    def get_sample_prediction(self):
        # get single image, lbl, pred for plotting
        self.img_batch = torch_imgresizer(self.img_batch).detach().cpu().numpy()
        self.depth_batch = torch_imgresizer(self.depth_batch).detach().cpu().numpy()
        
        imgs, depths, lbl_segs, pred_segs, lbl_lines, pred_lines = [], [], [], [], [], []
        for i in range(2): # show 2 images
            
            img = np.transpose(self.img_batch[i,...], (1,2,0))
            depth = np.transpose(self.depth_batch[i,...], (1,2,0))

            lbl_seg = self.seg_batch[i,...]
            pred_seg = self.preds_seg[i,...]

            lbl_line = self.line_batch[i,...]
            pred_line = self.preds_line[i,...]
            
            imgs.append((img*255).astype(np.uint8))
            depths.append((depth*255).astype(np.uint8))

            lbl_segs.append(g2c(lbl_seg.astype(np.uint8)))
            pred_segs.append(g2c(pred_seg.astype(np.uint8)))

            lbl_lines.append(g2c(lbl_line.astype(np.uint8)))
            pred_lines.append(g2c(pred_line.astype(np.uint8)))
        
        return imgs + depths + lbl_segs + lbl_lines + pred_segs + pred_lines

def eval_wrapper(evaluator, model, val_loader, total_avg_siou, total_avg_liou):

    model.eval() # <-set mode important
    sa, la = [], []
    vbar = tqdm(val_loader)
    for step, val_batch in enumerate(vbar):
        with torch.no_grad():
            evaluator.eval_step(val_batch)
            siou, liou = evaluator.get_scores()
            evaluator.reset_metric()

        sa.append(siou['iou_mean'])
        la.append(liou['iou_mean'])
        vbar.set_description(f'Validation - v_mIOU {siou["iou_mean"]:.4f}')

    img_gt_pred = evaluator.get_sample_prediction()
    tiled = imgviz.tile(img_gt_pred, shape=(3,4), border=(255,0,0))
    tiled = cv2.resize(tiled.astype(np.uint8), (512,256))# just for visulaization
    # plt.imshow(tiled)
    avg_siou = np.nanmean(sa)
    avg_liou = np.nanmean(la)
    total_avg_siou.append(avg_siou)
    total_avg_liou.append(avg_liou) 
    curr_viou = np.nanmax(total_avg_liou)

    return curr_viou, avg_siou, avg_liou, total_avg_siou, total_avg_liou, tiled
