# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:35:31 2023

@author: talha
"""

import numpy as np
from skimage.measure import label as sml
from skimage.morphology import skeletonize as sms
import matplotlib.pyplot as plt
import cv2
from fmutils import fmutils as fmu
from tqdm import tqdm, trange
from scipy.spatial import distance
from scipy.interpolate import splprep, splev

def calculate_metrics(gt, pred):
    # extract the line coordinates
    gt_coords = np.column_stack(np.where(gt > 0))
    pred_coords = np.column_stack(np.where(pred > 0))

    # fit a parametric curve to each line
    gt_tck, gt_u = splprep(gt_coords.T, s=0)
    pred_tck, pred_u = splprep(pred_coords.T, s=0)

    # find the corresponding points on the ground truth curve for each point on the predicted curve
    # or Evaluating the gt_plynomial(B-spline) using the pred-B-spline's parameter values
    gt_corresponding_coords = np.column_stack(splev(pred_u, gt_tck))

    # calculate the distances from each point on the predicted line to the corresponding point on the ground truth line
    # will return a 2D matrix of distances.
    distances = distance.cdist(pred_coords, gt_corresponding_coords, 'euclidean')
    min_distances = distances.min(axis=1)

    # calculate the robust mean ALD
    robust_mean_ald = np.median(min_distances)

    # calculate the directional ALD/in image x-axis is same i.e. increases left to right
    directions = np.sign(pred_coords[:, 1] - gt_corresponding_coords[:, 1])  # as second dimension is the horizontal one
    left_ald = np.mean(min_distances[directions < 0])
    right_ald = np.mean(min_distances[directions > 0])
    
    # # calculate the top-bottom ALD /in image y-axis is reversed i.e. increases top down
    # directions = np.sign(pred_coords[:, 0] - gt_corresponding_coords[:, 0])
    # top_ald = np.mean(min_distances[directions < 0])
    # bottom_ald = np.mean(min_distances[directions > 0])
    
    # # Combining the left/right and top/bottom to make positive and negative directional ALD
    # pos_ald = (right_ald + top_ald) / 2
    # neg_ald = (left_ald + bottom_ald) / 2
    
    # calculate the continuity metric
    gradient = np.gradient(pred_coords[:, 0])  # as first dimension is the vertical one
    gradient_change = np.abs(np.diff(gradient))
    continuity_metric = np.mean(gradient_change)

    return robust_mean_ald, left_ald, right_ald, continuity_metric

# def getLargestRegion(segmentation):
#     # labels = label(segmentation)
#     # assume at least 1 region present
#     assert( labels.max() != 0 ), 'No Segments found in the input image'
#     largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
#     return largestCC

gts = fmu.get_all_files('C:/Users/talha/Desktop/ibrahim/eval/gt/')
preds = fmu.get_all_files('C:/Users/talha/Desktop/ibrahim/eval/pred/')

mald, mlld, mrld, mcont = [],[],[],[]
for i in trange(len(gts), desc='Evaluating'):

    gt = cv2.imread(gts[i], 0)
    pred = cv2.imread(preds[i], 0)
    
    # remove noist
    pred = cv2.erode(pred.astype(np.uint8),
                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
                #   np.ones((3,3),np.uint8),
                  iterations = 1)
    
    mgt = sml(gt) #multi gt
    mpred = sml(pred)
    
    lines_gt = len(np.unique(mgt)[1:])
    lines_pred = len(np.unique(mpred)[1:])
    
    single_ald, single_lald, single_rald, single_cont = [],[],[],[]
    for j in np.unique(mgt)[1:]:
            sgt = mgt.copy() #single gt
            sgt[sgt!=j] = 0
            for k in np.unique(mpred)[1:]:
                spred = mpred.copy()
                spred[spred!=k] = 0
                
                _,sgt = cv2.threshold(sgt.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
                _,spred = cv2.threshold(spred.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
                
                _, counts = np.unique(spred, return_counts=True)
                if counts[1] > 256: # because if area is smalled then this the its noise not line.
                    robust_mean_ald, left_ald, right_ald, continuity_metric = calculate_metrics(sms(sgt).astype(np.uint8), sms(spred).astype(np.uint8))
                    # robust_mean_ald, left_ald, right_ald, continuity_metric = calculate_metrics(sgt, spred)
                else:
                    robust_mean_ald, left_ald, right_ald, continuity_metric = 0, 0, 0, 0
                
                single_ald.append(robust_mean_ald)
                single_lald.append(left_ald)
                single_rald.append(right_ald)
                single_cont.append(continuity_metric)
    
    if lines_gt == lines_pred and lines_gt != 1: # multi lines are detected
        ald = np.nanmean(np.partition(np.asarray(single_ald), lines_gt)[:lines_gt]) # get lowest three values and take mean
        lld = np.nanmean(np.partition(np.asarray(single_lald), lines_gt)[:lines_gt])
        rld = np.nanmean(np.partition(np.asarray(single_rald), lines_gt)[:lines_gt])
        con = np.nanmean(np.partition(np.asarray(single_cont), lines_gt)[:lines_gt])
    elif len(single_ald) != 0: # if single or sopme lines are found
        ald = np.nanmean(np.min(np.asarray(single_ald))) # get lowest three values and take mean
        lld = np.nanmean(np.min(np.asarray(single_lald)))
        rld = np.nanmean(np.min(np.asarray(single_rald)))
        con = np.nanmean(np.min(np.asarray(single_cont)))
    elif lines_gt == 0 or lines_pred == 0:
        ald, lld, rld, con = 0, 0, 0, 0
    
    mald.append(ald)
    mlld.append(lld)
    mrld.append(rld)
    mcont.append(con)

mald = np.nanmean(np.asarray(mald))
mlld = np.nanmean(np.asarray(mlld))
mrld = np.nanmean(np.asarray(mrld))
mcont = np.nanmean(np.asarray(mcont))


print(mald, mlld, mrld, mcont)