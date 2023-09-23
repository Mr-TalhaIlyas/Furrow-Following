# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:59:15 2023

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
from skimage.morphology import skeletonize

def completeness_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    # To avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union

def length_ratio(pred, gt):
    length_pred = np.sum(pred)
    length_gt = np.sum(gt)
    
    # To avoid division by zero
    if length_gt == 0:
        return 0
    
    return length_pred / length_gt

def max_possible_smoothness(image_shape):
    # Maximum gradient change for entire image would be twice the number of rows plus twice the number of columns
    return 2 * image_shape[0] + 2 * image_shape[1]

def normalize_smoothness(smoothness_val, image_shape):
    maximum_smoothness = max_possible_smoothness(image_shape)
    
    # To avoid division by zero
    if maximum_smoothness == 0:
        return 0
    
    return smoothness_val / maximum_smoothness

def smoothness_metric(pred):
    pred = skeletonize(pred).astype(np.uint8)
    # Calculate gradient
    gradient_x = np.gradient(pred, axis=0)
    gradient_y = np.gradient(pred, axis=1)
    
    total_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    
    return np.sum(total_gradient)

def norm_smoothness_metric(pred):
    smoothness_val = smoothness_metric(pred)
    nsm = normalize_smoothness(smoothness_val, pred.shape)
    return nsm 

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

gts = fmu.get_all_files('C:/Users/talha/Downloads/ibrahim/sandesh_/gt/')
preds = fmu.get_all_files('C:/Users/talha/Downloads/ibrahim/sandesh_/pred/')

mald, mlld, mrld, mcont = [],[],[],[]
mlr, mcs, mnsm  = [], [], []
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
    single_lr, single_cs, single_nsm = [], [], []
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
                    lr = length_ratio(spred, sgt)
                    cs = completeness_score(spred, sgt)
                    nsm = norm_smoothness_metric(spred)
                    # robust_mean_ald, left_ald, right_ald, continuity_metric = calculate_metrics(sgt, spred)
                else:
                    robust_mean_ald, left_ald, right_ald, continuity_metric = 0, 0, 0, 0
                    lr, cs, nsm = 0, 0, 0
                
                single_ald.append(robust_mean_ald)
                single_lald.append(left_ald)
                single_rald.append(right_ald)
                single_cont.append(continuity_metric)
                
                single_lr.append(lr)
                single_cs.append(cs)
                single_nsm.append(nsm)
    
    if lines_gt == lines_pred and lines_gt != 1: # multi lines are detected
        ald = np.nanmean(np.partition(np.asarray(single_ald), lines_gt)[:lines_gt]) # get lowest three values and take mean
        lld = np.nanmean(np.partition(np.asarray(single_lald), lines_gt)[:lines_gt])
        rld = np.nanmean(np.partition(np.asarray(single_rald), lines_gt)[:lines_gt])
        con = np.nanmean(np.partition(np.asarray(single_cont), lines_gt)[:lines_gt])
        
        slr = np.nanmean(np.partition(np.asarray(single_lr), lines_gt)[:lines_gt])
        scs = np.nanmean(np.partition(np.asarray(single_cs), lines_gt)[:lines_gt])
        snsm = np.nanmean(np.partition(np.asarray(single_nsm), lines_gt)[:lines_gt])
        
    elif len(single_ald) != 0: # if single or sopme lines are found
        ald = np.nanmean(np.min(np.asarray(single_ald))) # get lowest three values and take mean
        lld = np.nanmean(np.min(np.asarray(single_lald)))
        rld = np.nanmean(np.min(np.asarray(single_rald)))
        con = np.nanmean(np.min(np.asarray(single_cont)))
        
        slr = np.nanmean(np.min(np.asarray(single_lr)))
        scs = np.nanmean(np.min(np.asarray(single_cs)))
        snsm = np.nanmean(np.min(np.asarray(single_nsm)))
        
    elif lines_gt == 0 or lines_pred == 0:
        ald, lld, rld, con = 0, 0, 0, 0
        slr, scs, snsm = 0, 0, 0
    
    mald.append(ald)
    mlld.append(lld)
    mrld.append(rld)
    mcont.append(con)
    
    mlr.append(slr)
    mcs.append(scs)
    mnsm.append(snsm)
    
mald = np.nanmean(np.asarray(mald))
mlld = np.nanmean(np.asarray(mlld))
mrld = np.nanmean(np.asarray(mrld))
mcont = np.nanmean(np.asarray(mcont))

mlr = np.nanmean(np.asarray(mlr))
mcs = np.nanmean(np.asarray(mcs))
mnsm = np.nanmean(np.asarray(mnsm))


print(f'\nMean ALD:{mald}; \nMean LLD:{mlld}, \nMean RLD:{mrld}, \nMeanCont: {mcont}')
print(f'\nMean Length Ratio:{mlr}, \nMean Completeness Score:{mcs}, \nMean Smoothness:{mnsm}')
