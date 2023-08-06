# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:23:36 2023

@author: talha
# """

#%%
import numpy as np
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
import cv2

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
