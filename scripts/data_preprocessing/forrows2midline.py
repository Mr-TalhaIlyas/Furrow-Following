# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 20:04:03 2023

@author: talha
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import cv2
import numpy as np
from skimage import measure
from scipy import ndimage as ndi
from skimage import morphology  
import time


def get_triangle_vertices(junctions):
    # Find contours
    contours, _ = cv2.findContours(junctions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Get contour area to filter out small noise contours
        area = cv2.contourArea(cnt)
    
        # Consider contour if its area is above some threshold (you can adjust it based on your case)
        if area > 500:
            # Approximate contour with accuracy proportional to the contour perimeter
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
    
            # If the approximation has 3 vertices, then it's a triangle
            if len(approx) == 3:
                # print("Triangle vertices:")
                # for vertex in approx:
                #     print(vertex[0])
                return approx
            else:
                return None

def get_extended_point(p1, p2, y_extend=360):
    if p2[0] - p1[0] == 0:  # The line is vertical
        return np.asarray([p1[0], y_extend], dtype=np.int32)
    else:
        # Compute the slope of the line passing through p1 and p2
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        
        # Compute the y-intercept of the line, c = y - mx
        intercept = p1[1] - slope * p1[0]
        
        # Find the x-coordinate when y=y_extend, x = (y - c) / m
        x = (y_extend - intercept) / slope
        
        return np.asarray([x, y_extend], dtype=np.int32)
    
def get_median_line(img, resizer=2, dilate=True):
    ow, oh = img.shape[1], img.shape[0]
    w, h = img.shape[1]//resizer, img.shape[0]//resizer 
    
    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_NEAREST)
    
    
    ret,img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)
    
    lines = measure.label(img)
    thinned_lines = np.zeros((img.shape[0], img.shape[1], len(np.unique(lines))))
    
    
    for i in np.unique(lines):
        line = lines.copy()
        line[line!=i] = 0
        ret,line = cv2.threshold(line.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
        
        thin_line1, dist = morphology.medial_axis(line.astype(np.uint8), return_distance=True)
        thin_line1 = thin_line1.astype(np.uint8)
        # thin_line2 = morphology.skeletonize(line.astype(np.uint8))
        
        try:
            dist = ((dist - dist.min()) / ((dist.max() - dist.min())+1e-7) * 255).astype(np.uint8)
            junctions = cv2.threshold(dist, 200, 255, cv2.THRESH_BINARY)[1]
            vertices = get_triangle_vertices(junctions)
            centroid = np.mean(vertices, axis=0, dtype=int)
            
            thin_line1[vertices[0][0][1]:,:]=0
            # thin_line1[centroid[0][1]:,:]=0
            # x = junctions/255 + thin_line1
            
            extend_pt = get_extended_point(vertices[0][0], centroid[0], h)
            cv2.line(thin_line1, vertices[0][0], extend_pt, color=1, thickness=1)
        except np.AxisError:
            # print('AxisError')
            pass
        
        thinned_lines[..., i] = thin_line1 #+ thin_line2
        
    thinned_lines = np.sum(thinned_lines, axis=-1, dtype=np.uint8)
    # thinned_lines[0:60,:] = 0
    # thinned_lines[:,0:6] = 0
    
    if dilate:
        thinned_lines = cv2.dilate(thinned_lines, 
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
                                    iterations = 1)
    
    img = cv2.resize(thinned_lines, (ow,oh), interpolation=cv2.INTER_NEAREST)
    
    return img
    





