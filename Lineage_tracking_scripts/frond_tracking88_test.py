# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:55:25 2023

@author: tiago
"""

from ultralytics import YOLO

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import skimage.draw
from skimage import io
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def yolo_segment_to_binary_mask(segment, m):
    mask = np.zeros((m, m), dtype=np.uint8)
    #draw = ImageDraw.Draw(Image.fromarray(mask))

    # Convert coordinates to the range of the mask
    #segment = [[int(x), int(y)] for x, y in segment[0]]
    #annotation = [[int(x), int(y)] for x, y in segment[0]
    # Iterate over line segments and draw them on the mask
    annotationx=[]; annotationy=[];
    for i in range(len(segment[0])):
        x1, y1 = segment[0][i]
        annotationx.append(x1);
        annotationy.append(y1);
        #x2, y2 = segment[i + 1]
        #draw.line([(x1, y1), (x2, y2)], fill=255, width=1)
    #print(annotationx)
    if annotationy:
      rr, cc = skimage.draw.polygon(annotationy, annotationx)
      mask[rr, cc] = 1


    return mask
def calculate_centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        centroid_x = 0;
        centroid_y = 0;
    else: 
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    return (centroid_x, centroid_y)

def calculate_accuracy(ground_truth_masks, predicted_masks):
    """
    Calculates the accuracy of predicted masks against ground truth masks.

    Args:
    ground_truth_masks (list): List of ground truth masks.
    predicted_masks (list): List of predicted masks.

    Returns:
    float: Accuracy of predicted masks.
    """

    # Create a copy of the predicted masks list to preserve the original order
    matched_predicted_masks = predicted_masks[:]

    # Create a list to store the matched indices
    matched_indices = []

    # Iterate over the ground truth masks and find the best match for each mask
    acc=[];
    for ground_truth_mask in ground_truth_masks:
        best_match_index = None
        best_iou = 0.0

        # Iterate over the remaining predicted masks to find the best match
        for i, predicted_mask in enumerate(matched_predicted_masks):
            intersection = (ground_truth_mask & predicted_mask).sum()
            union = (ground_truth_mask | predicted_mask).sum()
            iou = intersection / union

            if iou > best_iou:
                best_iou = iou
                best_match_index = i
                best_intersection = intersection;
                best_predicted_mask = predicted_mask

        # If a match is found, add the matched index to the list
        if best_match_index is not None:
            matched_indices.append(best_match_index)
            # Remove the matched mask from the list to prevent duplicate matches
            matched_predicted_masks.pop(best_match_index)
            #acc.append(best_intersection/best_predicted_mask.sum())
            acc.append(best_iou)
        else:
            acc.append(0)

    # Calculate the accuracy as the number of matched masks divided by the total number of ground truth masks
    #accuracy = len(matched_indices) / len(ground_truth_masks)
    accuracy = sum(acc) / len(acc)
    #print(matched_indices)
    #print(len(ground_truth_masks))
    return accuracy

def decode(results):
  img = results[0].orig_img; masks=[];
  print("detected masks:")
  print(len((results[0])))
  if len((results[0]))>0:
   for mask in (results[0]).masks:
     color=np.random.rand(1,3)*255
     segment = mask.xy
     binary_mask =  yolo_segment_to_binary_mask(segment,(results[0].orig_img).shape[0])
     masks.append(binary_mask)
     img[binary_mask>0.5] = color;
    #print(np.sum(binary_mask))
    #print()
#    plt.imshow(binary_mask)
#    plt.show()
  #masks=np.squeeze(masks);
   return masks#img, binary_mask
  else:
   return []


def get_masks_from_image_simple(image,model,score = 0.5):
    #sample = image
    #image = image.to(DEVICE)
    #model.eval()
    results = model.predict(image,show_labels=False)
    masks = decode(results)
    #outputs = model([image.to(DEVICE)])[0]
    #outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    #masks = (outputs['masks']).cpu().detach().numpy();
    return masks;



def get_bounding_boxes(masks):
    bounding_boxes = []
    for mask in masks:
        indices = np.where(mask)
        if np.any(indices):  # Check if the mask is non-empty
            ymin, ymax = np.min(indices[0]), np.max(indices[0])
            xmin, xmax = np.min(indices[1]), np.max(indices[1])
            bounding_boxes.append((ymin, xmin, ymax, xmax))
        else:
            bounding_boxes.append(None)  # Empty mask, append None
    return bounding_boxes


def get_boxes_from_image(image,filename,folder2save,model,score = 0.5):
    sample = image
    #image = image.to(DEVICE)
    #model.eval()
    outputs = model([image.to(DEVICE)])[0]
    #outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    boxes = (outputs['boxes']).cpu().detach().numpy();
    return boxes;

def get_masks_from_image(image,filename,folder2save,model,score = 0.5):
    sample = image
    #image = image.to(DEVICE)
    #model.eval()
    outputs = model([image.to(DEVICE)])[0]
    #outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    masks = (outputs['masks']).cpu().detach().numpy();
    return masks;

def get_masks_from_image_simple(image,model,score = 0.5):
    #sample = image
    #image = image.to(DEVICE)
    #model.eval()
    outputs = model([image.to(DEVICE)])[0]
    #outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    masks = (outputs['masks']).cpu().detach().numpy();
    return masks;


def calculate_iou(mask1, mask2):
    # Calculate the intersection area
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)

    # Calculate the union area
    union = np.logical_or(mask1, mask2)
    union_area = np.sum(union)

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_ioa(mask1, mask2):
    # Calculate the intersection area
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    
    # Calculate the area for mask1 and mask2:
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)

    # Calculate IoU
    ioa = np.max([intersection_area / area1, intersection_area / area2])
    return ioa

def remove_overlapping_masks(masks, threshold=0.9):
    num_masks = len(masks)
    mask_to_keep = np.ones(num_masks, dtype=bool)

    for i in range(num_masks):
        for j in range(i+1, num_masks):
            iou = calculate_iou(masks[i], masks[j])
            ioa = calculate_ioa(masks[i], masks[j])
            if iou >= threshold or ioa >= threshold:
                # Mark the masks to be removed
                mask_to_keep[j] = False

    non_overlapping_masks = masks[mask_to_keep]
    return non_overlapping_masks


def merge_overlapping_masks(masks, threshold=0.8):
    num_masks = len(masks)
    merged_masks = []
    mask_merged = np.zeros_like(masks[0], dtype=bool)

    for i in range(num_masks):
        if not mask_merged.any():
            mask_merged = masks[i]
        else:
            iou = calculate_iou(mask_merged, masks[i])
            ioa = calculate_ioa(mask_merged, masks[i])
            if iou >= threshold or ioa >= threshold:
                mask_merged = np.logical_or(mask_merged, masks[i])
            else:
                merged_masks.append(mask_merged)
                mask_merged = masks[i]

    if mask_merged.any():
        merged_masks.append(mask_merged)

    return merged_masks


def get_masks_from_image_yolov8(image,model,score = 0.5, dist_threshold=400):
  #sample = image
  results = model.predict(image,show_labels=False)
  masks = decode(results)
  if len(masks)>0:
    kernel = np.ones((5, 5), np.uint8)
    
    centroids_set1 = np.array([calculate_centroid(mask) for mask in masks])
    radial_distances = np.linalg.norm(centroids_set1[:, np.newaxis] - [(image.shape[1])/2, (image.shape[1])/2] , axis=2)
    mask_updated=[]
    
    for i, dist in enumerate(radial_distances):
        if dist<dist_threshold:
           smooth_mask = cv2.GaussianBlur(masks[i], (0, 0), 3)
           smooth_mask = cv2.dilate(smooth_mask, kernel, iterations=1)
           num_ones = np.sum(smooth_mask == 1)
           if num_ones > 10:
               mask_updated.append(smooth_mask) 
     
    mask_updated = remove_overlapping_masks(np.array(mask_updated), threshold=0.8) 
    return mask_updated;
  else:
    return [];




def yolo_segment_to_binary_mask_gt(segment, m):
    mask = np.zeros((m, m), dtype=np.uint8)
    #draw = ImageDraw.Draw(Image.fromarray(mask))

    # Convert coordinates to the range of the mask
    #segment = [[int(x), int(y)] for x, y in segment[0]]
    #annotation = [[int(x), int(y)] for x, y in segment[0]
    # Iterate over line segments and draw them on the mask
    annotationx=[]; annotationy=[];
    for i in range(len(segment)):
        x1, y1 = segment[i] * m
        annotationx.append(x1);
        annotationy.append(y1);
        #x2, y2 = segment[i + 1]
        #draw.line([(x1, y1), (x2, y2)], fill=255, width=1)
    #print(annotationx)
    rr, cc = skimage.draw.polygon(annotationy, annotationx)
    mask[rr, cc] = 1

    return mask
def decode_gt(mask_encoded,m):
  gt_masks=[];
  for mask in mask_encoded:
    masks=[];
    #color=np.random.rand(1,3)*255
    segment = mask
    for maski in segment:
      binary_mask =  yolo_segment_to_binary_mask_gt(maski,m)
      masks.append(binary_mask)
      #img[binary_mask>0.5] = color;
      #print(np.sum(binary_mask))
      #print()
#    plt.imshow(binary_mask)
#    plt.show()
     # masks=np.squeeze(masks);
    gt_masks.append(masks)
  return gt_masks#img, binary_mask

#plt.imshow(masks2[6]); plt.show()
#resized_mask = cv2.resize(masks2[6], (100, 100))
#plt.imshow(resized_mask); plt.show()

import itertools

def generate_non_repeating_combinations(x, y):
    if x <= y:
        set1 = list(range(0, x))
        set2 = list(range(0, y))
    else:
        set1 = list(range(0, y))
        set2 = list(range(0, x))
    combinations = list(itertools.permutations(set2, len(set1)))
    if x <= y:
        return [list(zip(set1, combo)) for combo in combinations]
    else:
        return [list(zip(combo, set1)) for combo in combinations]




def generate_perturbed_cost_matrix(cost_matrix, perturbation_factor):
    # Add small perturbations to the cost matrix
    perturbed_matrix = cost_matrix + np.random.uniform(0, perturbation_factor, size=cost_matrix.shape)
    #print(perturbed_matrix)
    return perturbed_matrix

def calculate_distances(centroids_set1, centroids_set2):
    # Calculate the x and y distances between centroids in set2 to set1
    x_distances = centroids_set1[:, 0][:, np.newaxis] - centroids_set2[:, 0]
    y_distances = centroids_set1[:, 1][:, np.newaxis] - centroids_set2[:, 1]
    return x_distances, y_distances

#def combo(x_distances, y_distances, set2_masks, set1_masks):
#    set2_masks_check

def move_masks(set2_copy, x_distances, y_distances, mask_matches):
    # Move the masks in set2_copy using x and y distances and mask matches
    for i, j in mask_matches:
        x_offset, y_offset = x_distances[j, i], y_distances[j, i]
        set2_copy[j] = np.roll(set2_copy[j], x_offset, axis=1)
        set2_copy[j] = np.roll(set2_copy[j], y_offset, axis=0)
resized_masks = []




def roll_masks(rmasks2, assignments, x_distances, y_distances):
  rmasks2_copy=rmasks2.copy()
  for i,j in assignments:
    rmasks2_copy[j] = np.roll(rmasks2_copy[j], x_distances[i,j], axis=1)
    rmasks2_copy[j] = np.roll(rmasks2_copy[j], y_distances[i,j], axis=0)
  return rmasks2_copy;

def roll_single_mask(mask, x_distance, y_distance):
  masks_copy=mask.copy()
  masks_copy = np.roll(masks_copy, x_distance, axis=1)
  masks_copy = np.roll(masks_copy, y_distance, axis=0)
  return masks_copy;

def get_iou_mat(set1_masks,set2_masks):
  intersection = np.logical_and(set1_masks[:, np.newaxis], set2_masks)
  union = np.logical_or(set1_masks[:, np.newaxis], set2_masks)
  i_matrix = np.sum(intersection, axis=(2, 3));  # intersection
  iou_matrix = i_matrix / np.sum(union, axis=(2, 3)) #IoU
  return iou_matrix;

def get_iou_allrolled(set1_masks,set2_masks,x_distances, y_distances):
  iou_matrix = np.zeros([len(set1_masks),len(set2_masks)])
  for i, mask1 in enumerate(set1_masks):
    for j, mask2 in enumerate(set2_masks):
       mask2_copy = roll_single_mask(mask2, x_distances[i,j], y_distances[i,j])
       intersection = np.logical_and(mask1, mask2_copy)
       union = np.logical_or(mask1, mask2_copy)
       i_value = np.sum(intersection);  # intersection
       iou_value = i_value / np.sum(union) #IoU
       iou_matrix[i,j] = iou_value;
  return iou_matrix;

def get_assignment(set1_masks,set2_masks, set2_original, original_iou,iou_matrix, distances, forces, C_iou, C_d, C_iou_d,Ci_iou, C_f):
  #intersection = np.logical_and(set1_masks[:, np.newaxis], set2_masks)
  #union = np.logical_or(set1_masks[:, np.newaxis], set2_masks)
  #i_matrix = np.sum(intersection, axis=(2, 3));  # intersection
  #iou_matrix = i_matrix / np.sum(union, axis=(2, 3)) #IoU
  #centroids_set1 = np.array([calculate_centroid(mask) for mask in set1_masks])
  #centroids_set2 = np.array([calculate_centroid(mask) for mask in set2_original])
  #distances = np.linalg.norm(centroids_set1[:, np.newaxis] - centroids_set2, axis=2)
  cost_matrix = (1-iou_matrix)*C_iou+distances*C_d + (1-iou_matrix)*distances*C_iou_d +(1-original_iou)*Ci_iou+C_f*forces;
  row_indices, col_indices = linear_sum_assignment(cost_matrix)
  assignments = list(zip(row_indices, col_indices))
  total_cost = cost_matrix[row_indices, col_indices].sum()
  return assignments, total_cost;

def get_multiple_assignment(iou_matrix,distances,original_iou, siam_matrix, forces, avg_disp, avg_growth, C_param):
  
    
  A1 = distances.copy(); max_dist = np.max(A1);
  A1 = 1/(A1+0.5);
  A2 = iou_matrix.copy();
  A3 = original_iou.copy(); 
  A4 = siam_matrix.copy();  A4 = 1/(A4+0.5);
  A5 = forces.copy(); A5 = 1/(A5+0.5);
  if max_dist>0:
  	A6 = pow(pow(avg_disp[0,0]-avg_disp[1,0],2)+pow(avg_disp[0,1]-avg_disp[1,1],2),0.5)/max_dist; 
  else: 
        A6 = 0; 
  A7 = abs(avg_growth[1] -  avg_growth[0]);
  C1, C2, C3, C4, C5, C12, C13, C14, C23, C24, C34, C15, C25, C35, C45, C16, C26, C36, C46, C56, C17, C27, C37, C47, C57 = C_param;
 
  B = C1*A1+C2*A2+C3*A3+C4*A4+C5*A5+C12*A1*A2+C13*A1*A3+C14*A1*A4+C23*A2*A3+C24*A2*A4+C34*A3*A4+C15*A1*A5+C25*A2*A5+C35*A3*A5+C45*A4*A5 + C16*A1*A6+C26*A2*A6+C36*A3*A6+C46*A4*A6+C56*A5*A6 + C17*A1*A7+C27*A2*A7+C37*A3*A7+C47*A4*A7+C57*A5*A7;
  cost_matrix = B;
  print("cost mat")
  print(A1)
  print(A2)
  print(A3)
  print(A4)
  print(A5)
  print(avg_disp)
  print(A7)

  row_indices, col_indices = linear_sum_assignment(cost_matrix)
  assignment= list(zip(row_indices, col_indices))
  total_cost = cost_matrix[row_indices, col_indices].sum()

  return assignment, total_cost;

import cv2
import numpy as np

# List of binary masks (replace these with your actual binary masks)

def create_mini_masks(mask_list,img2):
  mini_masks = []
  for binary_mask in mask_list:
    # Find the bounding box of the foreground
    coords = np.column_stack(np.where(binary_mask > 0))
    x, y, w, h = cv2.boundingRect(coords)
    #print(x, y, w, h)

    # Calculate the new bounding box coordinates, ensuring it's within the image bounds
    x1, y1, x2, y2 = max(0, x - 10), max(0, y - 10), min(img2.shape[0], x + w + 10), min(img2.shape[1], y + h + 10)

    # Crop the binary mask using the updated bounding box
    #cropped_mask = binary_mask[x1:x2, y1:y2]

    # Create a mini-mask with the same size as the bounding box
    mini_mask = np.zeros((w + 20, h + 20, 3), dtype=np.uint8)

    # Calculate the offset for the cropped region in mini_mask
    offset_x, offset_y = max(0, 10 - x), max(0, 10 - y)

    # Copy the cropped binary mask into the mini-mask with black padding
    mini_mask[offset_x:offset_x + (x2 - x1), offset_y:offset_y + (y2 - y1), 0] = img2[x1:x2, y1:y2, 0]
    mini_mask[offset_x:offset_x + (x2 - x1), offset_y:offset_y + (y2 - y1), 1] = img2[x1:x2, y1:y2, 1]
    mini_mask[offset_x:offset_x + (x2 - x1), offset_y:offset_y + (y2 - y1), 2] = img2[x1:x2, y1:y2, 2]

    # Resize mini_mask to (128, 128)
    mini_mask = cv2.resize(mini_mask, (128, 128))
    mini_masks.append(mini_mask)

  return mini_masks


def compute_embeddings(model, img1, img2):
    with torch.no_grad():
        output1, output2 = model(img1, img2)
    return output1, output2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)  # Load pre-trained ResNet-50
        # Modify the final classification layer to remove it
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])  # Remove the last two layers

        # Add your own layers for similarity calculation
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),#100x32768  # 2048 is the output size of ResNet-50 before the classification layer
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward_one(self, x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class SiameseNetwork_oneimg(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)  # Load pre-trained ResNet-50
        # Modify the final classification layer to remove it
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])  # Remove the last two layers

        # Add your own layers for similarity calculation
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),#100x32768  # 2048 is the output size of ResNet-50 before the classification layer
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward_one(self, x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x

    def forward(self, input1):
        output1 = self.forward_one(input1)
        return output1


def compute_distances(img1, img2):
    #with torch.no_grad():
        #output1, output2 = model(img1, img2)
        distances = torch.norm(img2 - img1, dim=1)  # Calculate Euclidean distance
        return distances

class SiameseDatasetUnlabeled(Dataset):
    def __init__(self, images, transform=None):
        self.images = images  # List of image pairs (each pair is a list with two image tensors)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pair = self.images

        image1, image2 = pair[0], pair[1]

        # Apply transformations if specified
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2


def add_border(image, target_size, border_color):
    """
    Add a border around the image to make it the target size.
    """
    h, w, _ = image.shape
    #print(f"image size = {h} by {w}");   
    if h == target_size and w == target_size:
    #    print(f"image fulfills target size"); 
        return image  # No need to add border if the image is already of target size
    
    # Create a new array for the bordered image
    bordered_image = np.ones((target_size, target_size, 3), dtype="float32") * border_color
    
    # Calculate the position to place the original image
    y_offset = (target_size - h) // 2
    x_offset = (target_size - w) // 2
    
    # Place the original image onto the bordered image
    bordered_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return bordered_image

import numpy as np

def extract_center(bordered_image, new_h, new_w):
    """
    Extract the center of the bordered image to make it the new size.
    """
    h, w, _ = bordered_image.shape
    
    if h == new_h and w == new_w:
        return bordered_image  # No need to extract center if already the desired size
    
    # Calculate the position of the center rectangle
    y_start = (h - new_h) // 2
    y_end = y_start + new_h
    x_start = (w - new_w) // 2
    x_end = x_start + new_w
    
    # Extract the center rectangle
    extracted_center = bordered_image[y_start:y_end, x_start:x_end]
    
    return extracted_center


def get_image_info_expand(image_path, model_fd, dist):
    
    image = plt.imread(image_path);
    image = image[:,:,0:3]*255; 
    # Check image dimensions and add border if necessary
    target_size = 831
    border_color = [181, 176, 179]
    h, w, _ = image.shape
    print(f"image size = {h} by {w}");   
    #if h != target_size or w != target_size:
    image = add_border(image, target_size, border_color)
    img = image.copy()
    masks = np.array(get_masks_from_image_yolov8(img,model_fd,0.5, dist))
    if len(masks)>0:
      centroids_set = np.array([calculate_centroid(mask) for mask in masks])
    
      return image, masks, centroids_set
    else:
      return image, [], []

def get_image_info(image_path, model_fd, dist):
    
    image = plt.imread(image_path);
    image = image[:,:,0:3]*255; img = image.copy()
    masks = np.array(get_masks_from_image_yolov8(img,model_fd,0.5, dist))
    centroids_set = np.array([calculate_centroid(mask) for mask in masks])
    
    return image, masks, centroids_set

def frame_relative_distance(centroid_set1, centroid_set2, matched_indices_set1_ed, matched_indices_set2_ed):
    distances_set1 = np.linalg.norm(centroids_set1[:, np.newaxis] - centroids_set1, axis=2)
    distances_set2 = np.linalg.norm(centroids_set2[:, np.newaxis] - centroids_set2, axis=2)
    return 1;

def plot_color_fronds(image, masks, color_map, sequence):
    img = image.copy()
    img = img[:,:,0:3]/255.0;
    for i, mask in enumerate(masks):
        binary_img = mask>0.5;
        x = np.where(sequence == i)
        if len(binary_img)>0:
         #print("size of mask")
         #print(binary_img.shape);
         #print(np.sum(binary_img==1));
         #print(len(x));
         #print(x);
         if x[0].size>0:
           img[binary_img] = color_map[x,:];
        #index_text = f"{idx}"
        #x, y = centroids[i]
        #cv2.putText(img, index_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img;
    #plt.imshow(img);
    #plt.axis("off")
    #plt.show()
    
import cv2
import numpy as np

def plot_color_fronds2(image, masks, centroids, color_map, sequence):
    img = image.copy()
    img = (img[:, :, 0:3])  # Normalize image

    for i, mask in enumerate(masks):
        binary_img = (mask > 0.5).astype(np.uint8)
        x = np.where(sequence == i)
        color = tuple(int(val * 255) for val in color_map[x, :][0])
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Color the region in the image
        img = cv2.drawContours(img, contours, -1, color, -1)

    return img

def save_mini_masks(mini_masks, filename):
    filepath=f"/content/sample_data2/{os.path.basename(filename)[:-4]}"
    if os.path.isdir(filepath):
        pass
    else:
        os.mkdir(filepath)
    for i, mini_mask in enumerate(mini_masks):
        #plt.imshow(mini_mask)
        #plt.show()
        cv2.imwrite(f"{filepath}/{os.path.basename(filename)[:-4]}_{i}.png",mini_mask)

# Example usage:
# image = cv2.imread("your_image.jpg")
# masks = [...]  # List of masks
# centroids = [...]  # List of centroids
# color_map = [...]  # List of colors
# sequence = [...]  # List of sequences
# result_image = plot_color_fronds(image, masks, centroids, color_map, sequence)
# cv2.imwrite("output_image.jpg", result_image)

import networkx as nx
from networkx.algorithms import isomorphism    
from networkx.algorithms.similarity import graph_edit_distance
from itertools import combinations

def find_close_mappings(graph1, graph2, num_mappings_to_find=3):
    mappings = []
    min_distances = [float('inf')] * num_mappings_to_find

    for node1 in graph1.nodes:
        subgraph1 = graph1.subgraph([node1] + list(graph1.neighbors(node1)))

        for node2 in graph2.nodes:
            subgraph2 = graph2.subgraph([node2] + list(graph2.neighbors(node2)))

            distance = graph_edit_distance(subgraph1, subgraph2)
            for i in range(num_mappings_to_find):
                if distance < min_distances[i]:
                    min_distances.insert(i, distance)
                    min_distances.pop()
                    mappings.insert(i, (node1, node2))

    return mappings

def compute_forces(centroid_set1, centroid_set2, masks1, masks2):
    distances1 = np.linalg.norm(centroid_set1[:, np.newaxis] - centroid_set1, axis=2)
    distances2 = np.linalg.norm(centroid_set2[:, np.newaxis] - centroid_set2, axis=2)
    area_set1 = np.array([np.sum(mask == 1) for mask in masks1]);
    area_set2 = np.array([np.sum(mask == 1) for mask in masks2]); 

    force_field1 = np.zeros([len(area_set1),len(area_set1)]) 
    for i, area1_1 in enumerate(area_set1):
        for j, area1_2 in enumerate(area_set1):
            if i != j:
                force_field1[i,j] = area1_2/pow(distances1[i,j]+1e-9,2)
                
    force_total1 = force_field1.sum(1)   
         
    force_field2 = np.zeros([len(area_set2),len(area_set2)])
    for i, area2_1 in enumerate(area_set2):
        for j, area2_2 in enumerate(area_set2):
            if i != j:
                force_field2[i,j] = area2_2/pow(distances2[i,j]+1e-9,2)
    force_total2 = force_field2.sum(1)
    
    force_similarity = abs(force_total1[:, np.newaxis] - force_total2)
    
    return force_similarity;
 
def computer_change(centroid_set1, centroid_set2, masks1, masks2):
    
    area_set1 = np.array([np.sum(mask == 1) for mask in masks1]); area1_sum = area_set1.sum()
    area_set2 = np.array([np.sum(mask == 1) for mask in masks2]); area2_sum = area_set2.sum()
    weighted_average_centroid1 = np.average(centroid_set1, axis=0, weights=area_set1)
    weighted_average_centroid2 = np.average(centroid_set2, axis=0, weights=area_set2)
    centroid_diff = abs(weighted_average_centroid1 - weighted_average_centroid2);
    average_change = pow(pow(centroid_diff[0],2)+pow(centroid_diff[1],2),0.5);
    
    return average_change;
    

def main_function(image1, masks1, centroid_set1, image2, masks2, centroid_set2, model_fd, model_siam, device, C_param, C_iou=1, C_d=0.1, C_iou_d=0, Ci_iou=1, C_siam=1, C_f=1):
    #filename1= image1_path;
    #filename1 = "/content/sample_data/separate_images/image_camA_0_0_20221019-194008_Plate2_a4.png";
    #image1 = plt.imread(filename1);
    
    #filename2= image2_path;
    #filename2 = "/content/sample_data/separate_images/image_camA_0_0_20221020-194008_Plate2_a4.png"
    #image2 = plt.imread(filename2);
    
    iou_threshold = 0.2  # Adjust as needed
    euclidean_threshold=20;
    
    #image1 = image1[:,:,0:3]*255; 
    img1 = image1.copy()
    #masks1 = np.array(get_masks_from_image_yolov8(image1,model_fd,score = 0.5))
    #image2 = image2[:,:,0:3]*255; 
    img2 = image2.copy()
    #masks2 = np.array(get_masks_from_image_yolov8(image2,model_fd,score = 0.5))
    
    
    binary_masks1 = masks1.copy()
    mini_masks1 = create_mini_masks(binary_masks1,img1)
    binary_masks2 = masks2.copy()
    mini_masks2 = create_mini_masks(binary_masks2,img2)
    
    
    new_size = (100, 100)
    rmasks2=[];
    #mask_shape = masks2.shape();
    for mask in masks2:
        resized_mask = cv2.resize(mask, new_size)
        rmasks2.append(resized_mask)
    
    rmasks1=[];
    for mask in masks1:
        resized_mask = cv2.resize(mask, new_size)
        rmasks1.append(resized_mask)
        
    centroids_set1 = np.array([calculate_centroid(mask) for mask in rmasks1])
    centroids_set2 = np.array([calculate_centroid(mask) for mask in rmasks2])
    #area_set1 = np.array([np.sum(mask == 1) for mask in rmasks1]);
    #area_set2 = np.array([np.sum(mask == 1) for mask in rmasks2]);  
                         
    #set1 = np.array([x_centroid,y_centroid, area_val for [x_centroid,y_centroid], area_val in zip(centroids_set1,area_set1)])
    #set2 = np.array([x_centroid,y_centroid, area_val for [x_centroid,y_centroid], area_val in zip(centroids_set2,area_set2)])
    #print(set1)
    
    area_set1 = np.array([np.sum(mask == 1) for mask in masks1]); area_set1_sum=np.sum(area_set1); area_set1=area_set1/area_set1_sum
    area_set2 = np.array([np.sum(mask == 1) for mask in masks2]); area_set2_sum=np.sum(area_set2); area_set2=area_set2/area_set2_sum
    set1 = [[ii*kk, jj*kk] for [ii, jj], kk in zip(centroid_set1,area_set1)];
    set2 = [[ii*kk, jj*kk] for [ii, jj], kk in zip(centroid_set2,area_set2)];
    set1 = [sum(pair)/len(pair) for pair in zip(*set1)]
    set2 = [sum(pair)/len(pair) for pair in zip(*set2)]
    
    avg_disp = np.array([set1, set2]);   
    avg_growth = [area_set1_sum, area_set2_sum];          
    #graph1 = nx.Graph()
    #graph1.add_edges_from(set1)

    #graph2 = nx.Graph()
    #graph2.add_edges_from(set2)
    
    
    #closest_mapping = find_closest_mapping(graph1, graph2);   
    #graph_mat = np.zeros([len(area_set1), len(area_set2)]); 
    #print(f"close_mappings = {close_mappings}");         
    #for i, (node1, node2) in enumerate(close_mappings, start=1):
    #    graph_mat[node1, node2] = 1;                     
                         
    original_iou = get_iou_mat(np.array(rmasks1),np.array(rmasks2))
    #centroids_set1 = np.array([calculate_centroid(mask) for mask in rmasks1])
    #centroids_set2 = np.array([calculate_centroid(mask) for mask in rmasks2])
    distances = np.linalg.norm(centroids_set1[:, np.newaxis] - centroids_set2, axis=2)
    radial_distances = np.linalg.norm(centroids_set1[:, np.newaxis] - [50, 50] , axis=2)
    
    
    forces = compute_forces(centroid_set1, centroid_set2, masks1, masks2);
    print(forces)
    
    x_distances, y_distances = calculate_distances(centroids_set1, centroids_set2)
    iou_matrix = get_iou_allrolled(rmasks1,rmasks2,x_distances, y_distances)
    #assignments, cost  = get_assignment(np.array(rmasks1),np.array(rmasks2),np.array(rmasks2),original_iou,iou_matrix,distances/np.max(distances), forces, C_iou, C_d, C_iou_d,Ci_iou, C_f)
    
    
    
    
    #rmasks2_copy = rmasks2.copy();
    ## get first assignment guess
    #rmasks2_copy = roll_masks(rmasks2_copy, assignments, x_distances, y_distances);
    ## assess assignment quality
    
    #ass_new, cost = get_assignment(np.array(rmasks1),np.array(rmasks2_copy),np.array(rmasks2),original_iou,iou_matrix,distances/np.max(distances),forces, C_iou, C_d, C_iou_d,Ci_iou, C_f)
    #ass_arr=[assignments]; cost_arr=[cost]; cost_min=cost; ass_min=assignments;
    #ass_min = assignments; cost_min =cost;
    
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model_siam.eval();
    siamese_distance=10*np.ones([len(mini_masks1),len(mini_masks2)])
    plot_imgs = 0; counter_siamese = 0;
    
    
    for i, mini_img1 in enumerate(mini_masks1):
        for j, mini_img2 in enumerate(mini_masks2):
            if distances[i,j]<25:
                counter_siamese = counter_siamese+1;
                #print(counter_siamese)
                mini_imgs = [];
                mini_imgs.append(mini_img1);
                mini_imgs.append(mini_img2);
                unlabeled_siamese_dataset = SiameseDatasetUnlabeled(mini_imgs, transform=transform)
                unlabeled_loader = DataLoader(unlabeled_siamese_dataset, batch_size=1, shuffle=False)
                
                for batch_idx, (img1, img2) in enumerate(unlabeled_loader):
                   img1, img2 = img1.to(device), img2.to(device)
                   output1, output2 = compute_embeddings(model_siam, img1, img2)
                   pair_distance_siam = compute_distances(output1, output2)
                   siamese_distance[i,j] = pair_distance_siam[0]
                if plot_imgs == 1:   
                    img1 = img1.squeeze(0).cpu().numpy()
                    img2 = img2.squeeze(0).cpu().numpy()   
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    axs[0].imshow(img1[0,:,:], cmap='gray')
                    axs[0].set_title("Image 1")
                    axs[0].axis('off')  # Remove the axis
                    axs[1].imshow(img2[0,:,:], cmap='gray')
                    axs[1].set_title("Image 2")
                    axs[1].axis('off')  # Remove the axis 
                    plt.suptitle("Distance: {:.4f}".format(siamese_distance[j,i]))
                    plt.show()
    print(f"{counter_siamese} masks compared using siamese NN");
    print(iou_matrix)
    print(distances)
    print(original_iou)
    print(siamese_distance)
    print(forces)
    print(avg_disp)
    print(avg_growth)
    assignment_siam, total_cost_siam = get_multiple_assignment(iou_matrix,distances,original_iou, siamese_distance, forces, avg_disp, avg_growth, C_param)
    matching_matrix2 = np.zeros([len(centroids_set1),len(centroids_set2)]);
    
    #ass_arr=[assignment_siam]; cost_arr=[total_cost_siam]; cost_min=total_cost_siam; ass_min=ass_new;
    ass_min = assignment_siam; cost_min =total_cost_siam;
    ## daugther candidates
    unmatched_indices_set1_ed = [i for i in range(len(centroids_set1)) if i not in [idx for _, idx in ass_min]]
    unmatched_indices_set2_ed = [j for j in range(len(centroids_set2)) if j not in [jdx for jdx, _ in ass_min]]
    matched_indices_set1_ed = []; matched_indices_set2_ed = [];
    unmatched_indices_set2_ed = list(range(len(centroids_set2)));
    for pair in ass_min:
      matched_indices_set1_ed.append(pair[0])
      matched_indices_set2_ed.append(pair[1])
      if pair[1] in unmatched_indices_set2_ed:
        unmatched_indices_set2_ed.remove(pair[1])
    
    #matched_indices_set2_ed = [jdx for j in range(len(centroids_set2)) if j in [jdx for jdx2, _ in ass_min]]
    ## assign daugther to mother
    #print(matched_indices_set2_ed, ", ", unmatched_indices_set2_ed)
    if len(unmatched_indices_set2_ed)>0: # if there are unmatched masks, match them to mother fronds
        matched_masks2 = masks2[matched_indices_set2_ed]
        intersection = np.logical_and(matched_masks2[:, np.newaxis], masks2[unmatched_indices_set2_ed])
        i_matrix = np.sum(intersection, axis=(2, 3)) + 1;  # intersection
        centroids_set1_daughter = np.array([calculate_centroid(mask) for mask in matched_masks2])
        centroids_set2_daughter = np.array([calculate_centroid(mask) for mask in masks2[unmatched_indices_set2_ed]])
        distances_daughter = np.linalg.norm(centroids_set1_daughter[:, np.newaxis] - centroids_set2_daughter, axis=2)
        mat_daughter = distances_daughter*0.1 + 10/(i_matrix);
        row_indices = np.min(mat_daughter, axis=0)
        #row_indices, col_indices = linear_sum_assignment(distances_daughter*0.1 + 10/(i_matrix))
        min_indices = np.argmin(mat_daughter, axis=0)
        row_indices = min_indices
        col_indices = np.arange(mat_daughter.shape[1])
        
        daugther_assignments = list(zip(row_indices, col_indices))
        daugther_assignments_updated = [];
        for i in range(len(daugther_assignments)):
          daugther_assignments_updated.append([matched_indices_set1_ed[daugther_assignments[i][0]], unmatched_indices_set2_ed[daugther_assignments[i][1]]])
    
    #print(1/(i_matrix))
    #print(daugther_assignments)
    ## prepare matching matrix: (1 for matching fronds, 2 for daughter-mother fronds)
    matching_matrix = np.zeros([len(centroids_set1),len(centroids_set2)]);
    ML_matrix = matching_matrix.copy()
    for pair in ass_min:
      matching_matrix[pair[0],pair[1]] = 1;
      ML_matrix[:,pair[1]]=1;
    if len(unmatched_indices_set2_ed)>0:  
        for pair in daugther_assignments_updated:
            matching_matrix[pair[0],pair[1]] = 2;
      
       
    
    
    
    for pair in assignment_siam:
      matching_matrix2[pair[0],pair[1]] = 1;
      #ML_matrix[:,pair[1]]=1;
    if len(unmatched_indices_set2_ed)>0:   
        for pair in daugther_assignments_updated:
            matching_matrix2[pair[0],pair[1]] = 2;
    
    #image1 = cv2.imread(filename1)
    #image1_copy = write_index_on_image(image1, centers_set1)
    #plt.imshow(cv2.cvtColor(image1_copy,cv2.COLOR_BGR2RGB))
    #plt.axis('off');
    #plt.show()
    
    
    #image2 = cv2.imread(filename2)
    #image2_copy = write_index_on_image(image2, centers_set2)
    #plt.imshow(cv2.cvtColor(image2_copy,cv2.COLOR_BGR2RGB))
    #plt.axis('off');
    #plt.show()
    
    return matching_matrix, matching_matrix2, siamese_distance, assignment_siam, original_iou,iou_matrix,distances,siamese_distance,forces 
    
    
    
    
 #model_fd = YOLO("/content/drive/MyDrive/models/yolov8-fd-seg.pt");



