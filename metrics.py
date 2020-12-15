import torch
from torch.nn import functional as F
import numpy as np 
import math
import pdb 

class EuclideanMetric:
    def __init__(self, 
                block_size: int = 4,
                image_size: int = 64):
        self.block_size = block_size
        self.image_size = image_size

    def get_block_center(self, block_image, has_batch = False):
        block_image = block_image.reshape(self.image_size, self.image_size) 
        for row_idx in range(block_image.shape[0]):
            for col_idx in range(block_image.shape[1]):
                #first one is top left corner
                if block_image[row_idx, col_idx].item() == 1: 
                    center = (row_idx + self.block_size/2, col_idx + self.block_size/2)
                    return center 
        # should never happen 
        return (self.block_size/2, self.block_size/2) 
                
class TransformerEuclideanMetric(EuclideanMetric):
    def __init__(self,
                 block_size: int = 4,
                 image_size: int = 64,
                 patch_size: int = 4):
        super(TransformerEuclideanMetric, self).__init__(block_size=block_size,
                                                         image_size=image_size)  

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.num_patches_per_row = int(math.sqrt(self.num_patches)) 

    def get_metric(self, true_block_image, pred_patches): 
        max_patch_idx = torch.argmax(pred_patches, dim = 1) 
        patch_center = self.get_patch_center(max_patch_idx) 
        block_center = self.get_block_center(true_block_image)  
        distance = self.get_euclidean_distance(patch_center, block_center) 
        return distance 

    def get_patch_center(self, patch_idx):
        # get upper left corner coord 
        patch_row = patch_idx // self.num_patches_per_row
        patch_col = patch_idx % self.num_patches_per_row
        
        image_row = self.patch_size * patch_row
        image_col = self.patch_size * patch_col
    
        patch_center = (image_row + self.patch_size / 2, image_col + self.patch_size / 2)
        return patch_center, (image_row, image_col) 
         
    def get_euclidean_distance(self, c1, c2): 
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

class TeleportationMetric:
    def __init__(self, 
                 block_size: int = 4):
        self.block_size = block_size

class TransformerTeleportationMetric(TeleportationMetric):
    def __init__(self, 
                 block_size: int = 4,
                 image_size: int = 64,
                 patch_size: int = 4):
        super(TransformerTeleportationMetric, self).__init__(block_size) 

        self.image_size = image_size
        self.patch_size = patch_size
        self.euclid = TransformerEuclideanMetric(block_size=block_size,
                                                 image_size=image_size,
                                                 patch_size=patch_size) 

    def select_prev_block(self, mask, true_prev_image, pred_prev_image):
        pred_prev_image = F.softmax(pred_prev_image, dim=0)[1,:,:,:]
        # overlap 
        pred_prev_image *= mask.squeeze(0) 
        # get max pixel 
        row_values, row_indices = torch.max(pred_prev_image, dim=0)
        col_values, col_idx = torch.max(row_values, dim=0)
        row_idx = row_indices[col_idx]
    
        pred_block_id = true_prev_image[0, row_idx, col_idx, 0].item()  
        return pred_block_id, (row_idx, col_idx), pred_prev_image[row_idx, col_idx, 0]

    def select_next_location(self, pred_patches):
        pred_patches = pred_patches[:,1,0]
        max_patch_idx = torch.argmax(pred_patches, dim = 0).item() 
        patch_center, patch_lc = self.euclid.get_patch_center(max_patch_idx) 
        return patch_center, patch_lc

    def execute_move(self, pred_corner, pred_idx, true_prev_image):
        # if pred_idx != true_idx, then true block doesn't move from previous location 
        # zero-out true location of pred block
        true_prev_image[true_prev_image==pred_idx] = 0
        # add in block at pred location 
        true_prev_image[0, pred_corner[0]:pred_corner[0] + self.block_size, pred_corner[1]: pred_corner[1] + self.block_size, 0] = pred_idx 
        return true_prev_image  

    def get_metric(self, true_next_image, true_prev_image, pred_prev_image, pred_next_patches, block_to_move): 
        true_next_image = true_next_image.detach().cpu() 
        true_prev_image = true_prev_image.detach().cpu() 
        pred_prev_image = pred_prev_image.detach().cpu() 
        pred_next_patches = pred_next_patches.detach().cpu() 
        w, h, __, __ = true_next_image.shape
        true_next_image = true_next_image.reshape(1, w, h, 1)
        true_prev_image = true_prev_image.reshape(1, w, h, 1) 

        # filter to 1-0
        true_prev_image_oh = true_prev_image.clone() 
        true_prev_image_oh[true_prev_image_oh != block_to_move] = 0
        true_prev_image_oh[true_prev_image_oh == block_to_move] = 1
        # get a block id to move 
        prev_block_id, pred_idx, pred_value = self.select_prev_block(true_prev_image_oh, true_prev_image, pred_prev_image) 
        # get the center of the most likely next location, to move the block to 
        pred_block_center, pred_block_corner = self.select_next_location(pred_next_patches) 
        # execute the move on the previous state 
        pred_next_image = self.execute_move(pred_block_corner, prev_block_id, true_prev_image) 
        # filter to be 1-0 
        true_next_image_oh = true_next_image.clone() 
        pred_next_image_oh = pred_next_image.clone() 

        true_next_image_oh[true_next_image != block_to_move] = 0
        true_next_image_oh[true_next_image == block_to_move] = 1
        pred_next_image_oh[pred_next_image != block_to_move] = 0
        pred_next_image_oh[pred_next_image == block_to_move] = 1
        # get centers          
        true_block_center = self.euclid.get_block_center(true_next_image_oh) 
        pred_block_center = self.euclid.get_block_center(pred_next_image_oh) 
       
        distance_pix = self.euclid.get_euclidean_distance(pred_block_center, true_block_center) 
        # convert to distance in block_lengths 
        distance_normalized = distance_pix / self.block_size

        return distance_normalized
