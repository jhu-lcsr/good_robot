import torch
from torch.nn import functional as F
import numpy as np 
import math
import pdb 
import copy 

np.set_printoptions(linewidth=400, edgeitems=400)
np.random.seed(12) 
torch.manual_seed(12) 


class MSEMetric():
    def __call__(self, true, pred):
        total = 0
        total_mse = 0 
        true = true.reshape(-1).detach().cpu().numpy()
        pred = pred.reshape(-1).detach().cpu().numpy() 
        mse_val = (true - pred)**2
        mse_avg = np.mean(mse_val) 
        return mse_avg

class AccuracyMetric():
    def __call__(self, true, pred):
        pred_idxs = torch.argmax(pred, dim = 2).detach().cpu() 
        tp = torch.eq(true, pred_idxs) 
        tp = tp.detach().cpu().numpy()
        return np.mean(tp) 

class EuclideanMetric:
    def __init__(self, 
                 block_size: int = 4,
                image_size: int = 64):
        self.block_size = block_size
        self.image_size = image_size

    def get_euclidean_distance(self, c1, c2): 
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

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
        super(TransformerEuclideanMetric, self).__init__(block_size = block_size, image_size=image_size)  

        self.block_size = block_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.num_patches_per_row = int(math.sqrt(self.num_patches)) 

    def get_patch_center(self, patch_idx):
        # get upper left corner coord 
        patch_row = patch_idx // self.num_patches_per_row
        patch_col = patch_idx % self.num_patches_per_row
        
        image_row = self.patch_size * patch_row
        image_col = self.patch_size * patch_col
    
        patch_center = (image_row + self.patch_size / 2, image_col + self.patch_size / 2)
        return patch_center, (image_row, image_col) 

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
        c, w, h, __ = pred_prev_image.shape
        if c == 2:
            pred_prev_image = F.softmax(pred_prev_image, dim=0)[1,:,:,:]
        elif c == 1:
            pred_prev_image = pred_prev_image[0,:,:,:]
        else:
            raise AssertionError(f"Wrong number of channels: expected 1 or 2, got {c}") 
        # overlap 
        pred_prev_image *= mask.squeeze(0) 
        # get max pixel 

        row_values, row_indices = torch.max(pred_prev_image, dim=0)
        col_values, col_idx = torch.max(row_values, dim=0)
        row_idx = row_indices[col_idx]
   
        pred_block_id = true_prev_image[0, row_idx, col_idx, 0].item()  
        return pred_block_id, (row_idx, col_idx), pred_prev_image[row_idx, col_idx, 0]

    def select_next_location(self, pred_patches):
        n, c, __ = pred_patches.shape
        if c == 2: 
            pred_patches = pred_patches[:,1,0]
        elif c == 1: 
            pred_patches = pred_patches[:,0,0]
        else: 
            raise AssertionError(f"Wrong number of channels: expected 1 or 2, got {c}") 

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

    def compute_distance(self, pred_block_corner, prev_block_id, block_to_move, pair, place_location):
        # execute the move on the previous state 
        pred_next_image = self.execute_move(pred_block_corner, prev_block_id, pair.prev_state_image) 
        # filter to be 1-0 
        true_next_image_oh = true_next_image.clone() 

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
        return distance_normalized, pred_block_center, true_block_center

    def get_metric(self, true_next_image, true_prev_image, pred_prev_image, pred_next_patches, block_to_move, next_xyz = None): 
        true_next_image = true_next_image.detach().cpu() 
        true_prev_image = true_prev_image.detach().cpu() 
        pred_prev_image = pred_prev_image.detach().cpu() 
        pred_next_patches = pred_next_patches.detach().cpu() 
        w, h, __, __ = true_next_image.shape
        true_next_image = true_next_image.reshape(1, w, h, 1)
        true_prev_image = true_prev_image.reshape(1, w, h, 1) 

        # filter to 1-0
        true_prev_image_oh = true_prev_image.clone() 
        true_prev_image_oh[true_prev_image_oh != 0] = 1
        # get a block id to move 
        prev_block_id, pred_idx, pred_value = self.select_prev_block(true_prev_image_oh, true_prev_image, pred_prev_image) 

        if next_xyz is None: 
            # get the center of the most likely next location, to move the block to 
            pred_block_center, pred_block_corner = self.select_next_location(pred_next_patches) 
        else:
            next_xyz = next_xyz.detach().cpu()
            pred_block_corner = np.array([next_xyz[1], next_xyz[2]]) 
            pred_block_corner += 1
            pred_block_corner /= 2
            # TODO(elias) replace with resolution
            pred_block_corner *= 64
            pred_block_corner = np.rint(pred_block_corner).astype(int) 
            #pred_block_corner = [int(x) for x in pred_block_corner]
            # map from -1 to 1 to 0 
            

        distance_normalized, pred_block_center, true_block_center  = self.compute_distance(pred_block_corner, prev_block_id, block_to_move, true_prev_image, true_next_image) 
        # given gold source block, what is predicted distance 
        distance_oracle_source, __, __  = self.compute_distance(pred_block_corner, block_to_move, block_to_move, true_prev_image, true_next_image) 

        block_acc = 1 if block_to_move == prev_block_id else 0

        to_ret = {"distance": distance_normalized,
                  "oracle_distance": distance_oracle_source,
                  "block_acc": block_acc, 
                  "pred_center": pred_block_center,
                  "true_center": true_block_center} 

        return to_ret 


class UNetTeleportationMetric(TransformerTeleportationMetric):
    def __init__(self, block_size = 4, image_size = 64):
        super(UNetTeleportationMetric, self).__init__(block_size = 4,
                                                      image_size = image_size,
                                                      patch_size = -1) 

        self.euclid = EuclideanMetric(block_size = block_size,
                                      image_size = image_size) 

    def select_next_location(self, pred_next_image):
        pred_next_image = F.softmax(pred_next_image,dim=0)[1,:,:,:]
        row_values, row_indices = torch.max(pred_next_image, dim=0)
        col_values, col_idx = torch.max(row_values, dim=0)
        row_idx = row_indices[col_idx]
        row_idx = row_idx.long().item()
        col_idx = col_idx.long().item() 
        patch_center = (row_idx, col_idx)
        patch_lc  = (row_idx - int(self.block_size/2), col_idx - int(self.block_size/2))
        return patch_center, patch_lc 

    def get_metric(self, true_next_image, true_prev_image, pred_prev_image, pred_next_image, block_to_move): 
        true_next_image = true_next_image.detach().cpu() 
        true_prev_image = true_prev_image.detach().cpu() 
        pred_prev_image = pred_prev_image.detach().cpu() 
        pred_next_image = pred_next_image.detach().cpu() 

        w, h, __, __ = true_next_image.shape
        true_next_image = true_next_image.reshape(1, w, h, 1)
        true_prev_image = true_prev_image.reshape(1, w, h, 1) 

        # filter to 1-0
        true_prev_image_oh = true_prev_image.clone() 
        true_prev_image_oh[true_prev_image_oh != 0] = 1
        # get a block id to move 
        prev_block_id, pred_idx, pred_value = self.select_prev_block(true_prev_image_oh, true_prev_image, pred_prev_image) 

        # get the center of the most likely next location, to move the block to 
        pred_block_center, pred_block_corner = self.select_next_location(pred_next_image) 

        distance_normalized, pred_block_center, true_block_center  = self.compute_distance(pred_block_corner, prev_block_id, block_to_move, true_prev_image, true_next_image) 
        # given gold source block, what is predicted distance 
        distance_oracle_source, __, __  = self.compute_distance(pred_block_corner, block_to_move, block_to_move, true_prev_image, true_next_image) 

        block_acc = 1 if block_to_move == prev_block_id else 0

        to_ret = {"distance": distance_normalized,
                  "oracle_distance": distance_oracle_source,
                  "block_acc": block_acc, 
                  "pred_center": pred_block_center,
                  "true_center": true_block_center} 

        return to_ret

class GoodRobotTransformerTeleportationMetric(TransformerTeleportationMetric):
    def __init__(self, 
                 block_size: int = 4,
                 image_size: int = 64,
                 patch_size: int = 4):
        super(GoodRobotTransformerTeleportationMetric, self).__init__(block_size, image_size, patch_size) 
        self.color_to_idx = {"red":1, "blue": 2, "green": 3, "yellow": 4} 
        self.expand_names = {"r":"red", "b":"blue","y":"yellow", "g": "green"}
        self.idx_to_color = {v:k for k,v in self.color_to_idx.items()}
        self.block_ratio = 9/64

    def select_prev_block(self, pair, pred_prev_image):
        c, w, h = pred_prev_image.shape
        if c == 2:
            pred_prev_image = F.softmax(pred_prev_image, dim=0)[1,:,:]
        elif c == 1:
            pred_prev_image = pred_prev_image[0,:,:]
        else:
            raise AssertionError(f"Wrong number of channels: expected 1 or 2, got {c}") 

        mask = np.copy(pair.prev_state_image)
        mask[mask != 0] = 1 
        pred_prev_image_before = pred_prev_image.clone() 
        pred_prev_image *= mask 
        # get max pixel 
        row_values, row_indices = torch.max(pred_prev_image, dim=0)
        col_values, col_idx = torch.max(row_values, dim=0)
        row_idx = row_indices[col_idx]
        true_prev_image = pair.prev_state_image 

        pred_block_id = true_prev_image[row_idx, col_idx].item()  
        # might be no regions predicted 
        if pred_block_id == 0:
            #print(f"selecting block 1")
            pred_block_id = 1
        return pred_block_id, (row_idx, col_idx), pred_prev_image[row_idx, col_idx]

    def execute_move(self, pred_corner, pred_idx, json_data):
        # if pred_idx != true_idx, then true block doesn't move from previous location 
        pred_color = self.idx_to_color[int(pred_idx)]
        json_data = copy.deepcopy(json_data) 
        json_data[pred_color] = pred_corner 
        return json_data 

    def compute_distance(self, pred_block_loc, prev_block_id, pair, block_size):
        json_data = copy.deepcopy(pair.json_data)
        true_block_loc = pair.next_location
        # swap 
        pred_x, pred_y = copy.deepcopy(pred_block_loc) 
        pred_x, pred_y = pred_y, pred_x
        pred_block_loc = [pred_x, pred_y]
        distance_pix = self.euclid.get_euclidean_distance(pred_block_loc, true_block_loc[0:2]) 
        # convert to distance in block_lengths 
        distance_normalized = distance_pix / block_size
        return distance_normalized

    def convert_to_loc(self, state):
        offset = [0.15, 0.0, 0.0]
        grid_dim = 14
        side_len = 0.035
        x_offset = 0.58
        grid_len = grid_dim*side_len
        state = np.array(state)
        state[0] += x_offset 
        for i in range(len(state)):
            state[i] = (state[i] * 2) / grid_len - offset[i]
        state = (state + 1)/2 * self.image_size
        return state.astype(int)

    def select_next_location(self, pred_next_image):
        pred_next_image = F.softmax(pred_next_image,dim=0)[1,:,:]
        row_values, row_indices = torch.max(pred_next_image, dim=0)
        col_values, col_idx = torch.max(row_values, dim=0)
        row_idx = row_indices[col_idx]
        row_idx = row_idx.long().item()
        col_idx = col_idx.long().item() 
        patch_center = (row_idx, col_idx)
        #patch_lc  = (row_idx - int(self.block_size/2), col_idx - int(self.block_size/2))
        patch_lc = None
        return patch_center, patch_lc 

    def get_metric(self, pair, pred_prev_image, pred_next_patches): 
        pair = copy.deepcopy(pair)
        pair.resolution = self.image_size
        pair.resize()
        pred_prev_image = pred_prev_image.detach().cpu() 
        pred_next_patches = pred_next_patches.detach().cpu() 
        # get a block id to move 
        prev_block_id, pred_idx, pred_value = self.select_prev_block(pair, pred_prev_image)
        # get the center of the most likely next location, to move the block to 
        pred_block_center, pred_block_corner = self.select_next_location(pred_next_patches) 

        block_size = int(self.block_ratio * self.image_size)
        distance_normalized = self.compute_distance(pred_block_center, prev_block_id, pair, block_size)
        true_block_center =  pair.next_location
        block_to_move = self.color_to_idx[self.expand_names[pair.source_code]]
        block_acc = 1 if block_to_move == prev_block_id else 0

        to_ret = {"distance": distance_normalized,
                  #"oracle_distance": distance_oracle_source,
                  "block_acc": block_acc, 
                  "pred_center": pred_block_center,
                  "true_center": true_block_center} 
        return to_ret 