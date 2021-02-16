import numpy as np  
import json
import pathlib 
import torch
import cv2 
import re
import copy 
import pdb 
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
from IPython.display import clear_output
from tqdm import tqdm 


def check_success(data, idx):
    return data[idx][0] == 1    

class Pair:
    def __init__(self, prev_image, prev_location, next_image, next_location, resolution = 224):
        self.prev_image = prev_image
        self.prev_location = prev_location
        self.next_image = next_image
        self.next_location = next_location 
        self.w = 30
        self.source_code = None 
        self.source_location = None
        self.target_code = None
        self.target_location = None 
        self.relation_code = None
        self.resolution = resolution 

    def show(self):
        fig,ax = plt.subplots(1)
        ax.imshow(self.prev_image)
        prev_location = self.prev_location - int(self.w/2)
        next_location = self.next_location - int(self.w/2)
        rect = patches.Rectangle(prev_location, self.w, self.w ,linewidth=3,edgecolor='w',facecolor='none')
        ax.add_patch(rect)
        plt.show()
        fig,ax = plt.subplots(1)
        ax.imshow(self.next_image)
        rect = patches.Rectangle(next_location, self.w, self.w,linewidth=3,edgecolor='w',facecolor='none')
        ax.add_patch(rect)
        plt.show()

    def resize(self):
        self.prev_image = cv2.resize(self.prev_image, (self.resolution,self.resolution), interpolation = cv2.INTER_AREA)
        self.next_image = cv2.resize(self.next_image, (self.resolution,self.resolution), interpolation = cv2.INTER_AREA)
        self.raio = self.resolution / 224 

        # normalize location and width 
        self.w *= self.ratio 
        self.w = int(self.w)

        self.prev_location = self.prev_location.astype(float).copy() * self.ratio
        self.prev_location = self.prev_location.astype(int)
        self.next_location = self.next_location.astype(float).copy() * self.ratio
        self.next_location = self.next_location.astype(int)

    def get_mask(self, location):
        w, h, __ = self.prev_image.shape
        mask = np.zeros((1, w, h))
        start = location - int(self.w/2)
        start = start.astype(int)
        mask[:, start[1]: start[1] + self.w, start[0]: start[0] + self.w] = 1
        return mask 

    @classmethod
    def from_idxs(cls, grasp_idx, place_idx, data, image_home):
        prev_location = data[grasp_idx][2:][::-1]
        next_location = data[place_idx][2:][::-1]
        grasp_prefix = str(1000000 + grasp_idx)[1:]
        place_prefix = str(1000000 + place_idx)[1:]
        grasp_path = str(image_home.joinpath(f"{grasp_prefix}.0.color.png"))
        place_path = str(image_home.joinpath(f"{place_prefix}.2.color.png"))

        # TODO(elias) swap BG channels here 
        prev_image = cv2.imread(grasp_path)
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB)
        next_image = cv2.imread(place_path)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
        return cls(prev_image, prev_location, next_image, next_location)

    def read_json(self, json_path):
        with open(json_path) as f1:
            data = json.load(f1)
        num_blocks = data['num_obj']
        colors = data['color_names'][0:num_blocks]
        coords = data['positions']
        assert(len(coords) == num_blocks)
        assert(len(coords[0]) == 3)
        to_ret = {}
        for color, coord in zip(colors, coords):
            # normalize location to resolution 
            coord = np.array(coord)
            coord += 1
            coord /= 2
            coord *= self.resolution 
            to_ret[color] = coord 
        return to_ret 

    def get_moved_block(self, prev_coords, next_coords):
        diff = {k: prev_coords[k][0:2] - next_coords[k][0:2] for k in prev_coords.keys()}
        diff = [(k, np.sum(x)) for k, x in diff.items()]
        # get block with greatest diff in location 
        return list(sorted(diff, key = lambda x: x[1]))[-1]

    @classmethod
    def from_sim_idxs(cls, grasp_idx, place_idx, data, image_home, json_home): 
        pair = Pair.from_idxs(grasp_idx, place_idx, data, image_home)
        # annotate based on sim data 
        # rules for row-making
        grasp_json_path = json_home.joinpath(f"object_positions_and_orientations_{grasp_idx}_0.json")
        place_json_path = json_home.joinpath(f"object_positions_and_orientations_{place_idx}_2.json")

        json_data = pair.read_json(grasp_json_path)
        src_color, tgt_color = pair.combine_json_data(json_data)

        # TODO (elias) rules for stacking
        return pair 

    def combine_json_data(self, json_data, next_to=True, filter_left=False): 
        # first pass: all prompts say "next to" and reference the closest block to the left of the target location
        # if no such block exists, skip for now 
        def euclid_dist(p1, p2): 
            total = 0
            for i in range(len(p1)):
                total += (p1[i] - p2[i])**2
            return np.sqrt(total)

        pdb.set_trace() 
        # find block closest to grasp index 
        # min_grasp_color = self.get_moved_block(prev_json_data, next_json_data) 
        grasp_dists = [(x[0], euclid_dist(self.prev_location, x[1])) for x in json_data.items()]
        min_grasp_color = list(sorted(grasp_dists, key = lambda x:x[1]))[0][0]

        other_blocks = [x for x in json_data.items() if x[0] != min_grasp_color]

        # filter down to blocks that are to the left of place index 
        # x must be < 
        if filter_left: 
            left_of = [x for x in other_blocks if x[1][0] < self.next_location[0]]
            if len(left_of) == 0:
                print(f"no left of")
                return None
            remaining_blocks = left_of
        else:
            remaining_blocks = other_blocks

        # find block closest to place index 
        place_dists = [(x[0], euclid_dist(self.next_location, x[1])) for x in remaining_blocks]
        min_place_color = list(sorted(place_dists, key = lambda x:x[1]))[0][0]
        # get relation between place location and place block 
        #place_landmark_pos = json_data[min_place_color]
        #place_pos = self.next_locationA
        # 
        self.source_code = min_grasp_color[0]
        self.target_code = min_place_color[0]

        self.relation_code = "next_to"

        return min_grasp_color, min_place_color  



    

    def clean(self):
        # re-order codes so that "top", "bottom", come bfore "left" "right"
        if self.source_location == "none":
            self.source_location = "n"
        if self.target_location == "none":
            self.target_location = "n"
        if self.source_location == "as":
            self.source_location = "sa"
        if self.target_location == "as": 
            self.target_location = "sa"
        if self.source_location == "ds":
            self.source_location = "sd"
        if self.target_location == "ds": 
            self.target_location = "sd" 
        if self.source_location == "aw":
            self.source_location = "wa"
        if self.target_location == "aw": 
            self.target_location = "wa"   
        if self.source_location == "dw":
            self.source_location = "wd"
        if self.target_location == "dw": 
            self.target_location = "wd"
        if self.relation_code == "ds":
            self.relation_code = "sd"
        if self.relation_code == "wa":
            self.relation_code = "aw"
        if self.relation_code == "dw":
            self.relation_code = "wd"
        if self.relation_code == "sa":
            self.relation_code = "as"

    def generate(self):
        self.clean() 
        location_lookup_dict = {"w": "top", "d": "right", "a": "left", "s": "bottom", "n":""} 

        location_lookup_fxn = lambda x: " ".join([location_lookup_dict[y] for y in list(x)])
        relation_lookup_dict = {"on": "on top of",
                                "next_to": "next to",
                                "w": "over",
                                "s": "under",
                                "a": "to the left of",
                                "d": "to the right of",
                                "aw": "up and to the left of",
                                "as": "down and to the left of",
                                "wd": "up and to the right of",
                                "sd": "down and to the right of"}

        color_lookup_dict = {"r": "red", "b": "blue", "g": "green", "y": "yellow", 
                            "green":"green", "blue":"blue", "yellow":"yellow","red":"red"}

        stack_template = "stack the {source_location} {source_color} block {relation} the {target_location} {target_color} block"
        row_template = "move the {source_color} block {relation} the {target_color} block"

        # is stacking task 
        try:
            if self.source_location is not None:
                return stack_template.format(source_location = location_lookup_fxn(self.source_location), 
                                            source_color = color_lookup_dict[self.source_code],
                                            target_location = location_lookup_fxn(self.target_location), 
                                            target_color = color_lookup_dict[self.target_code],
                                            relation = relation_lookup_dict[self.relation_code])
        except KeyError:
            return "bad"
        try:
            return row_template.format(source_color = color_lookup_dict[self.source_code],
                                    target_color = color_lookup_dict[self.target_code],
                                    relation = relation_lookup_dict[self.relation_code])
        except KeyError:
            return "bad"

def flip_pair(pair, axis):
    pair.clean() 
    flip_lookup = {1: {"w": "w", "a": "d", "s": "s", "d": "a"},
                   2: {"w": "d", "a": "s", "s": "a", "d": "w"},
                   3: {"w": "s", "a": "a", "s": "w", "d": "d"},
                   4: {"w": "a", "a": "w", "s": "d", "d": "s"}}
    
    def replace(code):
        code = list(code)
        if code[0] == "n":
            return "".join(code)
        try:
            code = [flip_lookup[axis][x] for x in code]
        except:
            pdb.set_trace() 
        return "".join(code)

    new_pair = copy.deepcopy(pair)
    if new_pair.source_location is not None and new_pair.source_location != "none":
        new_pair.source_location = replace(new_pair.source_location)
    if new_pair.target_location is not None and new_pair.target_location != "none":
        new_pair.target_location = replace(new_pair.target_location)
    if new_pair.relation_code != "on": 
        new_pair.relation_code = replace(new_pair.relation_code)

    def flip_image(img):
        if axis == 1:
            # vertical flip
            flipped_img = cv2.flip(img, 1)
        elif axis == 3: 
            # horizontal flip 
            flipped_img = cv2.flip(img, 0)
        elif axis == 2: 
            # along backward diag
            flipped_img = np.transpose(np.rot90(img,2, axes=(0,1)), axes = (1,0,2))
        elif axis == 4:
            # along regular diag 
            flipped_img = np.transpose(img, axes = (1,0,2)) 
        else:
            raise AssertionError("Axis must be one of [1,2,3,4]")
        return flipped_img

    def flip_coords(coords):
        max = 224
        if axis == 1:
            # x coord flips 
            coords[0] = 224 - coords[0]
        elif axis == 2:
            # transpose and rotate 
            coords[0], coords[1] = coords[1], coords[0] 
            coords[0] = 224 - coords[0]
            coords[1] = 224 - coords[1]
        elif axis == 3:
            # y coord flips 
            coords[1] = 224 - coords[1]
        elif axis == 4:
            # transpose
            coords[0], coords[1] = coords[1], coords[0] 
        else:
            pass 
        return coords 

    new_pair.prev_image = flip_image(new_pair.prev_image)
    new_pair.next_image = flip_image(new_pair.next_image)
    new_pair.prev_location = flip_coords(new_pair.prev_location)
    new_pair.next_location = flip_coords(new_pair.next_location)
    return new_pair 


def get_pairs(data_home, is_sim = False):
    image_home = data_home.joinpath("data/color-heightmaps")
    if is_sim: 
        json_home = data_home.joinpath("data/variables")
    executed_action_path = data_home.joinpath("transitions/executed-action.log.txt")
    place_successes_path = data_home.joinpath("transitions/place-success.log.txt")
    grasp_successes_path = data_home.joinpath("transitions/grasp-success.log.txt")

    kwargs = {'delimiter': ' ', 'ndmin': 2}
    executed_action_data = np.loadtxt(executed_action_path, **kwargs)
    place_succ_data = np.loadtxt(place_successes_path, **kwargs)
    grasp_succ_data = np.loadtxt(grasp_successes_path, **kwargs)
    prev_act = None
    prev_grasp_idx = None 
    pick_place_pairs = []
    for demo_idx in range(len(executed_action_data)):
        ex_act = executed_action_data[demo_idx]
        grasp = False
        if int(ex_act[0]) == 0:
            continue
        elif int(ex_act[0]) == 1:
            data = grasp_succ_data
            grasp = True
        elif int(ex_act[0]) == 2:
            data = place_succ_data
        else:
            raise AssertionError(f"action must be of on [0, 1, 2]")

        try:
            was_success = check_success(data, demo_idx) 
        except IndexError:
            break

        if prev_act == "grasp":
            # next action must be place if prev was successful grasp 
            try:
                assert(not grasp)
            except AssertionError:
                print(f"double grasp at {demo_idx}")
        if prev_act == "place":
            try:
                assert(grasp)
            except AssertionError: 
                print(f"double place at {demo_idx}")

        # sanity checks 
        if grasp and was_success:
            prev_act = "grasp"
            prev_grasp_idx = demo_idx 
        if not grasp and was_success:
            prev_act = "place"
            # now you can create a pair with the previous action's grasp and current place 
            if is_sim:
                pair = Pair.from_sim_idxs(prev_grasp_idx, demo_idx, executed_action_data, image_home, json_home)
            else:
                pair = Pair.from_idxs(prev_grasp_idx, demo_idx, executed_action_data, image_home)
            pick_place_pairs.append(pair)
            prev_grasp_idx = None 

    return pick_place_pairs

def get_input(prompt, valid_gex):
    var = None
    while var is None:
        inp = input(prompt) 
        if valid_gex.match(inp) is not None:
            var = inp
        else:
            continue
    return var 

def annotate_pairs(pairs, 
                  is_stack = False):
    pairs_with_actions = []

    color_gex = re.compile("(bad)|[rbyg]")
    relation_gex = re.compile("[wasd]{1,2}")
    location_gex = re.compile("[wasd]{1,2}|(none)")
    for p in tqdm(pairs):
        p.show()
        source_color = get_input("Source color: ", color_gex)
        if is_stack: 
            source_location = get_input("Source location: ", location_gex)
        target_color = get_input("Target color: ", color_gex)
        if is_stack:
            target_location = get_input("Target location: ", location_gex)

        if not is_stack:
            # if row-making, get position
            relation = get_input("Relation: ", relation_gex)
            target_location, source_location = None, None
        else:
            # stacking only has one 
            relation = 'on'

        p.source_code = source_color
        p.target_code = target_color
        p.relation_code = relation
        p.source_location = source_location
        p.target_location = target_location 
        p.clean() 
        pairs_with_actions.append(p)
        clear_output(wait=True)

    return pairs_with_actions
