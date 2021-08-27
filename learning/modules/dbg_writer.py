from data_io.results import save_results_extra_image, save_results_extra_gif
from torch.autograd import Variable
from learning.inputs.vision import torch_to_np

from data_io.env import load_env_img
from visualization import Presenter


def get_map_overlaid(img, env_id):
    env_img = load_env_img(env_id, width=256, height=256)
    overlaid = Presenter().overlaid_image(env_img, img)
    return overlaid


def write_map_overlaid(md, img, args):
    name = ""
    if args is not None:
        world_size = args["world_size"]
        name = "_" + str(args["name"])
        img = img[0:world_size, 0:world_size, :]
    # Take the first 3 channels if there are more
    if img.shape[2] > 3:
        img = img[:,:,0:3]

    overlaid = get_map_overlaid(img, md.ENV_ID)
    save_results_extra_image(md.RUN_NAME, md.ENV_ID, md.SET_IDX, md.SEG_IDX, "map_overlaid" + name, overlaid)


gif_frames = {}
def write_gif_overlaid(md, img, args):
    name = ""
    if args is not None:
        world_size = args["world_size"]
        name = args["name"]
        img = img[0:world_size, 0:world_size, :]
    # Take the first 3 channels if there are more
    if img.shape[2] > 3:
        img = img[:, :, 0:3]

    overlaid = get_map_overlaid(img, md.ENV_ID)

    global gif_frames
    seg_identifier = str(md.RUN_NAME) + str(md.ENV_ID) + "_" + str(md.SET_IDX) + "_" + str(md.SEG_IDX) + "_map_overlaid" + str(name)
    if seg_identifier not in gif_frames:
        gif_frames[seg_identifier] = {
            "frames": [],
            "run_name": md.RUN_NAME,
            "env_id": md.ENV_ID,
            "set_idx": md.SET_IDX,
            "seg_idx": md.SEG_IDX,
            "name": name
        }

    gif_frames[seg_identifier]["frames"].append(overlaid)


def write_gif(md, img, args):
    name = ""
    if args is not None:
        name = args["name"]
    # Take the first 3 channels if there are more
    if img.shape[2] > 3:
        img = img[:, :, 0:3]

    img = Presenter().prep_image(img)

    global gif_frames
    seg_identifier = str(md.RUN_NAME) + str(md.ENV_ID) + "_" + str(md.SET_IDX) + "_" + str(md.SEG_IDX) + "_img" + str(name)
    if seg_identifier not in gif_frames:
        gif_frames[seg_identifier] = {
            "frames": [],
            "run_name": md.RUN_NAME,
            "env_id": md.ENV_ID,
            "set_idx": md.SET_IDX,
            "seg_idx": md.SEG_IDX,
            "name": name
        }

    gif_frames[seg_identifier]["frames"].append(img)


class DebugWriter():

    def __init__(self):
        self.schemas = {
            "map_overlaid": {
                "fun": write_map_overlaid
            },
            "gif_overlaid": {
                "fun": write_gif_overlaid
            },
            "gif": {
                "fun": write_gif
            }
        }

    def should_write(self):
        import rollout.run_metadata as md
        return md.IS_ROLLOUT and md.WRITE_DEBUG_DATA

    def write_schema(self, md, img, key, args):
        if key in self.schemas:
            self.schemas[key]["fun"](md, img, args)

    def write_img(self, img, key, args=None):
        import rollout.run_metadata as md
        img = torch_to_np(img)

        if key not in self.schemas:
            save_results_extra_image(md.RUN_NAME, md.ENV_ID, md.SET_IDX, md.SEG_IDX, key, img)
        else:
            self.write_schema(md, img, key, args)

    def commit(self):
        global gif_frames
        for key, gif in gif_frames.items():
            try:
                save_results_extra_gif(gif["run_name"], gif["env_id"], gif["set_idx"], gif["seg_idx"], "map_overlaid" + gif["name"], gif["frames"])
            except Exception as e:
                print("ding")
        gif_frames = {}