from data_io.instructions import get_all_instructions
from data_io.env import load_path
from learning.datasets.dynamic_ground_truth import get_dynamic_ground_truth, get_dynamic_ground_truth_smooth, get_dynamic_ground_truth_v2
import random
from learning.inputs.pose import Pose

from visualization import Presenter


def dyn_gt_test():
    presenter = Presenter()
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    all_instr = {**train_instr, **dev_instr, **test_instr}

    for i in range(10):
        path = load_path(i)
        segments = all_instr[i][0]["instructions"]
        for seg in segments:
            start_idx = seg["start_idx"]
            end_idx = seg["end_idx"]
            randInt = random.randint(10, 100)

            start_pose = Pose(path[start_idx] - randInt, 0)

            if end_idx - start_idx > 0:
                randInt = random.randint(10, 100)
                new_path = get_dynamic_ground_truth(path[start_idx:end_idx], (path[start_idx]-randInt))
                new_path1 = get_dynamic_ground_truth_smooth(path[start_idx:end_idx], (path[start_idx]-randInt))
                presenter.plot_path(i, [path[start_idx:end_idx], new_path, new_path1])
                #new_path = get_dynamic_ground_truth(path[start_idx:end_idx], (path[start_idx]-randInt))
                #new_path1 = get_dynamic_ground_truth_smooth(path[start_idx:end_idx], (path[start_idx]-randInt))
                # new_path2 = get_dynamic_ground_truth_v2(path[start_idx:end_idx], start_pose)
                # presenter.plot_path(i, [path[start_idx:end_idx], new_path2])
            # print(new_path)
            # print(path)


if __name__ == "__main__":
    dyn_gt_test()