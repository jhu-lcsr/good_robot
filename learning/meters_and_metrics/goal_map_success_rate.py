import torch

def goal_map_success_rate(goal_maps_pred, goal_maps_gt):
    bs = goal_maps_pred.shape[0]
    _, argmax_pred_goal = goal_maps_pred.view(bs, -1).max(1)
    _, argmax_gt_goal = goal_maps_gt.view(bs, -1).max(1)

    pred_stop_pos_x = argmax_pred_goal / goal_maps_pred.shape[1]
    pred_stop_pos_y = argmax_pred_goal % goal_maps_pred.shape[1]
    pred_stop_pos = torch.stack([pred_stop_pos_x, pred_stop_pos_y], dim=1)

    gt_stop_pos_x = argmax_gt_goal / goal_maps_gt.shape[1]
    gt_stop_pos_y = argmax_gt_goal % goal_maps_gt.shape[1]
    gt_stop_pos = torch.stack([gt_stop_pos_x, gt_stop_pos_y], dim=1)

    dst_to_best_stop = torch.norm((pred_stop_pos.float() - gt_stop_pos.float()), dim=1)

    # TODO: Grab from config!
    good = dst_to_best_stop < 3.2
    num_good = good.sum()
    sucess_rate = num_good.float() / goal_maps_pred.shape[0]

    return sucess_rate
