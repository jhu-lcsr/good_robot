def get_obs_mask_every_n(ever_n_steps, N):
    mask = []
    for i in range(N):
        mask.append(i % ever_n_steps == 0)
    return mask


def get_obs_mask_segstart(segment_data):
    mask = []
    seg_idx = -1
    for i in range(len(segment_data)):
        if segment_data[i] is not None and segment_data[i]["metadata"]["seg_idx"] != seg_idx:
            mask.append(True)
            seg_idx = segment_data[i]["metadata"]["seg_idx"]
        else:
            mask.append(False)
    return mask

def get_obs_mask_every_n_and_segstart(ever_n_steps, segment_metadata):
    mask_n = get_obs_mask_every_n(ever_n_steps, len(segment_metadata))
    mask_start = get_obs_mask_segstart(segment_metadata)
    mask = [a or b for a, b in zip(mask_n, mask_start)]
    return mask