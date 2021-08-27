import numpy as np
import matplotlib.pyplot as plt
import re
import copy
import torch

from utils.simple_profiler import SimpleProfiler
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import hflip, crop, resize
import os


def get_display(image, xy_list, title ="", condition=[], imshow=True, border=False, delete_box=False, save=False):
    img_display = np.array(image)
    x_max, y_max = img_display.shape[:2]
    if not(xy_list is None):
        for i_pos, pos in enumerate(xy_list):
            if not(condition):
                pass
            elif condition[i_pos]:
                pass
            else:
                continue
            for ix in range(-2, 2):
                for iy in range(-2, 2):
                    img_display[min(max(0, int(pos[0]) + ix), x_max-1),
                                        min(max(0, iy + int(pos[1])), y_max-1)] = \
                        np.array([255, 0, 255])
    if border:
        img_display[20:x_max - 20, 20] = 0
        img_display[20:x_max - 20, y_max - 20] = 0
        img_display[20, 20:y_max - 20] = 0
        img_display[x_max - 20, 20:y_max - 20] = 0
    if imshow:

        if delete_box:
            fig, ax = plt.subplots(1, 1)
            img_display[:10, :10] = np.array([255, 0, 0])

            def onclick(event):
                if (event.xdata < 10) & (event.ydata < 10):
                    print("delete image")
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))

            cid = fig.canvas.mpl_connect('button_press_event', onclick)
        fig = plt.figure()
        plt.title(title)
        plt.imshow(img_display/256)
        plt.show()
        if save:
            folder = "/media/valts/shelf_space/droning/drone-sim/drones/train_real/saved_images"
            filenames = os.listdir(folder)
            if len(filenames)>0:
                i_list = [int(re.findall("[\w']+", x)[1]) for x in filenames]
                i = np.max(i_list)+1
            else:
                i = 1
            fig.savefig(os.path.join(folder,"fig {}.pdf".format(i)))
    return img_display


def rot(image, xy_list, angle):
    PROFILE = False
    prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

    #im_rot = rotate(image, angle)

    im_rot = image.rotate(angle)#affine(image, angle=angle, translate=(0, 0), scale=1, shear=0)
    prof.tick("rotation")

    width0, height0 = image.size
    width_rot, height_rot = im_rot.size
    org_center = (np.array([height0, width0])-1)/2.
    rot_center = (np.array([height_rot, width_rot])-1)/2.

    org_list = [xy-org_center for xy in xy_list]
    a = -np.deg2rad(angle)
    new_list = [np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
             -org[0]*np.sin(a) + org[1]*np.cos(a)]) for org in org_list]
    xy_list_new = [new+rot_center for new in new_list]
    prof.print_stats()
    return im_rot, xy_list_new


def rotate_and_label(img_to_transform, lm_idx, lm_pos_fpv):
    # rotate image
    angle = np.random.randint(-10, 10)
    angle_rad = np.deg2rad(angle)
    img_rotated, lm_pos_rotated = rot(img_to_transform,
                                      np.array(lm_pos_fpv), angle)
    x0 = int(round(img_rotated.size[0] * np.tan(np.abs(angle_rad))))
    y0 = int(round(img_rotated.size[1] * np.tan(np.abs(angle_rad))))

    xmax = int(round(img_rotated.size[1] - x0))
    ymax = int(round(img_rotated.size[0] - y0))

    img_cropped = crop(img_rotated, x0, y0, -x0+xmax+1, -y0+ymax+1)
    lm_pos_out = []
    lm_idx_out = []
    for pos, idx in zip(lm_pos_rotated, lm_idx):
        if (pos[0] > x0) & (pos[0] < xmax) & \
                (pos[1] > y0) & (pos[1] < ymax):
            pos_cropped = np.array([pos[0]-x0, pos[1]-y0])
            lm_pos_out.append(pos_cropped)
            lm_idx_out.append(idx)
    lm_indices_out = np.array(lm_idx_out, dtype=int)
    lm_pos_fpv_out = np.array(lm_pos_out, dtype=float)

    out_img = img_cropped
    return out_img, lm_indices_out, lm_pos_fpv_out


def flip_and_label(img, lm_pos_fpv):
    ymax, _  = img.size
    img_out = hflip(img)
    lm_pos_fpv_out = np.array([[pos[0], ymax - pos[1]] for pos in np.array(lm_pos_fpv)], dtype=float)

    return img_out, lm_pos_fpv_out


def random_crop_and_label(img, lm_idx, lm_pos_fpv, ratio):
    width, height = img.size
    size = np.random.rand()*0.5+0.5

    new_height = int(height*size)
    new_width = new_height * ratio

    x0 = np.random.randint(0, height - new_height +1)
    y0 = np.random.randint(0, width - new_width +1)

    xmax = x0 + new_height - 1
    ymax = y0 + new_width - 1

    out_img = crop(img, x0, y0, new_height, new_width)
    lm_pos_out = []
    lm_idx_out = []

    for pos, idx in zip(lm_pos_fpv, lm_idx):
        if (pos[0] > x0) & (pos[0] < xmax) & \
                (pos[1] > y0) & (pos[1] < ymax):
            pos_cropped = np.array([pos[0]-x0, pos[1]-y0])
            lm_pos_out.append(pos_cropped)
            lm_idx_out.append(idx)

    lm_idx_out = np.array(lm_idx_out, dtype=int)
    lm_pos_fpv_out = np.array(lm_pos_out, dtype=float)

    return out_img, lm_idx_out, lm_pos_fpv_out


def data_augmentation(image, lm_indices, lm_pos_fpv, IMG_HEIGHT, IMG_WIDTH, eval, prof, show=False):
    width, height = image.size
    out_img = copy.copy(image)
    out_lm_indices = np.array(lm_indices)
    out_lm_pos_fpv = np.array(lm_pos_fpv)
    prof.tick("to_pil + reshape")

    ratio = width/height
    if not eval:
        p_flip, p_rot, p_crop = (0.5, 0.5, 0.5)
        if show:
            get_display(out_img, np.array(out_lm_pos_fpv), title="Original image")

        if np.random.rand() > 1 - p_flip:  # flip image

            out_img, out_lm_pos_fpv = flip_and_label(out_img, out_lm_pos_fpv)
            prof.tick("flip")
            if show:
                get_display(out_img, out_lm_pos_fpv, title="Flipped image")

        if np.random.rand() > 1 - p_rot:  # rotate image
            out_img, out_lm_indices, out_lm_pos_fpv = rotate_and_label(out_img, out_lm_indices, out_lm_pos_fpv)
            prof.tick("rotate")
            if show:
                get_display(out_img, out_lm_pos_fpv, title="Rotated image")

        if len(out_lm_indices) > 0:
            if np.random.rand() > 1 - p_crop:  # crop image
                out_img, out_lm_indices, out_lm_pos_fpv = random_crop_and_label(out_img, out_lm_indices, out_lm_pos_fpv, ratio)
                prof.tick("crop")
                if show:
                    get_display(out_img, out_lm_pos_fpv, title="Cropped image")

    old_width, old_height = out_img.size
    if len(out_lm_pos_fpv) > 0:
        out_lm_pos_fpv[:, 0] = out_lm_pos_fpv[:, 0] * IMG_HEIGHT / old_height
        out_lm_pos_fpv[:, 1] = out_lm_pos_fpv[:, 1] * IMG_WIDTH / old_width
    out_img = resize(out_img, (IMG_HEIGHT, IMG_WIDTH)) #scipy.misc.imresize(out_img, (IMG_HEIGHT, IMG_WIDTH))
    prof.tick("resize")
    if show:
        get_display(out_img, out_lm_pos_fpv, title="Final image")

    prof.tick("img to torch")

    if show:
        plt.close("all")

    return (out_img, out_lm_indices, out_lm_pos_fpv)
