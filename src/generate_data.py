"""Provides a script to augment training images for BB8

Copyright 2017 Mahdi Rad, ICG,
Graz University of Technology <mahdi.rad@icg.tugraz.at>

This file is part of BB8.

BB8 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BB8 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BB8.  If not, see <http://www.gnu.org/licenses/>.
"""

import random
import numpy as np
import cv2
from util.helpers import *
import os

# instances of the objects
objs = None
# 3D bounding boxes of the object
bbs_3d = None
# number of background images to load to the memory to speed up the process
nb_bg_images = 15           # JUST FOR TEST
#nb_bg_images = 150000      # USE THIS NUMBER TO TRAIN BB8

# Image window size
c = 3
h = 128
w = 128
# 8 corners of 2D projection of 3D bounding box (8*2)
output_dim = 16

# Path to background images (ImageNet)
background_images = np.zeros((nb_bg_images, h, w, c), dtype='uint8')
bg_path = join(os.path.dirname(__file__), '..', 'data', 'BG')

# Apply scale changes to augment training data
apply_scale = True
# final scale would be in [1 - scale/10., 1+ scale/10.]
scale = 2

# shifted by some pixels from center of the image to handle the inaccuracy of the detection
x_shift = 10
y_shift = 10


def init():
    pass


def get_dim():
    return c, h, w, output_dim


def add_obj_to_img(dst_img, dst_img_mask, obj_rgb, obj_mask, index, bb_3d, Rt, K):
    """
        Add instance of the object to the center of the destination image.
        :param dst_img: destination rgb image
        :param dst_img_mask: destination mask image
        :param obj_rgb: rgb obj to add to the image
        :param obj_mask: mask object to add to dst_img_mask
        :param index: index of the object
        :param bb_3d: 3D bounding box
        :param Rt: transformation matrix - object pose
        :param K: camera intrinsic parameters
        :return: 2D projection of 3D bounding box
    """
    # crop obj from obj_rgb
    (y_start, x_start), (y_stop, x_stop) = get_mask_bb(obj_mask)
    obj_w = x_stop - x_start
    obj_h = y_stop - y_start

    if obj_w % 2 == 1:
        obj_w += 1
        x_stop += 1
    if obj_h % 2 == 1:
        obj_h += 1
        y_stop += 1

    obj_mask_cropped = obj_mask[y_start:y_stop, x_start:x_stop] / 255
    obj_rgb_cropped = obj_rgb[y_start:y_stop, x_start:x_stop, :]

    # Add obj to the center of the image
    img_h, img_w = dst_img.shape[:2]
    c_x = img_w/2
    c_y = img_h/2

    dst_img[c_y - obj_h/2:c_y + obj_h/2,
            c_x - obj_w/2:c_x + obj_w/2, :] = obj_rgb_cropped*obj_mask_cropped +\
                                              dst_img[c_y - obj_h/2:c_y + obj_h/2,
                                                      c_x - obj_w/2:c_x + obj_w/2, :] * (1 - obj_mask_cropped)

    # index zero is reserved for BG
    index += 1
    dst_img_mask[c_y - obj_h/2:c_y + obj_h/2,
                 c_x - obj_w/2:c_x + obj_w/2, :] = index*obj_mask_cropped +\
                                                   dst_img_mask[c_y - obj_h/2:c_y + obj_h/2,
                                                                c_x - obj_w/2:c_x + obj_w/2, :] * (1 - obj_mask_cropped)

    # UNCOMMENT BELOW FOR BOWL AND CUP (SYMMETRICAL OBJECTS)
    '''
    import transforms3d.euler
    R_1 = obj_transformation[:3, :3]
    alpha, beta, gamma = euler.mat2euler(R_1)
    delta_R = euler.euler2mat(0, 0, gamma)
    R_2 = R_1.dot(delta_R)
    obj_transformation[:3, :3] = R_2
    '''

    # project 3D bb to 2D and translate it to image center
    bb_proj2d = compute_projection(bb_3d, Rt, K)

    tx = c_x - obj_w / 2 - x_start
    ty = c_y - obj_h / 2 - y_start
    t = np.array([tx, ty], dtype='float32').reshape(2, 1)
    bb_proj2d = bb_proj2d + t

    return bb_proj2d


def create_data(args):
    """
        Randomly pick one instance of one random object and place it .
        :param args: args[0] is the index where to put generated image and label in the array of images (args[1]) and
        array of labels (args[2])
    """
    i = random.randint(0, len(bbs_3d) - 1)
    bb_3d_i = bbs_3d[i]
    obj_i = objs[i]

    idx = args[0]
    train_set_x = args[1]
    train_set_y = args[2]

    j = random.randint(0, len(obj_i) - 1)
    obj_ij = obj_i[j]
    obj_rgb = obj_ij['rgb']
    obj_mask = obj_ij['mask']
    obj_Rt = obj_ij['Rt']
    obj_scale = obj_ij['s']
    K = obj_ij['K']

    height_in, width_in = obj_rgb.shape[:2]

    img = np.zeros(obj_rgb.shape, dtype='uint8')
    img_mask = np.zeros(obj_rgb.shape, dtype='uint8')
    
    # For training with no occlusion, pass always index 0 
    bb_proj2d = add_obj_to_img(img, img_mask, obj_rgb, obj_mask, 0, bb_3d_i, obj_Rt, K)

    if apply_scale:
        obj_scale *= 1 + np.random.randint(-scale, scale + 1)*0.1

    img = cv2.resize(img, (int(width_in*obj_scale),
                           int(height_in*obj_scale)), interpolation=cv2.INTER_NEAREST)

    img_mask = cv2.resize(img_mask, (int(width_in*obj_scale),
                                     int(height_in*obj_scale)), interpolation=cv2.INTER_NEAREST)

    bb_proj2d *= obj_scale

    dx = np.random.randint(-x_shift, x_shift)
    dy = np.random.randint(-y_shift, y_shift)

    det_x = img.shape[1]/2 + dx
    det_y = img.shape[0]/2 + dy

    img = crop_image(img, (det_x - w/2, det_y - h/2), (w, h))
    img_mask = crop_image(img_mask, (det_x - w/2, det_y - h/2), (w, h))

    # translate projection of 3D bounding box respect to image window center
    t = np.array([det_x, det_y], dtype='float32').reshape(2, 1)
    bb_proj2d = bb_proj2d - t

    # Add random BG to the image window
    bg_i = background_images[random.randint(0, background_images.shape[0] - 1), :, :, :].copy()
    final_img = img * img_mask + bg_i * (1 - img_mask)

    train_set_x[idx] = (final_img.swapaxes(2, 0).swapaxes(2, 1)/128. - 1.)
    train_set_y[idx] = bb_proj2d.transpose().flatten()


def load_bgs_to_memory():
    """
        Load background images to the memory.
    """
    bg_file_names = get_all_files(bg_path)
    print("{0} background images are found".format(len(bg_file_names)))

    print('loading background images to memory...')
    bg_nb, bg_h, bg_w = background_images.shape[:3]
    for bg_idx in range(bg_nb):
        random_bg_index = random.randint(0, len(bg_file_names) - 1)

        bg = cv2.imread(bg_file_names[random_bg_index])
        bg = cv2.resize(bg, (int(bg_w), int(bg_h)), interpolation=cv2.INTER_NEAREST)
        background_images[bg_idx] = bg
        printProgressBar(bg_idx, bg_nb, prefix = 'Progress:', suffix = 'Complete', length = 50)

    printProgressBar(bg_nb, bg_nb, prefix = 'Progress:', suffix = 'Complete', length = 50)


def pre_create_data():
    load_objs_to_memory()
    load_bgs_to_memory()


def load_objs_to_memory():
    """
        Load instances of the objects to the memory.
    """
    from data.LINEMOD.linemod_utils import load_object_instances
    global objs
    global bbs_3d

    # List of 13 objects of LINEMOD
    '''
    obj_names = ['ape',
                 'benchvise',
                 'cam',
                 'can',
                 'cat',
                 'driller',
                 'duck',
                 'eggbox',
                 'glue',
                 'holepuncher',
                 'iron',
                 'lamp',
                 'phone']
    '''
    # Using only one object with tiny_BB8 architecture
    obj_names = ['cat']

    bbs_3d = {}
    objs = {}

    for idx, obj_name in enumerate(obj_names):
        obj_i, obj_3d_i = load_object_instances(obj_name=obj_name)
        objs[idx] = obj_i
        bbs_3d[idx] = obj_3d_i


if __name__ == '__main__':

    init()
    pre_create_data()
    nb_images = 100
    c, h, w, output_dim = get_dim()
    training_set_x = np.zeros((nb_images, c, h, w), dtype='float32')
    training_set_y = np.zeros((nb_images, output_dim), dtype='float32')

    for i in range(nb_images):
        create_data([i, training_set_x, training_set_y])

    for i in range(nb_images):
        img_i = ((training_set_x[i, :, :, :].swapaxes(0, 1).swapaxes(1, 2) + 1.) * 128).astype('uint8')
        bb_proj2d = training_set_y[i, :].reshape(8, 2).transpose()
        img_copy = img_i.copy()
        draw_bb(img_copy, bb_proj2d + np.array([h/2, w/2]).reshape(2, 1), (0, 255, 0))
        cv2.imshow('img', img_copy)
        cv2.waitKey()
