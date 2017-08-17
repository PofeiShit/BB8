"""
This is the file for th objects and camera intrinsic of the LINEMOD dataset for BB8.

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

import numpy as np
import cv2
from os.path import join
import os

data_path = join(os.path.dirname(__file__), '.')


def get_nb_frames(obj_name):
    dict_nb_frames = {'ape': 1236,
                      'benchvise': 1215,
                      'driller': 1188,
                      'cam': 1201,
                      'can': 1196,
                      'iron': 1152,
                      'lamp': 1227,
                      'phone': 1225,
                      'cat': 1179,
                      'holepuncher': 1237,
                      'duck': 1254,
                      'cup': 1240,
                      'bowl': 1233,
                      'eggbox': 1253,
                      'glue': 1220
                      }

    if obj_name.lower() in dict_nb_frames:
        return dict_nb_frames[obj_name.lower()]
    else:
        raise ValueError('invalid object name: {}'.format(obj_name))


def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float32')
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.
    return K


def get_Rt(filename):
    Rt = np.loadtxt(filename, dtype='float32')[:3, :]
    return Rt


def load_object_instances(obj_name=None):
    assert obj_name is not None, 'obj_name cannot be None'

    obj = []
    scale = get_obj_scale(obj_name)

    # Use 15 % of LINEMOD images for training, and the reset for testing
    training_range_path = join(data_path, 'training_range', '{}.txt'.format(obj_name))
    training_range = np.loadtxt(training_range_path, dtype='int32')
    print('loading {} images to memory...'.format(obj_name))
    for idx in training_range:
            path_rgb = join(data_path, 'objects', obj_name, 'rgb', '{:04d}.jpg'.format(idx))
            path_mask = join(data_path, 'objects', obj_name, 'mask', '{:04d}.png'.format(idx))
            path_Rt = join(data_path, 'objects', obj_name, 'pose', '{:04d}.txt'.format(idx))

            rgb = cv2.imread(path_rgb)
            mask = cv2.imread(path_mask)

            obj_i = {'rgb': rgb,
                     'mask': mask,
                     'Rt': get_Rt(path_Rt),
                     'K': get_camera_intrinsic(),
                     's': scale}
            obj.append(obj_i)  

    bb_3d = load_bb_3d(obj_name=obj_name)
    print('done!')

    return obj, bb_3d


def load_bb_3d(obj_name=None):
    assert obj_name is not None, 'obj_name cannot be None'
    path_bb = join(data_path, 'bounding_boxes', '{}_bb.txt'.format(obj_name))
    bb_3d = np.loadtxt(path_bb)
    bb_3d_i = np.ones((8, 4))
    bb_3d_i[:, :-1] = bb_3d
    bb_3d_i = bb_3d_i.transpose()
    return bb_3d_i


def get_obj_scale(obj_name=None):
    assert obj_name is not None, 'obj_name cannot be None'
    dim_in = [480, 640]
    if obj_name in ['benchvise', 'cam', 'can', 'driller', 'iron', 'lamp']:
        dim_out = [240, 320]
    else:
        dim_out = [360, 480]
    scale = dim_out[0]*1.0/dim_in[0]
    return scale
