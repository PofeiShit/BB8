"""
This is a script for testing BB8 on LINEMOD.

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
from data.LINEMOD.linemod_utils import *
from util.helpers import *
from util.MeshPly import MeshPly
import os


def load_net(filename):
    from net.poseregnet import PoseRegNet, PoseRegNetParams
    import cPickle
    rng = np.random.RandomState(23455)

    f = file(filename + ".cfg", 'rb')
    netParams = cPickle.load(f)
    f.close()

    network = PoseRegNet(rng, cfgParams=netParams)
    network.load(filename + ".weight")

    return network


def test_on_gt_det(rgb, mask, net, scale, h=128, w=128, margin_size=(64, 64)):
    m_h, m_w = margin_size=(64, 64)

    rgb_s = cv2.resize(rgb, (int(rgb.shape[1]*scale), int(rgb.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    mask_s = cv2.resize(mask, (int(mask.shape[1]*scale), int(mask.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    
    # add margin to the image 
    rgb_margin = np.zeros((rgb_s.shape[0] + 2*m_h,
                           rgb_s.shape[1] + 2*m_w, 3), dtype='uint8')
    mask_margin = np.zeros((mask_s.shape[0] + 2*m_h,
                            mask_s.shape[1] + 2*m_w, 3), dtype='uint8')
    rgb_margin[m_h:-m_h, m_w:-m_w, :] = rgb_s
    mask_margin[m_h:-m_h, m_w:-m_w, :] = mask_s
    
    # Use center of 2D mask as the center of the image window
    (ystart, xstart), (ystop, xstop) = get_mask_bb(mask_margin)
    x_det = int(xstart / 2. + xstop / 2.) - w/2.
    y_det = int(ystart / 2. + ystop / 2.) - h/2.
    image_window = crop_image(rgb_margin, (x_det, y_det), (h, w))

    # Predict 2D projection of 3D bounding box
    bb_est = net.computeOutput(image_window.swapaxes(0, 2).swapaxes(1, 2).reshape(-1, 3, h, w)/128. - 1)[0].reshape(8, 2).transpose()
    # translate 2D BB respect to the original image size
    bb_est += np.array([x_det, y_det], dtype='float32').reshape(2, 1) + np.array([w/2 - m_w, h/2 - m_h], dtype='float32').reshape(2, 1)
    bb_est /= scale
    return bb_est


def test_dataset(obj_name, path, h=128, w=128, net=None):
    internal_calibration = get_camera_intrinsic()

    path_bb = join(path, 'bounding_boxes', '{}_bb.txt'.format(obj_name.lower()))
    bb_3d = np.loadtxt(path_bb)
    bb_3d_i = np.c_[bb_3d, np.ones((bb_3d.shape[0], 1))].transpose()

    path_mesh = join(path, 'models', '{}.ply'.format(obj_name.lower()))
    mesh = MeshPly(path_mesh)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()

    scale = get_obj_scale(obj_name)

    errs_2d = []

    for idx_frame in range(get_nb_frames(obj_name)):
        #print('frame #{:04d}'.format(idx_frame))

        path_rgb = join(path, 'objects', obj_name, 'rgb', '{:04d}.jpg'.format(idx_frame))
        path_mask = join(path, 'objects', obj_name, 'mask', '{:04d}.png'.format(idx_frame))
        path_Rt = join(path, 'objects', obj_name, 'pose', '{:04d}.txt'.format(idx_frame))

        rgb = cv2.imread(path_rgb)
        mask = cv2.imread(path_mask)

        Rt_gt = get_Rt(path_Rt)
        bb_gt = compute_projection(bb_3d_i, Rt_gt, internal_calibration)

	if net:
            bb_est = test_on_gt_det(rgb, mask, net, scale, h=h, w=w)
            Rt_est = pnp(bb_3d, bb_est.transpose(), internal_calibration)
            bb_est_pnp = compute_projection(bb_3d_i, Rt_est, internal_calibration)
            draw_bb(rgb, bb_gt, color=(0, 255, 0))
            draw_bb(rgb, bb_est, color=(255, 255, 0)) 
            draw_bb(rgb, bb_est_pnp, color=(255, 0, 0))

            proj_2d_gt = compute_projection(vertices, Rt_gt, internal_calibration) 
            proj_2d_est = compute_projection(vertices, Rt_est, internal_calibration) 
            err_2d = np.mean(np.linalg.norm(proj_2d_gt - proj_2d_est, axis=0))
            errs_2d.append(err_2d)

        cv2.imshow('rgb', rgb)
        cv2.imshow('mask', mask)
        cv2.waitKey(1)

    return errs_2d


if __name__ == '__main__':
    # Image window dim
    h = 128
    w = 128

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

    path_dataset = join(os.path.dirname(__file__), '..', 'data', 'LINEMOD')
    path_dataset = '/media/mahdi/data2/linemod_toolkit/data/LINEMOD'
    net = load_net('./savedNets/BB8')
    #net = load_net('./savedNets/BB8_tiny')
    for obj_name in obj_names:
        errs_2d = test_dataset(obj_name, path_dataset, h, w, net)
        if errs_2d:
            px_threshold = 5
            acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / len(errs_2d)
            print('Acc using {} px 2D Projection on {} = {:.2f}%'.format(px_threshold, obj_name, acc))
