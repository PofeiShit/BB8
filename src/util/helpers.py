"""
This is the file for diverse helper functions.

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

import numpy

__author__ = "Mahdi Rad <mahdi.rad@icg.tugraz.at>"
__copyright__ = "Copyright 2017, ICG, Graz University of Technology, Austria"
__credits__ = ["Mahdi Rad"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Mahdi Rad"
__email__ = "mahdi.rad@icg.tugraz.at"
__status__ = "Development"

import numpy as np
from os import listdir
from os.path import isfile, join
import cv2


def get_all_files(directory):
    files = []

    for f in listdir(directory):
        if isfile(join(directory, f)):
            files.append(join(directory, f))
        else:
            files.extend(get_all_files(join(directory, f)))
    return files


def draw_bb(src, points, color=None):
    line_w = 2
    if color is None:
        color = (0, 255, 0)
    for j in range(points.shape[1]):
        cv2.circle(src, (int(points[0, j]), int(points[1, j])), line_w, color, -1)
    for j in range(4):
        cv2.line(src, (int(points[0, j]), int(points[1, j])),
                 (int(points[0, (j + 1) % 4]), int(points[1, (j + 1) % 4])), color, line_w, lineType=cv2.CV_AA)
    for j in range(4, 8):
        cv2.line(src, (int(points[0, j]), int(points[1, j])),
                 (int(points[0, (j + 1) % 4 + 4]), int(points[1, (j + 1) % 4 + 4])), color, line_w, lineType=cv2.CV_AA)
    for j in range(4):
        cv2.line(src, (int(points[0, 0 + j]), int(points[1, 0 + j])),
                      (int(points[0, 4 + j]), int(points[1, 4 + j])), color, line_w, lineType=cv2.CV_AA)


def crop_image(src, top_left, size):
    x = int(top_left[0])
    y = int(top_left[1])
    w = int(size[0])
    h = int(size[1])
    if len(src.shape) == 2:
        dst = src[y:y + h, x:x + w]
    else:
        dst = src[y:y + h, x:x + w, :]
    return dst


def get_mask_bb(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    bb = np.argwhere(mask)
    (y_start, x_start), (y_stop, x_stop) = bb.min(0), bb.max(0) + 1
    return (y_start, x_start), (y_stop, x_stop)


def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '='):
    import sys
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total: 
        print()


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of veritces'
    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs)

    R, _ = cv2.Rodrigues(R_exp)
    Rt = np.c_[R, t]

    return Rt



