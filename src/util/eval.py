"""
This is the file for eval functions.

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
import math


def calcAngularDistance(rot_1, rot_2):
    rot_diff = np.dot(rot_1, np.linalg.inv(rot_2))
    trace = np.trace(rot_diff)
    trace = min(3.0, max(-1.0, trace))
    return 180.0 * math.acos((trace - 1.0) / 2.0)/np.pi

def calcTransDistance(t1, t2):
    return np.linalg.norm(t1 - t2)
