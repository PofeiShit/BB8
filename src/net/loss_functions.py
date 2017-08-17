"""
This is the file for activation functions.

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

def huber(delta):
    """
    Huber loss, robust at 0
    :param delta: delta parameter
    :return: loss value
    """
    import theano.tensor as T

    def inner(target, output):
        d = target - output
        a = .5 * d**2
        b = delta * (T.abs_(d) - delta / 2.)
        l = T.switch(T.abs_(d) <= delta, a, b)
        return l
    return inner
