"""Provides PoseRegNet class that implements deep CNNs.

PoseRegNet provides interface for building the CNN.
PoseRegNetParams is the parametrization of these CNNs.

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

import theano.tensor as T
from net.convpoollayer import ConvPoolLayer, ConvPoolLayerParams
from net.hiddenlayer import HiddenLayer, HiddenLayerParams
from net.netbase import NetBase, NetBaseParams
from activations import ReLU

__author__ = "Mahdi Rad <mahdi.rad@icg.tugraz.at, radmahdi@gmail.com>"
__copyright__ = "Copyright 2017, ICG, Graz University of Technology, Austria"
__credits__ = ["Mahdi Rad"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Mahdi Rad"
__email__ = "mahdi.rad@icg.tugraz.at, radmahdi@gmail.com"
__status__ = "Development"


class PoseRegNetParams(NetBaseParams):
    def __init__(self, type=0, n_chan=3, w_in=128, h_in=128, batchSize=128, output_dim=16):
        """
        Init the parametrization

        :type type: int
        :param type: type of network
        """

        super(PoseRegNetParams, self).__init__()

        self.batch_size = batchSize
        self.output_dim = output_dim
        self.inputDim = (batchSize, n_chan, h_in, w_in)

        if type == 0:  # tiny - BB8

            self.layers.append(ConvPoolLayerParams(inputDim=(batchSize, n_chan, h_in, w_in), 
                                               nFilters=32,
                                               filterDim=(5, 5),
                                               poolsize=(4, 4),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=32,
                                               filterDim=(5, 5),
                                               poolsize=(2, 2),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=50,
                                               filterDim=(3, 3),
                                               poolsize=(1, 1),
                                               activation=ReLU))

            l3out = self.layers[-1].outputDim
            self.layers.append(HiddenLayerParams(inputDim=(l3out[0], l3out[1] * l3out[2] * l3out[3]),
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, output_dim),
                                                 activation=None))

            self.outputDim = self.layers[-1].outputDim
        elif type == 1: 
            # VGG - architecure
            self.layers.append(ConvPoolLayerParams(inputDim=(batchSize, n_chan, h_in, w_in), 
                                                   nFilters=64,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=64,
                                                   filterDim=(3, 3),
                                                   poolsize=(2, 2),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=128,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=128,
                                                   filterDim=(3, 3),
                                                   poolsize=(2, 2),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=256,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=256,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=256,
                                                   filterDim=(3, 3),
                                                   poolsize=(2, 2),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=512,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=512,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=512,
                                                   filterDim=(3, 3),
                                                   poolsize=(2, 2),
                                                   activation=ReLU))
            
            # Last 3 conv. layers of VGG-16 archituecture are removed from BB8,
            # because of having smaller input patches
            '''
            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=512,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=512,
                                                   filterDim=(3, 3),
                                                   poolsize=(1, 1),
                                                   activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                                   nFilters=512,
                                                   filterDim=(3, 3),
                                                   poolsize=(2, 2),
                                                   activation=ReLU))
            '''


            l3out = self.layers[-1].outputDim
            self.layers.append(HiddenLayerParams(inputDim=(l3out[0], l3out[1] * l3out[2] * l3out[3]),
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, output_dim),
                                                 activation=None))  # last one is linear for regression

            self.outputDim = self.layers[-1].outputDim
        else:
            raise NotImplementedError("not implemented")


class PoseRegNet(NetBase):
    def __init__(self, rng, inputVar=None, cfgParams=None):
        """
        :type cfgParams: PoseRegNet
        """

        if cfgParams is None:
            raise Exception("Cannot create a Net without config parameters (ie. cfgParams==None)")

        if inputVar is None:
            inputVar = T.tensor4('x')  # input variable
        elif isinstance(inputVar, str):
            inputVar = T.tensor4(inputVar)  # input variable

        # create structure
        super(PoseRegNet, self).__init__(rng, inputVar, cfgParams)
