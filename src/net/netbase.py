"""Provides NetBase class for generating networks from configurations.

NetBase provides interface for building CNNs.
It should be inherited by all network classes in order to provide
basic functionality, ie computing outputs, creating computational
graph, etc.
NetBaseParams is the parametrization of these NetBase networks.

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

import time
import numpy
import cPickle
import theano
import theano.tensor as T
from net.convpoollayer import ConvPoolLayer, ConvPoolLayerParams
from net.hiddenlayer import HiddenLayer, HiddenLayerParams

__author__ = "Mahdi Rad <mahdi.rad@icg.tugraz.at>"
__copyright__ = "Copyright 2017, ICG, Graz University of Technology, Austria"
__credits__ = ["Mahdi Rad", "Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Mahdi Rad"
__email__ = "mahdi.rad@icg.tugraz.at, radmahdi@gmail.com"
__status__ = "Development"


class NetBaseParams(object):
    def __init__(self):
        """
        Init the parametrization
        """

        self.numInputs = 1
        self.numOutputs = 1
        self.layers = []
        self.inputDim = None
        self.outputDim = None

    def getMemoryRequirement(self):
        """
        Get memory requirements of weights
        :return: memory requirement
        """
        mem = 0
        for l in self.layers:
            mem += l.getMemoryRequirement()
        return mem


class NetBase(object):
    def __init__(self, rng, inputVar, cfgParams, twin=None):
        """
        Initialize object by constructing the layers
        :param rng: random number generator
        :param inputVar: input variable
        :param cfgParams: parameters
        :param twin: determine to copy layer @deprecated
        :return: None
        """

        self.inputVar = inputVar
        self.cfgParams = cfgParams

        # create network
        self.layers = []
        self.params = []
        self.weights = []
        for i, layerParam in enumerate(self.cfgParams.layers):
            # first input is inputVar, otherwise input is output of last one
            if i == 0:
                inp = inputVar
            else:
                # flatten output from conv to hidden layer
                if ((isinstance(self.layers[-1], ConvPoolLayer)) and (
                    isinstance(layerParam, HiddenLayerParams))):
                    inp = self.layers[-1].output.flatten(2)
                else:
                    inp = self.layers[-1].output

            id = layerParam.__class__.__name__[:-6]
            constructor = globals()[id]

            copyLayer = None if (twin is None) else twin.layers[i]

            self.layers.append(constructor(rng,
                                           inputVar=inp,
                                           cfgParams=layerParam,
                                           copyLayer=copyLayer,
                                           layerNum=i))

            self.params += self.layers[-1].params
            self.weights += self.layers[-1].weights

        # assemble externally visible parameters
        self.output = self.layers[-1].output



    def __str__(self):
        """
        prints the parameters of the layers of the network
        :return: configuration string
        """

        cfg = "Network configuration:\n"
        i = 0
        for l in self.layers:
            cfg += "Layer {}: {} with {} \n".format(i, l.__class__.__name__, l)
            i += 1

        return cfg



    def computeOutput(self, input):
        """
        compute the output of the network for given input
        :param inputs: input data
        :return: output of the network
        """

        floatX = theano.config.floatX  # @UndefinedVariable
        batch_size = self.cfgParams.batch_size
        nSamp = input.shape[0]
        descrLen = self.cfgParams.outputDim[1]
        padSize = int(batch_size * numpy.ceil(nSamp / float(batch_size)))
        out = numpy.zeros((padSize, descrLen))
        shape = list(input.shape)
        shape[0] = padSize
        input_pad = numpy.zeros(tuple(shape), dtype=floatX)
        input_pad[0:input.shape[0]] = input
        if not hasattr(self, 'compiledComputeOutputFast'):
            self.compiledComputeOutputFast = True

            index = T.lscalar()
            # zeropad to batch size

            self.input_data = theano.shared(input_pad[0:batch_size], borrow=True)

            print("compiling compute_output() ...")
            self.compute_output = theano.function(inputs=[index],
                                             outputs=self.layers[-1].output,
                                             givens={
                                             self.inputVar: self.input_data[index * batch_size:(index + 1) * batch_size]})

            print("done")


        # iterate to save memory
        n_test_batches = input_pad.shape[0] / batch_size
        start = time.time()
        for i in range(n_test_batches):
            self.input_data.set_value(input_pad[i * batch_size:(i + 1) * batch_size], borrow=True)
            #print(compute_output(0)[0, :])
            out[i * batch_size:(i + 1) * batch_size] = self.compute_output(0)
        end = time.time()
        #print("{} in {}s".format(padSize, end - start))\
        return out[0:nSamp]


    def save(self, filename):
        """
        Save the state of this network to a pickle file on disk.
        :param filename: Save the parameters of this network to a pickle file at the named path.
        :return: None
        """

        state = dict([('class', self.__class__.__name__), ('network', self.__str__())])
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'layerNum'):
                key = '{}-values'.format(layer.layerNum)
            else:
                key = '{}-values'.format(idx)

            state[key] = [p.get_value() for p in layer.params]
        handle = open(filename, 'wb')
        cPickle.dump(state, handle, -1)
        handle.close()
        print('Saved model parameter to {}'.format(filename))


    def load(self, filename):
        """
        Load the parameters for this network from disk.
        :param filename: Load the parameters of this network from a pickle file at the named path.
        :return: None
        """

        opener = open
        handle = open(filename, 'rb')
        saved = cPickle.load(handle)

        handle.close()

        for idx, layer in enumerate(self.layers):

            if hasattr(layer, 'layerNum'):
                for p, v in zip(layer.params, saved['{}-values'.format(layer.layerNum)]):
                    p.set_value(v)
            else:
                for p, v in zip(layer.params, saved['{}-values'.format(idx)]):
                    p.set_value(v)
        print('Loaded model parameters from {}'.format(filename))


    def load_vgg(self):
        """
        Load the parameters for this network from disk.
        :param filename: Load the parameters of this network from a pickle file at the named path.
        :return: None
        """
        from os.path import join
        filename = join('weights', 'vgg.weight')
        handle = open(filename, 'rb')
        saved = cPickle.load(handle)

        handle.close()
        nb_layers_to_use = 6
        for idx, layer in enumerate(self.layers):
            #if idx > 9:		# to finetune last 3 conv layers, for nb_layers_to_use = 6
            if idx > nb_layers_to_use:  # to retrain last 3 conv layers, for nb_layers_to_use = 6.
                break
            if hasattr(layer, 'layerNum'):
                for p, v in zip(layer.params, saved['{}-values'.format(layer.layerNum)]):
                    p.set_value(v)
            else:
                for p, v in zip(layer.params, saved['{}-values'.format(idx)]):
                    p.set_value(v)
        self.params = self.params[(nb_layers_to_use + 1)*2:]
        print(self.params)
        print('Loaded model parameters from {}'.format(filename))



