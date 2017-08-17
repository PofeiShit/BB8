"""Provides train network function for using in CNNs.

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
from __future__ import print_function

import numpy
import time
import math
import scipy.spatial.distance

import theano
import theano.tensor as T

from net.poseregnet import PoseRegNet
from trainer.nettrainer import NetTrainingParams, NetTrainer
from loss_functions import huber

import sys

from multiprocessing import Process, Queue, Pool


def inspect_inputs(i, node, fn):
    print("{} {}".format( i, node))
    print("input(s) value(s):")
    print("{}".format( [input[0] for input in fn.inputs]))


class PoseRegNetTrainingParams(NetTrainingParams):
    def __init__(self):
        super(PoseRegNetTrainingParams, self).__init__()


class PoseRegNetTrainer(NetTrainer):
    '''
    classdocs
    '''

    def __init__(self, poseNet=None, cfgParams=None, rng=None):
        '''
        Constructor
        
        :param poseNet: initialized DescriptorNet
        :param cfgParams: initialized PoseRegNetTrainingParams
        '''

        super(PoseRegNetTrainer, self).__init__(cfgParams)
        self.poseNet = poseNet
        self.net = poseNet
        self.cfgParams = cfgParams
        self.rng = rng

        if not isinstance(cfgParams, PoseRegNetTrainingParams):
            raise ValueError("cfgParams must be an instance of PoseRegNetTrainingParams")

        self.first_call = True
        self.setupFunctions()


    def setupFunctions(self):
        floatX = theano.config.floatX  # @UndefinedVariable

        dnParams = self.poseNet.cfgParams

        # params
        self.learning_rate = T.scalar('learning_rate', dtype=floatX)
        self.momentum = T.scalar('momentum', dtype=floatX)

        # input
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = self.poseNet.inputVar

        # targets
        y = T.matrix('y', dtype=floatX)     # R^Dx3

        # COST
        my_huber = huber(0.01)
        cost = my_huber(T.reshape(self.poseNet.output, (self.cfgParams.batch_size, self.poseNet.cfgParams.output_dim)), y).sum(axis=1)
        
        self.cost = cost.mean() # The cost to minimize

        # weight vector length for regularization (weight decay)       
        totalWeightVectorLength = 0
        for W in self.poseNet.weights:
            totalWeightVectorLength += self.cfgParams.weightreg_factor * (W ** 2).sum()
        self.cost += totalWeightVectorLength  # + weight vector norm

        # create a list of gradients for all model parameters
        self.params = self.poseNet.params
        self.grads = T.grad(self.cost, self.params)

        # euclidean mean errors over all joints
        errors = T.sqrt(T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.output_dim))-y).sum(axis=1))
        
        # mean error over full set
        self.errors = errors.mean()

        # store stuff                    
        self.y = y


    def setDataAndCompileFunctions(self, train_data, train_y, val_data, val_y):
        '''
        :param train_data: struct with train_data.x  
        '''

        self.setData(train_data, train_y, val_data, val_y)

        # setup and compile functions
        self.n_train_batches = self.getNumFullMiniBatches()
        self.n_val_batches = val_data.shape[0] / self.cfgParams.batch_size

        if self.first_call:
            # TRAIN
            self.setupTrain()

            # VALIDATE
            self.setupValidate()

            self.first_call = False

    def setupTrain(self):
        # train_model is a function that updates the model parameters by SGD

        # from: https://github.com/gwtaylor/theano-rnn/blob/master/rnn.py
        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction        
        floatX = theano.config.floatX  # @UndefinedVariable

        self.last_param_update = {}
        for param in self.params:
            initVals = numpy.zeros(param.get_value(borrow=True).shape, dtype=floatX)
            self.last_param_update[param] = theano.shared(initVals,borrow=True)

        updates = []
        for param_i, grad_i in zip(self.params, self.grads):
            last_upd = self.last_param_update[param_i]
            upd = self.momentum * last_upd - self.learning_rate * grad_i
            updates.append((param_i, param_i + upd))
            updates.append((last_upd, upd))

        batch_size = self.cfgParams.batch_size
        givens_train = {self.x: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_train[self.y] = self.train_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        givens_train[self.momentum] = numpy.array(self.cfgParams.momentum).astype(floatX)

        print("compiling train_model() ... ", end="")
        self.train_model = theano.function(inputs=[self.index, self.learning_rate],
                                           outputs=self.cost,
                                           updates=updates,
                                           givens=givens_train
                                          )  

        print("done.")


    def setupValidate(self):
        batch_size = self.cfgParams.batch_size
        givens_val = {self.x: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_val[self.y] = self.val_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        print("compiling validation_cost() ... ", end="")
        self.validation_cost = theano.function(inputs=[self.index],
                                               outputs=self.cost,
                                               givens=givens_val)
        print("done.")




