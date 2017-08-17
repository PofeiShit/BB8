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
import errno

import theano
import theano.tensor as T

import sys

from multiprocessing import Process, Queue, Pool
import sharedmem
import itertools
from sigtrainer import SIGTrainer

class NetTrainingParams(object):
    def __init__(self):
        self.batch_size = 128
        self.momentum = 0.9
        self.learning_rate_steps = []  
        self.learning_rate_scales = []
        self.weightreg_factor = 0.001  # regularization on the weights


class NetTrainer(object):
    """
    Basic class for different trainers that handels general memory management.
    Full training data (must be in RAM) is divided into chunks (macro batches) that are transferred to the GPU memory
    in blocks. Each macro batch consists of mini batches that are processed at once. If a mini batch is requested,
    which is not in the GPU memory, the macro block is automatically transferred.
    """

    def __init__(self, cfgParams):
        """
        Constructor
        :param cfgParams: initialized NetTrainingParams
        """

        self.cfgParams = cfgParams

        if not isinstance(cfgParams, NetTrainingParams):
            raise ValueError("cfgParams must be an instance of NetTrainingParams")

        # get GPU memory info
        mem_info = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        from theano.sandbox.rng_mrg import MRG_RandomStreams

        self.GPUMem = (mem_info[0] / 1024 ** 2) / 2.  # MB, use half of free memory

        print("GPU:" + str(self.GPUMem))
        self.RAMMem = 10000  # MB
        self.currentMacroBatch = 0  # current batch on GPU
        self.trainSize = 0
        self.sampleSize = 0
        self.para_load = False

        self.killer = SIGTrainer()	# CTRL + C: stop training after finishing the epoch

    def setData(self, train_data, train_y, val_data, val_y, show_msg=True):
        """
        Set the data of the network for the first time, assuming train size << val size
        :param train_data: training data
        :param train_y: training labels
        :param val_data: validation data
        :param val_y: validation labels
        :return: None
        """

        # check sizes
        if (train_data.shape[0] != train_y.shape[0]) or (val_data.shape[0] != val_y.shape[0]):
            raise ValueError("Number of samples must be the same as number of labels")

        self.trainSize = train_data.nbytes / 1024. / 1024.
        self.sampleSize = self.trainSize / train_data.shape[0]
        self.numTrainSamples = train_data.shape[0]
        self.numValSamples = val_data.shape[0]

        # at least one minibatch per macro
        assert self.GPUMem > self.sampleSize*self.cfgParams.batch_size

        # keep backup of original data
        # self.traindataDB = self.alignData(train_data)
        # self.trainyDB = self.alignData(train_y)
        # pad last macro batch separately to save memory
        self.traindataDB = train_data[0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
        self.traindataDBlast = self.alignData(train_data[(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():])
        self.trainyDB = train_y[0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
        self.trainyDBlast = self.alignData(train_y[(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():])
        # no need to cache validation data
        self.valdataDB = val_data
        self.valyDB = val_y

        if show_msg:
            print("Train size: {}MB, GPU memory available: {}MB, sample size: {}MB".format(self.trainSize, self.GPUMem,
                                                                                           self.sampleSize))
            print("{} samples, batch size {}".format(train_data.shape[0], self.cfgParams.batch_size))
            print("{} macro batches, {} mini batches per macro, {} full mini batches total".format(self.getNumMacroBatches(),
                                                                                              self.getNumMiniBatchesPerMacroBatch(),
                                                                                              self.getNumMiniBatches()))
        # shared variable already exists?
        if hasattr(self, 'train_data_x'):
            if show_msg:
                print("Reusing shared variables!")
            if self.trainSize > self.getGPUMemAligned():
                if show_msg:
                    print("Loading {} macro batches a {}MB".format(self.getNumMacroBatches(), self.getGPUMemAligned()))
                # load first macro batch
                idx = self.getNumSamplesPerMacroBatch()
                self.replaceTrainingData(train_data[:idx], train_y[:idx])
                self.replaceValData(val_data, val_y)
            else:
                if show_msg:
                    print("Loading single macro batch {}/{}MB".format(self.trainSize, self.getGPUMemAligned()))
                self.replaceTrainingData(train_data, train_y)
                self.replaceValData(val_data, val_y)
        else:
            # load shared data
            if self.trainSize > self.getGPUMemAligned():
                if show_msg:
                    print("Loading {} macro batches a {}MB".format(self.getNumMacroBatches(), self.getGPUMemAligned()))
                # load first macro batch
                idx = self.getNumSamplesPerMacroBatch()
                self.train_data_x = theano.shared(train_data[:idx], name='train_data_x', borrow=True)
                self.train_data_y = theano.shared(train_y[:idx], name='train_data_y', borrow=True)
                self.val_data_x = theano.shared(val_data[:idx], name='val_data_x', borrow=True)
                self.val_data_y = theano.shared(val_y[:idx], name='val_data_y', borrow=True)
            else:
                if show_msg:
                    print("Loading single macro batch {}/{}MB".format(self.trainSize, self.getGPUMemAligned()))
                self.train_data_x = theano.shared(train_data, name='train_data_x', borrow=True)
                self.train_data_y = theano.shared(train_y, name='train_data_y', borrow=True)
                self.val_data_x = theano.shared(val_data, name='val_data_x', borrow=True)
                self.val_data_y = theano.shared(val_y, name='val_data_y', borrow=True)


    def set_data_para(self, train_data, train_y, val_data, val_y):
        self.traindataDB = train_data[0:(self.getNumMacroBatches() - 1) * self.getNumSamplesPerMacroBatch()]

        self.traindataDBlast[:train_data.shape[0] - (self.getNumMacroBatches() - 1) * self.getNumSamplesPerMacroBatch()] = train_data[(self.getNumMacroBatches() - 1) * self.getNumSamplesPerMacroBatch():]
        self.trainyDB = train_y[0:(self.getNumMacroBatches() - 1) * self.getNumSamplesPerMacroBatch()]
        self.trainyDBlast[:train_data.shape[0] - (self.getNumMacroBatches() - 1) * self.getNumSamplesPerMacroBatch()] = train_y[(self.getNumMacroBatches() - 1) * self.getNumSamplesPerMacroBatch():]
        # no need to cache validation data
        self.valdataDB = val_data
        self.valyDB = val_y

        if self.trainSize > self.getGPUMemAligned():
            # load first macro batch
            idx = self.getNumSamplesPerMacroBatch()
            self.replaceTrainingData(train_data[:idx], train_y[:idx])
            self.replaceValData(val_data, val_y)
        else:
            self.replaceTrainingData(train_data, train_y)
            self.replaceValData(val_data, val_y)

    def replaceTrainingData(self, train_data, train_y):
        """
        Replace the shared data of the training data
        :param train_data: new training data
        :param train_y: new training labels
        :return: None
        """
        self.train_data_x.set_value(train_data, borrow=True)
        self.train_data_y.set_value(train_y, borrow=True)

    def replaceValData(self, val_data, val_y):
        """
        Replace the shared data of the validation data, should not be necessary
        :param val_data: new validation data
        :param val_y: new validation labels
        :return: None
        """
        self.val_data_x.set_value(val_data, borrow=True)
        self.val_data_y.set_value(val_y, borrow=True)

    def alignData(self, data):
        """
        Align data to a multiple of the macro batch size
        :param data: data for alignment
        :return: padded data
        """
        # pad with zeros to macro batch size, but only along dimension 0 ie samples
        topad = self.getNumSamplesPerMacroBatch() - data.shape[0] % self.getNumSamplesPerMacroBatch()
        sz = []
        sz.append((0, topad))
        for i in range(len(data.shape) - 1):
            sz.append((0, 0))
        return numpy.pad(data, sz, mode='constant', constant_values=0)

    def getSizeMiniBatch(self):
        """
        Get the size of a mini batch in MB
        :return: size of mini batch in MB
        """
        return self.cfgParams.batch_size * self.sampleSize

    def getSizeMacroBatch(self):
        """
        Get the size of a macro batch in MB
        :return: size of macro batch in MB
        """
        return self.getNumMacroBatches() * self.getSizeMiniBatch()

    def getNumFullMiniBatches(self):
        """
        Get total number of completely filled mini batches. drop last minibatch otherwise we might get problems with the zeropadding (all then zeros that are learnt)
        :return: number of training samples
        """
        return int(numpy.floor(self.trainSize / self.sampleSize / self.cfgParams.batch_size))

    def getNumMiniBatches(self):
        """
        Get total number of mini batches, including zero-padded patches
        :return: number of training samples
        """
        return int(numpy.ceil(self.trainSize / self.sampleSize / self.cfgParams.batch_size))

    def getNumMacroBatches(self):
        """
        Number of macro batches necessary for handling the training size
        :return: number of macro batches
        """
        return int(numpy.ceil(self.trainSize / float(self.getGPUMemAligned())))

    def getNumMiniBatchesPerMacroBatch(self):
        """
        Get number of mini batches per macro batch
        :return: number of mini batches per macro batch
        """
        return int(self.getGPUMemAligned() / self.sampleSize / self.cfgParams.batch_size)

    def getNumSamplesPerMacroBatch(self):
        """
        Get number of mini batches per macro batch
        :return: number of mini batches per macro batch
        """
        return int(self.getNumMiniBatchesPerMacroBatch() * self.cfgParams.batch_size)

    def getGPUMemAligned(self):
        """
        Get the number of MB of aligned GPU memory, aligned for full mini batches
        :return: usable size of GPU memory in MB
        """
        return self.sampleSize * self.cfgParams.batch_size * int(
            self.GPUMem / float(self.sampleSize * self.cfgParams.batch_size))

    def loadMiniBatch(self, mini_idx):
        """
        Makes sure that the mini batch is loaded in the shared variable
        :param mini_idx: mini batch index
        :return: index within macro batch
        """
        macro_idx = int(mini_idx / self.getNumMiniBatchesPerMacroBatch())
        self.loadMacroBatch(macro_idx)
        return mini_idx % self.getNumMiniBatchesPerMacroBatch()

    def loadMacroBatch(self, macro_idx):
        """
        Make sure that macro batch is loaded in the shared variable
        :param macro_idx: macro batch index
        :return: None
        """
        if macro_idx != self.currentMacroBatch:
            # last macro batch is handled separately, as it is padded
            if self.isLastMacroBatch(macro_idx):
                start_idx = 0
                end_idx = self.getNumSamplesPerMacroBatch()
                self.replaceTrainingData(self.traindataDBlast[start_idx:end_idx], self.trainyDBlast[start_idx:end_idx])
                # remember current macro batch index
                self.currentMacroBatch = macro_idx
            else:
                start_idx = macro_idx * self.getNumSamplesPerMacroBatch()
                end_idx = min((macro_idx + 1) * self.getNumSamplesPerMacroBatch(), self.traindataDB.shape[0])
                self.replaceTrainingData(self.traindataDB[start_idx:end_idx], self.trainyDB[start_idx:end_idx])
                # remember current macro batch index
                self.currentMacroBatch = macro_idx

    def loadMacroBatchMP(self, recv_queue, send_queue, data_queue):
        """
        Function which is started as thread, that loads and prepares macro batches.
        It can further be used to load augmented data from the macro batches, with doing the augmentation on CPU,
        and the calculations in parallel on the GPU.
        :param recv_queue: recv_queue is only for receiving
        :param send_queue: send_queue is only for sending
        :return: None
        """

        new_data = {}
        for var in self.trainingVar:
            if not hasattr(self, var):
                raise ValueError("Variable " + var + " not defined!")
            if var.startswith("train_"):
                new_data[var] = numpy.zeros_like(getattr(self, var + 'DB')[0:self.getNumSamplesPerMacroBatch()])

        while True:
            # getting the macro batch index to load
            (macro_idx, macro_params) = recv_queue.get()

            # kill signal
            if macro_idx == -1:
                return

            if self.isLastMacroBatch(macro_idx):
                start_idx = 0
                end_idx = self.cfgParams.batch_size * (
                self.getNumFullMiniBatches() - self.getNumMiniBatchesPerMacroBatch() * (self.getNumMacroBatches() - 1))
                print("Loading last macro batch {}, start idx {}, end idx {}".format(macro_idx, start_idx, end_idx))
                last = True
            else:
                start_idx = macro_idx * self.getNumSamplesPerMacroBatch()
                end_idx = min((macro_idx + 1) * self.getNumSamplesPerMacroBatch(), self.train_data_xDB.shape[0])
                print("Loading macro batch {}, start idx {}, end idx {}".format(macro_idx, start_idx, end_idx))
                last = False

            # invoke function to generate new data
            assert macro_params['fun'] is not None
            getattr(self, macro_params['fun'])(macro_params, macro_idx, last, range(start_idx, end_idx), new_data)

            # put new data in data queue
            data_queue.put((macro_idx, new_data))
            send_queue.put(self.SYNC_BATCH_FINISHED)

    def isLastMacroBatch(self,macro_idx):
        """
        Check if macro batch is last macro batch
        :param macro_idx: macro batch index
        :return: True if batch is last macro batch
        """

        return macro_idx >= self.getNumMacroBatches()-1 #zero index

    def train(self, n_epochs=50, storeFilters=False):
        if self.para_load:
            queue_l2t = self.config['queue_l2t']
            queue_t2l = self.config['queue_t2l']

        wvals = []

        # training procedure:
        # n epochs, batch_size, validation
        n_train_batches = self.n_train_batches
        n_val_batches = self.n_val_batches
        # n_test_batches = self.n_test_batches

        # early-stopping parameters
        patience = n_epochs * n_train_batches / 2  # look as this many batches regardless (do at least half of the epochs we were asked for)
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)
        # go through this many minibatches before checking the network on the validation set; in this case we
        # check every epoch
        learning_rate = self.cfgParams.learning_rate
        steps = self.cfgParams.learning_rate_steps
        scales = self.cfgParams.learning_rate_scales

        best_validation_loss = numpy.inf

        start_time = time.clock()

        train_costs = []
        validation_errors = []
        done_looping = False
        epoch = 0
        step_idx = 0

        print("------------------------------------------------------------------------------")
        print("|  EPOCH  |  MINIBATCH  |  AVG COST  |   %  |  VAL COST  |   LR   |    e/s   |")
        while (epoch < n_epochs) and (not done_looping) and (not self.killer.kill_now):
            epoch = epoch + 1
            if steps != [] and step_idx < len(steps) and steps[step_idx]== epoch:
                learning_rate *= scales[step_idx]
                step_idx += 1 
            train_costs_epoch = []
            validation_error_epoch = []
            if self.para_load:
                while True:
                    if queue_l2t.empty():
                        if epoch == 1:
                            pass  # Wait for load_para to load next partition
                        else:
                            break
                    else:
                        msg = queue_l2t.get()
                        if msg == 'load_done':
                            self.set_new_partition()
                        elif msg == 'ready_training':
                            pass
                        else:
                            assert False, msg
                        queue_t2l.put('training')
                        break
            start_time_epoch = time.clock()
            for minibatch_index in xrange(n_train_batches):
                # call parent to make sure batch is loaded
                mini_idx = self.loadMiniBatch(minibatch_index)

                minibatch_avg_cost = self.train_model(mini_idx, learning_rate)

                if math.isnan(minibatch_avg_cost):
                    print("minibatch {0:4d}, average cost: NaN".format(minibatch_index))
                    # check which vars are nan
                    self.checkNaNs()
                    assert False

                sys.stdout.write("\r|   {0:3d}   |    {1:4d}     |   {2}  | {3:3d}  |".format(epoch,
                                                                                              minibatch_index,
                                                                                              round(minibatch_avg_cost,
                                                                                                    5),
                                                                                              int(
                                                                                                  100 * minibatch_index / n_train_batches)))
                sys.stdout.flush()
                train_costs_epoch.append(minibatch_avg_cost)
                iter_count = (epoch - 1) * n_train_batches + minibatch_index

                if (iter_count + 1) % validation_frequency == 0:
                    if storeFilters:
                        wval = self.net.layers[0].W.get_value()
                        wvals.append(wval)

                    # compute errors on training data
                    validation_losses = [self.validation_cost(i) for i in xrange(n_val_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    validation_error_epoch.append(this_validation_loss)

                    sys.stdout.write("\r|   {0:3d}   |    {1:4d}     |   {2}  | {3:3d}  |".format(epoch,
                                                                                                  minibatch_index,
                                                                                                  round(reduce(lambda x,
                                                                                                                      y: x + y,
                                                                                                               train_costs_epoch) / len(
                                                                                                      train_costs_epoch),
                                                                                                        5),
                                                                                                  100))
                    sys.stdout.flush()

                    print("  {0}   |  {1}  |  {2}   |".format(round(this_validation_loss, 5),
                                                              str(learning_rate),
                                                              str(int((time.clock() - start_time_epoch) ))))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter_count * patience_increase)

                        best_validation_loss = this_validation_loss

                if patience <= iter_count:
                    done_looping = True
                    break

            epoch_avg_cost = reduce(lambda x, y: x + y, train_costs_epoch) / len(train_costs_epoch)
            train_costs.append(epoch_avg_cost)
            if validation_error_epoch:
                epoch_avg_val_error = reduce(lambda x, y: x + y, validation_error_epoch) / len(validation_error_epoch)
            else:
                epoch_avg_val_error = validation_errors[-1]
            validation_errors.append(epoch_avg_val_error)

        if self.para_load:
            queue_t2l.put('done')
        end_time = time.clock()
        print(('Optimization complete with best validation score of %f,') % (best_validation_loss))
        print('The code run for %d epochs, with %f epoch/sec' % (epoch, (end_time - start_time) / (1. * epoch)))
        print('Total training time:{}'.format(end_time - start_time))

        return (train_costs, wvals, validation_errors)

    def checkNaNs(self):

        # floatX = theano.config.floatX  # @UndefinedVariable

        for param_i in self.params:
            if numpy.any(numpy.isnan(param_i.get_value())):
                print("NaN in weights")

        for lpu in self.last_param_update:
            if numpy.any(numpy.isnan(lpu.get_value())):
                print("NaN in last_param_update")

        if self.compileDebugFcts:
            n_train_batches = self.n_train_batches
            # batch_size = self.cfgParams.batch_size
            for i in range(n_train_batches):
                descr = self.compute_train_descr(i)
                if numpy.any(numpy.isnan(descr)):
                    print("NaN in descriptor in batch {}".format(i))

    def set_new_partition(self):
        print("------------------------------------------------------------------------------ New set")
        self.set_data_para(self.training_set_x[self.val_size:],
                           self.training_set_y[self.val_size:],
                           self.training_set_x[:self.val_size],
                           self.training_set_y[:self.val_size])

    def init_worker(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def load_para(self, config, func, num_process=10):
        queue_l2t = config['queue_l2t']
        queue_t2l = config['queue_t2l']
        pool = Pool(num_process, self.init_worker)

        while True and (not self.killer.kill_now):
            if queue_t2l.empty():
                pass
            else:
                msg = queue_t2l.get()
                if msg == 'training':
                    pool.map(func, zip(range(self.training_set_x.shape[0]),
                    #pool.map(func, zip(range(num_process),
                                      itertools.repeat(self.training_set_x),
                                       itertools.repeat(self.training_set_y)))
                    queue_l2t.put('load_done')

                elif msg == 'done':
                    break

        pool.close()
        pool.terminate()
        pool.join()

    def setup_para(self, nb_images, c, h, w, y, type=None, val_size=2000):
        assert type is not None
        print("setup_para")
        self.val_size = val_size
        self.training_set_x = sharedmem.empty((nb_images, c, h, w), dtype='float32')
        if type == 'classifier':
            self.training_set_y = sharedmem.empty((nb_images,), dtype='int32')
        elif type == 'regressor':
            self.training_set_y = sharedmem.empty((nb_images, y), dtype='float32')
        else:
            assert False, 'Type has to be classfier or regressor'


        self.setDataAndCompileFunctions(self.training_set_x[val_size:],
                                        self.training_set_y[val_size:],
                                        self.training_set_x[:val_size],
                                        self.training_set_y[:val_size])
        self.para_load = True
        import warnings
        warnings.filterwarnings("ignore")

    def train_para(self, nb_epoch, function_para):

        self.config = {}
        self.config['queue_l2t'] = Queue(1)
        self.config['queue_t2l'] = Queue(1)
        self.config['queue_t2l'].put('training')

        load_proc = Process(target=self.load_para, args=(self.config,
                                                         function_para))

        load_proc.start()
        self.train(nb_epoch)
        load_proc.join()


