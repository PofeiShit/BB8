"""Provides train function for using in CNNs.

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
import sys
from net.poseregnet import PoseRegNet, PoseRegNetParams
from trainer.poseregnettrainer import PoseRegNetTrainingParams, PoseRegNetTrainer
import theano
import numpy as np
import cPickle
import yaml
from os.path import join

class Network:
    def __init__(self):
        self.type = 1               # 1 = Regressor
        self.network_model = '0'    # 0 = Tiny BB8, 1 = BB8 - VGG arch.

        self.batch_size = 128
        self.optimizer = 'MOMENTUM'     
        self.learning_rate = 0.001
        self.steps = []
        self.scales = []
        self.nb_epoch = 300

        self.network = None
        self.trainer = None
        self.network_name = None
        self.save_path = './'

        self.validation_size = 5000
        self.train_set_para = None

        self.config = None

    def setup_from_config(self):
        if self.config is not None:
            with open(self.config, 'r') as f:
                config = yaml.load(f)
                for key in config.keys():
                    value = config[key]
                    print('set {0} to {1}'.format(key, value))
                    setattr(self, key, value)

    def update(self):
        self.setup_from_config()
        self.print_type()

        sys.path.insert(0, self.train_set_para[:self.train_set_para.rindex('/')])
        self.create_training = __import__(self.train_set_para[self.train_set_para.rindex('/') + 1:],
                                              fromlist=['init',
                                                        'pre_create_data',
                                                        'create_data',
                                                        'get_dim'])
        self.create_training.init()

        self.regressor()


    def regressor(self):

        regressorNetType = regressorNetType=int(self.network_model)
        batch_size = int(self.batch_size)
        learning_rate = float(self.learning_rate)
        optimizer = self.optimizer

        assert len(self.steps) == len(self.scales)
        n_chan, h_in, w_in, output_dim = self.create_training.get_dim()
        assert n_chan is not None
        assert h_in is not None
        assert w_in is not None
        assert output_dim is not None
        nb_training = int(self.nb_training)

        rng = np.random.RandomState(23455)
        #theano.config.compute_test_value = 'warn'
        theano.config.exception_verbosity = 'high'

        regressorNetParams = PoseRegNetParams(type=regressorNetType,
                                              n_chan=n_chan,
                                              w_in=w_in,
                                              h_in=h_in,
                                              batchSize=batch_size,
                                              output_dim=output_dim)

        self.network = PoseRegNet(rng, cfgParams=regressorNetParams)
        if regressorNetType == 1:
            self.network.load_vgg()
        print(self.network)

        regressorNetTrainingParams = PoseRegNetTrainingParams()
        regressorNetTrainingParams.batch_size = batch_size
        regressorNetTrainingParams.learning_rate = learning_rate
        if self.scales != []:
            scales = [float(s) for s in self.scales.split()]
            regressorNetTrainingParams.learning_rate_scales = scales
        if self.steps != []:
            steps = [int(s) for s in self.steps.split()]
            regressorNetTrainingParams.learning_rate_steps = steps

        regressorNetTrainingParams.optimizer = optimizer

        self.trainer = PoseRegNetTrainer(self.network, regressorNetTrainingParams, rng)

        self.trainer.setup_para(nb_training, n_chan, h_in, w_in, output_dim, type='regressor')
        self.create_training.pre_create_data()


    def train(self):
        self.trainer.train(n_epochs=int(self.nb_epoch), storeFilters=True)


    def train_para(self):
        self.trainer.train_para(int(self.nb_epoch),
                                self.create_training.create_data)


    def save(self):
        if self.network_name is None:
            self.network_name += "_model"
            self.network_name += str(self.network_model)
            self.network_name += "_epoch"
            self.network_name += str(self.nb_epoch)

        self.network.save(join(self.save_path, self.network_name + ".weight"))
        f = file(join(self.save_path, self.network_name + ".cfg"), 'wb')
        cPickle.dump(self.network.cfgParams, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    def print_type(self):
        if int(self.type) == 1:
            print('*****************************************************')
            print('*                    REGRESSOR                      *')
            print('*****************************************************')
        else:
            assert False, 'It is not implemented'



def help():
    print('train_set_path')
    print('network_model=0')
    print('batch_size=128')
    print('learning_rate = 0.001')
    print('optimizer = MOMENTUM')
    print('nb_epoch = 300')
    print('network_name=Date')
    print('para = (True = 1), (false = 0)')
    import sys
    sys.exit(0)

if __name__ == '__main__':

    network = Network()

    if len(sys.argv) == 1:
        print('Using default configuration... ')
    else:
        if len(sys.argv) == 2:
            help()
        else:
            for i in range(1, len(sys.argv), 2):
                key = sys.argv[i]
                '''
                if "--" not in key:
                    assert False, "KEY MUST START WITH --"
                '''
                key = key.replace('--', '')
                value = sys.argv[i+1]
                print('set {0} to {1}'.format(key, value))
                if not hasattr(network, key):
                    print('wrong attribute', key)
                    help()
                    import sys
                    sys.exit(0)
                setattr(network, key, value)

    network.update()
    network.train_para()
    
    network.save()

