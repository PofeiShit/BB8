# BB8 - A Scalable, Accurate, Robust to Partial Occlusion Method for Predicting the 3D Poses of Challenging Objects without Using Depth

Author: Mahdi Rad <mahdi.rad@icg.tugraz.at, radmahdi@gmail.com>

## Requirements:
  * OS
    * Ubuntu 14.04
    * CUDA 8
  * via Ubuntu package manager:
    * python2.7
    * python-numpy
    * python-pip
    * OpenCV
  * via pip install:
    * theano (0.9)
    * sharedmem

For a description of our method see:

M. Rad, and V. Lepetit. BB8: A Scalable, Accurate, Robust to Partial Occlusion Method for Predicting the 3D Poses of Challenging Objects without Using Depth. In Proc. IEEE Int'l Conf. on Computer Vision, 2017,
or the [project page](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/3d-pose-estimation/).

## Setup:
  * Put dataset files into ./data. We provide the [Cat data](https://files.icg.tugraz.at/f/d7bde012c5/) which contains some images of the Cat of the LINEMOD dataset (Hinterstoisser et al.) in the format our code requires. Unzip the file and put it in: ./data/LINEMOD/objects/ or download the full dataset from [here](https://files.icg.tugraz.at/f/472f5d1108/). For the original dataset visit http://campar.in.tum.de/Main/StefanHinterstoisser. For more info see [dataset info](./data/LINEMOD/dataset_info.md).
  * Goto ./src and see the generate_data.py file parameters to generate data. Set the bg_path to directory of background images (ImageNet). Change load_objs_to_memory function, regarding the objects of the interest and the dataset, which you want to use.


## Train BB8:
For training BB8, we use first 7 pre-trained VGG-16 convolutional weights. You can download the weights [here](https://files.icg.tugraz.at/f/8d1c0e3017/), and put it in the weights directory (./weights).
Train BB8 using following command:
```
python src/train.py --config src/bb8.yaml
```
Note that if there is only one object of interest, we can replace VGG-16 by a simpler architecture BB8-tiny, the computation time then become less, with similar accuracy.


## Train BB8-tiny:
Train BB8-Tiny using following command:
```
python src/train.py --config src/bb8_tiny.yaml
```

## Test BB8:
Test BB8 using following command:
```
python src/test.py
```

## Datasets:
S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige, and N. Navab. Model Based Training, De- tection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes. In ACCV, 2012.

O. Russakovsky, J. Deng, H. Su, J. Krause, S.Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. Berg, and L. Fei-Fei. Imagenet Large Scale Visual Recog- nition Challenge. IJCV, 115(3):211–252, 2015.
