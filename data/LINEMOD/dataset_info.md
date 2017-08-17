## Dataset parameters

* The LINEMOD dataset: Download full dataset from https://files.icg.tugraz.at/f/472f5d1108/
  or from [1]
* Objects: 15
* Object models: Mesh models with surface color and normals.


## Training images and testing images

The training images were obtained by using 15% of the images and we use the rest for
testing. The training images are selected such that relative orientation between 
them should be larger than a threshold. You can also try random selection, but it 
might affect a slight drop in performance.

You can find the list of training images we used in [2] in the following path:
./training_range/<object_name>


## References

[1] Hinterstoisser et al. "Model based training, detection and pose estimation
    of texture-less 3d objects in heavily cluttered scenes" ACCV 2012,
    web: http://campar.in.tum.de/Main/StefanHinterstoisser

[2] Rad et al. "BB8: A Scalable, Accurate, Robust to Partial Occlusion Method for
    Predicting the 3D Poses of Challenging Objects without Using Depth" ICCV 2017,
    web: https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/3d-pose-estimation/

