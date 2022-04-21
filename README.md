# Deep-Learning-
Deep learning is process of training a deep neural network to perform complex operations like pattern recognition , image processing , time series predictions and much more.The major libraries that have been used in this repository are OpenCV and Tensorflow. 

Here we develop models by using Keras, Tensorflow and OpenCv for most of the projects since they have a very well defined documentation and easy to understand code. 
        
### Contents :

* [Face mask detection system ](https://github.com/umang299/Deep-Learning-/blob/main/face_mask_detection_mobilenetv2.py)
![mask op.JPG](https://github.com/umang299/Deep-Learning-/blob/main/mask%20op.JPG)

* [Natural scenes classification for UAV's](https://github.com/umang299/Deep-Learning-/blob/main/Natural%20Scenes_classifier_for_UAV's.py)
![Clasification otput](https://github.com/umang299/Deep-Learning-/blob/main/Screenshot%202022-04-21%20161111.png)

#### Description : 
* The problem statement is to predict the natural scenes around a UAV to tune the UAS to identify and operate accordingly
* Applied various augmentations (rotation, zoom, shear, brightness and night vision) on the images in two approaches, one applies it to a random set of images             second applies to all images. 
* Used 3 Deep Convolution Neural Networks for classification. The 1st and 2nd models are custom models the final is a transfer learning model VGG16.
* Evaluated the performance of each of the three models and choose the best one for final testing. Arrived at an accuracy of 0.82. The loss can be attributed to           bad data and less model training infrastructure. 


