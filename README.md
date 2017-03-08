
## **End-to-end Driving Behavioral Cloning Project**

The goals / steps of this project are the following:

* Dataset creation and data acquisition strategy
* Learning model architecture
* Training and validation

[//]: # (Image References)

[image1]: ./output_images/nvidia_architecture.png "Architecture"
[image2]: ./output_images/error_loss.png "Loss"
[image3]: ./output_images/simulator.jpg "Simulator"


### [Rubric](https://review.udacity.com/#!/rubrics/432/view) Points

---
### Introduction
In this project; it is aimed to design self-driving simulator agent who estimates required steering command to keep the vehicle in lane. By using [udacity simulator](https://github.com/udacity/self-driving-car-sim), user driving steering angles and camera images are recorded and model has been trained to reflect user's driving characteristic into the track

![Simulator Image][image3]

---
#### Dataset and Data Acquisition Strategy

It is very important to have consistent, wide-variety and sufficient data to have successful result. By doing so followed steps will be given in this section.

As an initial step, images and steering angles are gathered from simulator by using following order;

* driving on lane, following center line
* veer of the vehicle by purpose and recover to center of lane
* driving on edges of lane, following left and right lines

Here center lane data are classified into straight drivings, smooth left turns, smooth right turns, sharp left turns and sharp right turns. Because in later all these such a different steering characteristics should be sampled as same amount as others to have a good result.

And also veer maneuvers are filtered to have positive steering values left veerings and negative steering values for right veerings. 

Driving on egdes of the lane is helps to recover the vehicle when it closes to off-track. Virtually added shifts (0.2) for their steering angles helps to increase number of image for recoveries when vehicle felt into trouble.

It should also be mentioned; not only center camera images but also side images has been used to increase total number of images. By adding small amount (0.05) of shifts to their steering angles, agent also learns to keep the vehicle in center of lane. 

After all of the efforts mentioned above here; we have approximately 9000 images here is to be used for training section. There is about 2000 images have been recorded as well for to be validate training results.

---
#### Model Architecture

As a model architecture end-to-end nVIDIA architecture has been used. [nVIDIA paper](https://arxiv.org/abs/1604.07316) has been reviewed and five Convolution layer for feature extraction and five Dense layer for regression has been implemented as can be seen in below image.

![Architecture Image][image1]

Before to pass images into learning layers, images has been cropped to (200, 66) as same as nVIDIA did and they normalized to have zero mean as well as 1 standart variation.

In this learning model, dropout technique has been used to overcome overfitting. Dropout has been choosen as 0.5 and only applied between dense layers.

    Convolution2D --> (24, 5, 5)
    Convolution2D --> (36, 5, 5)
    Convolution2D --> (48, 5, 5)
    Convolution2D --> (64, 3, 3)
    Convolution2D --> (64, 3, 3)
    
    Dense --> (1164)
    Dropout(0.5)
    Dense --> (100)
    Dropout(0.5)
    Dense --> (50)
    Dropout(0.5)
    Dense --> (10)
    Dropout(0.5)
    Dense --> (1)

It is also should be stated; Adam optimizer is used, loss metric is choosen as mean squared error. Nonlinearity for activation function is tanh function. 

---
#### Training and Validation

During the training session; 20 epoches is used. 128 data is used due performance capacity of used laptop. Specified batch sizes are guarantee from generator. In generator read csv file rows processed and images and steerings are read. Random sampling from dataset and image balancing from different cameras has been implemented in here. Small amount of shifts for steering angle for different cameras is applied within generator as well.

As a result in below loss plot has been tracked whether the model is underfitting, overfitting or doing good. After many experimentation below image shows saved and demonstrated version of training result.

![Loss Image][image2]

---
#### Result of test video

Here's a [link to my video result](./run1.mp4)

---
### Discussion

Here in this project driving behavioral cloning has been implemented. Different data acquisition strategies, generators and many experimentation to overcome overfitting is performed.

For further imporevements, developed architectured could be experimented for real-life data, current result could be tried on different simulator tracks and image augmentation tecniques could be applied more while reducing number of images in the dataset.
