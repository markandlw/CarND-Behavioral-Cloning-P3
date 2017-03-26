#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/center_recover.jpg "Recovery Image"
[image4]: ./examples/center_recover2.jpg "Recovery Image"
[image5]: ./examples/center_recover3.jpg "Recovery Image"
[image6]: ./examples/center_to_flip.jpg "Normal Image"
[image7]: ./examples/center_flip.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I implement Nvidia model for the problem.

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 80). 

####2. Attempts to reduce overfitting in the model

The Nvidia model contains no dropout layers, and it is counterintuitive that the model is good to solving the problem while presenting overfitting.

After several try and error steps, I decide to cancel out the validation process since it does not stand for a good test performance.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I also provide fail-recovering data at several locations where the car is not good at driving. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement Nvidia model as first and tune it like doing a transfer learning.
The deriving process is also documented in git history.

My first step was to implement Nvidia model and introduce ELU for nonlinearity. At the beginning of tuning process, my goal was to decrease MSE as much as possible, so I chose steps that was good for MSE like flipping images, resizing images or using HSV color space. I also used training and validation data for understanding the overfitting trend at this stage. After several try and error steps, I noticed that I could get my car running with not-so-bad test performance, says completing the track with bumping to the edges, while the model is presenting overfitting about 3 orders of magnitude between training and validation set. 

As a result, I gave up pursuing better MSE but focusing on providing image data where I failed the test on the track, and it was getting better. The dropout layer is not being introduced with the same reason. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 79-98) consisted of a convolution neural network with the following layers and layer sizes.

There are 20 keras layers in the model.

* Cropping: 160,200,3 -> 77,200,3
* Normalization: 77,200,3 -> 77,200,3
* Conv2D: 77,200,3 -> 37,98,24
* ELU
* Conv2D: 37,98,24 -> 17,47,36
* ELU
* Conv2D: 17,47,36 -> 7,22,48
* ELU
* Conv2D: 7,22,48 -> 5,20,64
* ELU
* Conv2D: 5,20,64 -> 3,18,64
* Flatten: 3,18,64 -> 3456
* Dense: 3456 -> 100
* ELU
* Dense: 100 -> 50
* ELU
* Dense: 50 -> 10
* Elu
* Dense: 10 -> 1

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive to the center when there is a tendency that the car is going off the road. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I flipped images and angles thinking that this would reducing the left-turn-trend bias in track 1. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 67223 number of data points. I then preprocessed this data by choosing center, left or right images randomly, flipped 50% of the images and resized the images to feed into the Nvidia model.

I used this training data for training the model. The ideal number of epochs was 3-5 as evidenced by testing the performance on track 1. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Reflections

---

Data is king! I may not have the best model but the model achieved the goal by being fed enough data. The fact that my model can achieve the project goal while presenting overfitting could be that it learned enough data in track 1, and it shall not generalize well in other scenarios like track 2. A test on track 2 shows that it did not do well. Introducing dropout layer and fine-grained data augmentation should help the model to generalize.