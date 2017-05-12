#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/color.png "Color"
[image9]: ./examples/gray.png "Gray"
[image3]: ./examples/augmented.png "Random Noise"
[image4]: ./webimages/14 "Traffic Sign 1"
[image5]: ./webimages/24 "Traffic Sign 2"
[image6]: ./webimages/35 "Traffic Sign 3"
[image7]: ./webimages/40 "Traffic Sign 4"
[image8]: ./webimages/7 "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/michaelbrown2/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4410
* The size of test set is 12,630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples of signs are in each class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image9]

I also normalized the data to give it equal variance.

I then augmented the data to a total of 2200 images per class, and applied a rotational effect to random images in the classes to supplement. I plan to also implement a sheering and translation effect that will be applied randomly. 

![alt text][image3]

The end data set had 43 classes with 2200 images each for a total training set of 94,600 images


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   
| Convolution 3x3   | 1x1 stride, same padding, outputs 28x28x6 
| RELU					|						
| Max pooling	      	| 2x2 stride,  outputs 14x14x6
| Convolution 3x3	| 1x1 stride, same padding, outputs 10x10x16
| RELU					|
| Max Pooling			|2x2 stride, outputs 5x5x16
| Flatten				| Input 5x5x16 Output 400
| Layer 3 Fully Connected	| input 400 Output 200
| RELU					| 
| DROPOUT				|
| Layer 4 Fully Connected | Input 200, Output 100
| RELU					|
| DROPOUT				|
| Layer 5 Fully Connected | Input 100, Output 43
| OUTPUT

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

EPOCHS = 10
BATCH_SIZE = 128
rate = 0.005
mu = 0
sigma = 0.1

I also included a drop out of .6 in the training loop and 1.0 in the validation.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.963
* test set accuracy of 0.931

I went through MANY iterations of the process to narrow down the best settings I could identify for the neural network.  The neural network portion made sense to me, however my lack of experience with Python made the process quite long.  I attempted implementing avg_pooling instead of max_pooling and it did not make any noticeable change to my accuracy.  The biggest thing that helped with my accuracy was implementing Dropout in the network.  My original pipeline was overfitting. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The image that the network had the hardest time identifying was the Speed Limit sign and my thought on it is because so many of the signs are round, as well as there are very strongly defined lines in the background that the network may have interpreted as part of the sign. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign
| Road narrows on the right     			| Road narrows on the right
| Ahead Only					| Ahead Only
| Roundabout	      		| Roundabout	
| 100 KM / h		|  Yield 

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Image | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
| 14| [99.72 11.14 -0.07 -3.32 -7.38] | [14 17  2 38  4]   									| 
| 24 | [107.10 48.12 38.51 32.44 23.59]| [24 26 29 21 28]										 |
| 35 | [125.04 33.51 11.22 8.01 3.55] | [35  3  9 36 28]											|
| 40 | [90.77 43.73 18.24 -5.16 -9.13] | [40 12  7 11  1]				 				|
| 7 | [14.19 12.97 9.29 8.62 2.43] | [13 10 33  9 14]     							|


Only image 7 was INCORRECTLY classified.
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


