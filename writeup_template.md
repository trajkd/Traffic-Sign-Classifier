# **Traffic Sign Recognition** 

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
[image4]: ./web-traffic-signs/sign33.png "Traffic Sign 1"
[image5]: ./web-traffic-signs/sign27.png "Traffic Sign 2"
[image6]: ./web-traffic-signs/sign1.png "Traffic Sign 3"
[image7]: ./web-traffic-signs/sign17.png "Traffic Sign 4"
[image8]: ./web-traffic-signs/sign12.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data has different counts of each sign.

![Counts][image1]

### Design and Test a Model Architecture

#### 1. Describe how preprocessed the image data was preprocessed. What techniques were chosen and why these techniques were chosen?

I normalized the image data because training performs better when the data has zero mean and equal variance.

#### 2. Describe what the final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU           		|         				    					|
| Max pooling			| 2x2 stride, outputs 5x5x16   					|
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Dropout				| Keep probability of 50%						|
| Fully connected		| outputs 84									|
| RELU					|												|
| Dropout				| Keep probability of 50%						|
| Fully connected		| outputs 43									|

#### 3. Describe how the model was trained.

To train the model, I used an Adam optimizer, a batch size of 128 and a learning rate of 0.001 and trained for 10 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.2%
* validation set accuracy of 94.6%
* test set accuracy of 92.8%

An iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
I used the LeNet architecture as suggested in the lesson.
* What were some problems with the initial architecture?
Because of overfitting the accuracy of the initial architecture was lower than 93% on the validation set.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Two dropout layers were added to the LeNet architecture.
* Which parameters were tuned? How were they adjusted and why?
The batch size, learning rate, epochs and keep probability were initially set to a reasonable value and never adjusted as they were already keeping the accuracy high.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
They are good techniques for reducing overfitting and thus improving accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![33. Turn right ahead][image4] ![27. Pedestrian][image5] ![1. Speed limit (30km/h)][image6] 
![17. No entry][image7] ![12. Priority road][image8]

The first image might be difficult to classify because the sign is a little skewed.
The second image might be difficult to classify because a portion of it is hidden by leaves and the shape is very similar to several other signs.
The third image might be difficult to classify because the number inside the sign is written in a relatively small size and is enclosed in a panel.
The fourth image might be difficult to classify because the red background can mistakenly classify it as a stop sign.
The fifth image might be difficult to classify because a portion of it is hidden by leaves and is a little skewed.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      | Turn right ahead   							| 
| Pedestrian     		| General caution 								|
| Speed limit (30km/h)	| End of no passing								|
| No entry	      		| No entry  					 				|
| Priority Road 		| Priority Road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares slightly favorably to the accuracy on the test set of 92.8%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is extremely sure that this is a Turn right ahead sign (probability of 0.996), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| Turn right ahead								| 
| .00     				| Turn left ahead								|
| .00	      			| Roundabout mandatory			 				|
| .00					| Ahead only									|
| .00				    | Go straight or left  							|

For the second image, the model is extremely sure that this is a General caution sign (probability of 0.97), but the image contains a Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| General caution								| 
| .02	      			| Traffic signals  				 				|
| .00				    | Speed limit (70km/h) 							|
| .00     				| Wild animals crossing							|
| .00					| Road work										|

For the third image, the model is slightly sure that this is a Children crossing sign (probability of 0.42), but the image contains a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42         			| Children crossing								| 
| .33     				| End of all speed and passing limits			|
| .12					| End of no passing  							|
| .04	      			| Beware of ice/snow			 				|
| .03				    | Keep right		  							|

For the fourth image, the model is extremely sure that this is a No entry sign (probability of 1.00), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry										| 
| .00     				| Stop 											|
| .00	      			| Speed limit (70km/h)			 				|
| .00				    | Wild animals crossing 						|
| .00					| Bicycles crossing								|

For the fifth image, the model is extremely sure that this is a Priority Road sign (probability of 0.99), and the image does contain a Priority Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| .99         			| Priority Road 										| 
| .00   				| No entry									  		    |
| .00					| End of all speed and passing limits				    |
| .00	      			| End of no passing									    |
| .00				    | End of no passing by vehicles over 3.5 metric tons 	|