# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/example_color2.png "Traffic sign image -- color"
[image2]: ./examples/example_gray.png "Traffic sign image -- gray"
[image3]: ./examples/histogram.png "Distribution of data set"
[image4]: ./test_new/1.jpg "Traffic Sign 1"
[image5]: ./test_new/2.jpg "Traffic Sign 2"
[image6]: ./test_new/3.jpg "Traffic Sign 3"
[image7]: ./test_new/4.jpg "Traffic Sign 4"
[image8]: ./test_new/5.jpg "Traffic Sign 5"
[image9]: ./examples/histogram_aug.png "Distribution of augmented training set"
[image10]: ./examples/rotation20.png "rotation image"
[image11]: ./examples/exampe_pedestrain.png "predestrain"
[image12]: ./examples/exampe_th.png "caution"
[image13]: ./examples/prob_top5.png "barCharts"

---

### Data Set Summary & Exploration

#### 1. A basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset

Here is an exploratory visualization of the data set. 
It is a bar chart showing the distribution of classes in the training, validation and test set.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Pre-processing image data

As a first step, I decided to convert the images to grayscale because experiments show that a greater accuracy is achieved by using grayscale images instead of color in [Sermanet‎2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).


Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1] before grayscaling

![alt text][image2] after grayscaling

Then, I normalized the image data because data with optimizer converges faster on normalized data (zero mean and equal variance).

I decided to generate additional data because 1) it can be seen that the number of images in each class varies a lot: some classes have very limited images; 2) augmenting the training set will help improve model performance.

To add more data to the the data set, I used the following techniques rotate images by using [scipy.ndimage.interpolation.rotate](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.interpolation.rotate.html). A list of rotation angles is given and the algorithm randomly picks up rotation angles from this list. After the training data augmentation, each class has at least 1000 images for training.

Here is a rotation image which is rotated for 20 deg:

![alt text][image10]

Here is the distribution of classes in the augmented training set:

![alt text][image9]



#### 2. ConvNet model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| tanh					| We can use RELU instead						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| tanh					| We can use RELU instead						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 = 400				|
| Fully connected		| outputs 200 									|
| RELU					| 	        									|
| Dropout				|												|
| Fully connected		| outputs 120									|
| RELU					| 	        									|
| Dropout				|												|
| Fully connected		| outputs 43									|
 


#### 3. Training


To train the model, softmax cross entropy between logits from the ConvNet and labels is computed, an Adam optimizer is constructed to minimize the mean of cross entropy. The batch size is set to be 160, the number of epochs is 10, and the learning rate is 0.004.



#### 4. Approaches and Results



My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.956 (greater than 0.93, less than 0.97)
* test set accuracy of 0.932
* I used the [LeNet-5](http://yann.lecun.com/exdb/lenet/) with max pooling but without dropout at first. I have tried different number of feature maps in each layer and tuned the batch number, epoch number, learning rate, etc. But the problem is the validation accuracy is no greater than 0.9 (around 0.88). I don't think there will be an obvious improvement by tuning the parameters in the LeNet-5 architecture, so I augmented the training set. It turns out that after augmenting the training set, the validation accuracy is always greater than 0.93 (Data is the fuel of ML). 
* Later on, I tested the architecture using different activation function, different number of feature maps in each layer, with and without dropout, with average/max pooling. I find that the performance of improvement by adjusting the architecture is not as obvious as the augment of data set. Note that the architecture is still LeNet rather than the multi-stage ConvNet. I'll test the architecture of multi-stage ConvNet later.
* By adjusting the architecture, I find that using sigmoid as the activation function after the convolutional layer is not a fit: the validation accuracy is low. Also, using dropout makes the architecture stable and robust.

Great questions in the original template to ask oneself when adjusting an architecture. I copy them here:
* If an iterative approach was chosen: What was the first architecture that was tried and why was it chosen? What were some problems with the initial architecture? How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. Which parameters were tuned? How were they adjusted and why? What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* If a well known architecture was chosen: What architecture was chosen? Why did you believe it would be relevant to the traffic sign application? How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Pedestrians			| Pedestrians									|
| General caution  		| General caution				 				|
| Stop					| Stop				 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

Note that the images downloaded from the internet are clipped to be square, reshaped to 32x32x3, and pre-processed (grayscaling and normalization).


The traffic signs of *Pedestrians* (the third image) and *General caution* (the fourth image) can be confused because the resolution of images are 32x32, which is low and cannot show the details and difference if two signs look similar. These are examples of the signs of pedestrians and general caution in the given training set respectively. I think this problem can be mitigated if the data size is larger and if the resolution of images is greater. Given the article [Sermanet‎2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), this problem can be properly solved when we use multi-stage ConvNet or other deep learning architectures.

![alt text][image11]

![alt text][image12]

However, without augmenting the training data, the *Speed limit (30km/h)* will be predicted as *Speed limit (20km/h)* and the *Pedestrians* will be predicted as *General caution* more easily.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (30km/h)							| 
| 1.0     				| Bumpy road 									|
| .99					| General caution								|
| .79	      			| Pedestrians					 				|
| 1.0				    | Stop      							|

![alt text][image13]

The top one softmax probabilities are approximately 1 except the prediction of *Pedestrians*. Using the same architecture and same training data set, the ConvNet sometimes predicted *Pedestrians* as *General caution*, which has been discussed in the above subsection.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


