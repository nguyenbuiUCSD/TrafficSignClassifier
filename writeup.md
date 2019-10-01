
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
[image2]: ./examples/loss.png "Loss"
[image3]: ./examples/valid_acc.png "Validation accuracy"
[image4]: ./examples/confushion_matrix.png "Confushion matrix"
[image5]: ./examples/false_predictions.png "False predictions"
[image6]: ./examples/web_examples.png "German Traffic Signs"
[image7]: ./examples/web_examples_prediction.png "German Traffic Signs Prediction"
[image8]: ./testimages/test1.JPG "Visual of feature maps"
[image9]: ./examples/feature_maps.png "Visual of feature maps"

---
## Data Set Summary & Exploration

### 1. Data Set Summary.

Summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 2. Visualization:

Here is an exploratory visualization of the data set. It is a bar chart showing the number of train and validation images in each class. Note that, test data still remain undiscovered. 

![alt text][image1]

---
## Design and Test a Model Architecture

### 1. Preprocessed the image data:
In this project, only normalization is used to preprocess data. Normalization help saving trainning time and help model perform better in case features in images have diferrent mean and std

### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         				|     Description	  	      							| 
|:-----------------------------:|:-----------------------------------------------------:|
| Input         				| 32x32x3 RGB image   									|
| Convolution 5x5x3     		| 1x1 stride, output depth: 6, outputs 28x28x6	|
| RELU							|														|
| Max pooling	     		 	| 2x2 stride, outputs 14x14x6	 						|
| Convolution 5x5x6  		   	| 1x1 stride, output depth: 16, outputs 10x10x16	|
| RELU							|														|
| Max pooling	      			| 2x2 stride, outputs 5x5x16	 						|
| Flatten	    				| Outputs 400      								|
| Fully connected				| Outputs 120        								|
| Fully connected				| Outputs 84        								|
| Fully connected				| Outputs 43        								|



### 3. Trained model:

Batch size and epochs:
* EPOCHS = 500
* BATCH_SIZE = 128

Hyperparameters:
* mu = 0
* sigma = 0.1

Trainning:
* rate = 0.001
* optimizer = tf.train.AdamOptimizer()

Trainning process:
loss:

![alt text][image2]

Validation accuracy:

![alt text][image3]

### 4. Choosing Final Model:

Final model results:

* Validation Accuracy = 0.961
* Test Accuracy = 0.942

To get the final result, hyper parametters are twisted with defferent values. In addition, the initial weights and trainning image order will also effect the result since loss function are non convex and there is no guarantee to reach global minimum. So, with the same parametters, restarting the trainning process (reinitialize weights) also will yeild different results.

Confushion matrix:

![alt text][image4]

Based on confushion matrix. We can see some of highest false prediction rate:

![alt text][image5]

---

## Test a Model on New Images

### 1.  German traffic signs found on the web:

![alt text][image6]


### 2. Model's predictions on these new traffic signs:
Results of the prediction:

![alt text][image7]


The model accuracy on new images:
New images test accuracy = 100.00%

### 3. Softmax probabilities:

For most of test images, model are very sure about the probability of image being corrected label ( probability ~= 1.).

Here is example of the 5 highest softmax probabilities of the first images: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Right-of-way at the next intersection   		| 
| 0.     				| Speed limit (20km/h) 							|
| 0.					| Speed limit (30km/h)							|
| 0.	      			| Speed limit (50km/h)					 		|
| 0.				    | Speed limit (60km/h)      					|


## Visualizing the Neural Network
### 1. Visual output of trained network's feature maps:
Here is the visual output of trained network's feature maps on first convolutional layer. We can see the all the edges and color of image are activated:
Input:
![alt text][image8]
Feature maps:
![alt text][image9]


