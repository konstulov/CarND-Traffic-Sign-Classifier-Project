# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/random_sample_of_training_images.png "Visualization"
[image2]: ./examples/speed_limit_yuv_channels.png "YUV Channels"
[image3]: ./test_images/german_traffic_sign_yield.png "Traffic Sign 1"
[image4]: ./test_images/german_traffic_sign_speed_limit_50.png "Traffic Sign 2"
[image5]: ./test_images/german_traffic_sign_end_of_speed_limit.png "Traffic Sign 3"
[image6]: ./test_images/german_traffic_sign_no_entry.png "Traffic Sign 4"
[image7]: ./test_images/german_traffic_sign_priority_road.png "Traffic Sign 5"
[image8]: ./test_images/german_traffic_sign_wild_animals_crossing.png "Traffic Sign 6"
[image9]: ./examples/priority_road_sign_conv1_feature_maps.png "Visualizing Activations"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and my project code is in the Python Notebook `Traffic_Sign_Classifier.ipynb` (or the HTML file `Traffic_Sign_Classifier.html`)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used NumPy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the images in the training data set (for this I wrote a helper function `sample_images()` which I used to explore different images randomly sampled from the training, test, or validation dataset)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Following the idea in the suggested paper (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) I converted images to the YUV color space. And for the final model, I trained only on the Y channels.

Here is an example of a traffic sign image decomposed into YUV channels.

![alt text][image2]

As a last step, I normalized the image data to be between -1 and 1.
I didn't do any data augmentation.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y channel (of the YUV image)  		| 
| Convolution 1 5x5   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6   				|
| Convolution 2 5x5   	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16    				|
| Fully connected 1 	| outputs 120  									|
| Fully connected 2 	| outputs 84  									|
| Fully connected 3 	| outputs 43  									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used `tf.nn.softmax_cross_entropy_with_logits()` for the loss and `AdamOptimizer` for optimization.
I trained for 30 epochs using batches of size 128. The final chosen learning rate was set to `0.005`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.942
* test set accuracy of 0.922

Among the different approaches I took, I only changed the input images, initially training on RGB images (i.e. with 3 channels) and then switching to YUV images (training only on the Y channels).
Additionally, the very first model I trained was trained on the unnormalized RGB images.
I also adjusted the learning rate from 0.001 to 0.005.
The best performance was obtained when training on normalized Y channels with learning rate of 0.005.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image        				| Prediction        						| 
|:-------------------------:|:-----------------------------------------:| 
| yield             		| yield   									| 
| speed_limit_50     		| speed_limit_50 							|
| end_of_speed_limit		| end_of_speed_limit 						|
| no_entry          		| no_entry  								|
| priority_road   			| priority_road    							|
| wild_animals_crossing 	| wild_animals_crossing  					|


The model was able to correctly guess all 6 of the 6 traffic signs, which gives an accuracy of 100% on this small data set. This compares favorably to the accuracy on the test set of 92.2%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell 164 of the Ipython notebook.
Here's the images and the corresponding predicted classes with their softmax scores
```
img_idx. image_name: pred_class_1 (softmax_score_1), ..., pred_class_5 (softmax_score_5)
0. german_traffic_sign_yield.png: Yield, (100.0%), Speed limit (20km/h), (0.0%), Speed limit (30km/h), (0.0%), Speed limit (50km/h), (0.0%), Speed limit (60km/h), (0.0%)
1. german_traffic_sign_speed_limit_50.png: Speed limit (50km/h), (100.0%), Speed limit (30km/h), (0.0%), Speed limit (20km/h), (0.0%), Speed limit (60km/h), (0.0%), Speed limit (70km/h), (0.0%)
2. german_traffic_sign_end_of_speed_limit.png: End of all speed and passing limits, (100.0%), Go straight or right, (0.0%), Roundabout mandatory, (0.0%), End of speed limit (80km/h), (0.0%), End of no passing, (0.0%)
3. german_traffic_sign_no_entry.png: No entry, (100.0%), Speed limit (20km/h), (0.0%), Speed limit (30km/h), (0.0%), Speed limit (50km/h), (0.0%), Speed limit (60km/h), (0.0%)
4. german_traffic_sign_priority_road.png: Priority road, (100.0%), Speed limit (20km/h), (0.0%), Speed limit (30km/h), (0.0%), Speed limit (50km/h), (0.0%), Speed limit (60km/h), (0.0%)
5. german_traffic_sign_wild_animals_crossing.png: Wild animals crossing, (100.0%), Speed limit (20km/h), (0.0%), Speed limit (30km/h), (0.0%), Speed limit (50km/h), (0.0%), Speed limit (60km/h), (0.0%)
```

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The image below shows the first convolutional layer activations on the "Priority Road" sign image. Here we can clearly see that each of the 6 feature maps in the layer activates mostly around the sign edges. Specifically, FeatureMap0 detects the bottom edges of the road sign, whereas FeatureMap3 detects the top left diagonal edge of the sign. Several of the feature maps pickup some of the image background (most pronounced in feature maps 1, 3, and 4), whereas others only detect the sign or the edges of the sign and ignore other parts of the image.
![alt text][image9]
