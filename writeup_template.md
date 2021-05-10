# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[label_sampels]: ./result_images/sample_counts_classes.png "Labels"
[image_per_label]: ./result_images/input_images_dataset.png "Images"
[transformed]: ./result_images/image_transformations_grey.png "Transformation"
[oversampling_count]: ./result_images/oversample_sample_counts_classes.png "Oversampling"

## Rubric Points

The code of this project is linked [here](https://github.com/mennchen22/udacity_traffic_sign_classifier)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 172000
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

There are 43 different image classes to be classified. The example below shows one random image ber label from the image set

![alt text][image_per_label]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][label_sampels]

### Design and Test a Model Architecture

#### 1. Image preprocessing pipeline

To use the images for a neural network sequential processing steps will be done over each image of the dataset
Parallel we want to fill the image sample count gap for individual lable groups. Shown in the image above there are plenty of label groups
with less than 500 images. Therefore we create additional images by adding image transformation to each group until a minimal 
number of samples is reached. Afterwards we process each image with global transformations:

    for each label
        while the label count less than 4000
            create at least 200 images with random rotation and transformation
            add the images to the label group
        resize images to 32x32 width and height
        convert images to greyscale

For the image creation a random angle between 0 and 10 and a translation [x: 0 - 3, y: 0 - 6] is used to create new images

The sample count is equalized to 4000 images afterwards: 
![alt text][oversampling_count]
![alt text][transformed]

Benefits of these steps:

New images don't mean, that the net will train better if the data is the same. Therefore, we add changes to the image (like rotation and translation) to have better generalization results in the training process. 
This includes that classes with fewer samples can be classified better.

#### 2. Model choices

Different model types have been tested. The first approach was a LeeNet with two convolution layer and two fully connected ones. 
For testing a net with parallel convolutions through the net 

    Input 5x5 convolution --> max pooling --> 5x5 convolution --> max pooling --> |
    Input 3x3 convolution --> max pooling --> 3x3 convolution --> max pooling --> | => fc1 --> fc2 --> logits(43 classes)
    Input 1x1 convolution --> max pooling --> 1x1 convolution --> max pooling --> |

and another model with blocks equal to the first one have been approached

    Input 5x5 convolution --> max pooling --> |                     out_layer_23x3 --> 5x5 convolution --> max pooling --> |
    Input 3x3 convolution --> max pooling --> | ==> out_layer_2 --> out_layer_23x3 --> 3x3 convolution --> max pooling --> | => fc1 --> fc2 --> logits(43 classes)
    Input 1x1 convolution --> max pooling --> |                     out_layer_23x3 --> 1x1 convolution --> max pooling --> |

Additional generalization methods like dropout have been included to the architecture in the fc layers. In the end a parameter tuning with different setups has lead to the final architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x16				    |
| Flattening		    | output 400        							|
| Fully connected		| output 120        							|
| Dropout       		| keep 50%          							|
| RELU					|												|
| Fully connected		| output 84       								|
| Dropout       		| keep 50%          							|
| RELU					|												|
| Softmax				|           									|
 
#### 3. Training approach

There is a set of hyperparameter to change:

1) Epochs --> [10,100,200]
2) Learning Rate --> [0.001, 0.002, 0.01]
3) Batch size --> [128, 64, 32]
4) Dropout keep --> [0.5, 0.7, 1.]

Therefore changes will be made iteratively and the results are compared to find the best setup.
The best results came from this setup:

1) Epochs --> [200]
2) Learning Rate --> [0.001]
3) Batch size --> [32]
4) Dropout keep --> [0.5]

Below the accuracy of different training runs are shown:

[parallel_net]: ./result_images/train_loss_model_Parallelnet_bs64_ep100_lt0.001_dk0.5.png "Labels"
[lee_net_1]: ./result_images/train_loss_bs128_ep100_lt0.001_dk0.5.png "Labels"
[lee_net_2]: ./result_images/train_loss_bs64_ep100_lt0.002_dk0.5.png "Labels"
[lee_net_3]: ./result_images/train_loss_model_Leenet_bs32_ep100_lt0.001_dk0.5.png "Labels"
[lee_net_4]: ./result_images/train_loss_model_Leenet_bs32_ep200_lt0.001_dk0.5.png "Labels"
[new_images_german]: ./result_images/gtsrb_image_transformations.png "German Traffic Sign Dataset"
[five_images]: ./result_images/gtsrb_five_images.png "Five selected images"

Parallel net

![alt text][parallel_net]

LeNet

![alt text][lee_net_1]
![alt text][lee_net_2]
![alt text][lee_net_3]
![alt text][lee_net_4]

From the hyperparameter tuning this information could be gathered:

Better results comes with lower batch size
A lowered learning rate reduce the variance of the loss over time 
An epoch count of 200 fits the training model because the learning will increase even after 100eEpochs and stabelize at > 150 epochs 

### Final results 

Best Validation accuracy: 93.24 % in epoch 158 
Test Accuracy: 89,99%

### Test a Model on New Images 


#### 1. German Traffic Sign Dataset 

The german traffic sign dataset to test the model on new data. The data is processed the same way ass the training images:

![alt text][new_images_german]

From this five random images were taken and classified

![alt text][five_images]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road      	| Priority Road  								| 
| Speed Limit 100kmh    | Speed Limit 100kmh  							|
| Ahead only			| Ahead only									|
| Yield	      		    | Yield					 				        |
| Roundabout mandatory	| Roundabout mandatory      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Prediction results

The code for making predictions on my final model is located at the bottom of the Ipython notebook.
Next following the five images prediction results for the top five classes:

#### Image 1

For the first image, the model is relatively sure that this is a Priority Road sign (probability of 0.6), and the image does contain a stop sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999999999         			| Priority Road 									| 
| .0000000337     				| Stop										|
| .0000000022					| Yield											|
| .0000000005	      			| Right-of-way at the next intersection				 				|
| .0000000000				    | Road work   							|

Here it seems like a sharp geometry filter divides the signs in this set.

#### Image 2

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999999999         			| Speed Limit (100km/h)  											| 
| .0000000000     				| Speed limit (80km/h)										|
| .0000000000					| Priority road											|
| .0000000000	      			| Speed limit (30km/h) |
| .0000000000				    | Speed limit (120km/h)  							|

Althoght the correct sign is classified by nearly 100% (cut at 10 digits) the net detects speed signs expect the priority road sign with the same color set (in grayscale).

#### Image 3

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999999999         			| Ahead only  											| 
| .0000000000     				| Go straight or right										|
| .0000000000					| No passing											|
| .0000000000	      			| Priority road |
| .0000000000				    | Keep left  							|

The sign is classified with almost 100% certainty. Other detected shapes contains arrows like the sign itself.

#### Image 4

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999999999         			| Yield 											| 
| .0000000000     				| Keep right									|
| .0000000000					|  Go straight or right													|
| .0000000000	      			|Stop |
| .0000000000				    | Ahead only 							|

Expect to the other images the other detected classes dont fit a clear subset of features in the image. The yield sign has no letters or arrows, but the classic shape flipped over.


#### Image 5

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9993985891         			|  Roundabout mandatory  											| 
| .0005886111     				| Priority road									|
| .0000096762					|  Stop										|
| .0000027947	      			|Vehicles over 3.5 metric tons prohibited |
| .0000003302				    | Speed limit (100km/h)						|

At least the fith image has less certainty. Other round shaped objects are likely to fit the label, but with a result less than 1%

# Discussion

This approach uses a basic LeNet with dropout. This could be extended with more layers to improve the classification result. Additionally, a noise algorithm was used to randomize the images even more, but with worse results. This method could be tested with more parameter tuning to generalize the data classifiication even more.