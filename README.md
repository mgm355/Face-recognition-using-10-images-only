# Face-recognition-using-10-images-only


## VGG Architecture and Algorithm:
VGG, Visual Geometry Group Convolution network, is the next step after the revolutionary AlexNet. VGG consists of much fewer parameters than Alexnet while still having more accuracy in the ImageNet competition.The focus of VGG was mainly on the depth and its effect on the accuracy on image recognition tasks. It uses very small convolution filters (3x3) and deep layers set to 16-19  [1]. In this project I used the 16 layers or what we refer as VGG16. VGG also have 5 Maxpooling layers of dimensions (2,2) , fixed zero padding (1,1), and relu activations among its layers. It also has dropout set to 0.5 between its layers to perform regularization.
- Using VGG 16:
I’ll first make use of previously trained VGG 16 so that I ensure it have already learned how to extract basic representations of the pictures through its deep layers before start learning my photos. Then I’ll remove the last fully connected layers that are used specifically for certain task and add my own classification fully connected layers. Then In the training, I only use the pretrained layers for embedding extraction from my images before feeding them to the fully connected layer. Then, only the fully connected layer will perform the backpropagation for gradients updating.
Also, I’ll use a tool (face_recognition library) to crop the people faces so that I guarantee I’ll learn exactly what I want in this few dataset available.
## Loss Function:
I used as a loss function the keras built in SparseCategoricalCrossentropy which computes the categorical cross entropy loss. Where the loss function is defined as:

 ![J(w)=−1/N ∑i=1,N[yi log(y^i)+(1−yi)log(1−y^i)]](/images/eq.jpg)

•	w refer to the model parameters, e.g. weights of the neural network

•	yi is the true label, (integer not cross entropy)

•	yi is the predicted label

## Training, Testing data
The Cropped_Images is identical cope of Images dircetory for Train and Test subfolder, but after running the source code which will contain only the faces cropped of the images.
In the Images Directory we'll have the sets as follows:
Images
	Train
		donald_trump
			donald_1.jpg
			donald_2.jpg...

		marcel_ghanem
			marcel_1.jpg
			marcel_2.jpg...
	Test
		donald_trump
			donald_1.jpg
			donald_2.jpg
		marcel_ghanem
			marcel_1.jpg
			marcel_2.jpg

Each of Train and Test folders should have 11 subfolder
For the Train; each subfolder should be named as <firstname>_<lastname> containing 10 picture. Each picture inside this subfolder should be named as <firstname>_<index>.jpg where index between 1 and 10.
For the Test each subfolder should be named as <firstname>_<lastname> containing 2 pictures. Each picture inside this subfolder should be named as <firstname>_<index>.jpg where index is just 1 or 2.
In Addition, we'll have folder named Mixed_Pics that contains multiple persons in same picture just for demonstration at the end.

Note: you can convert all your images to jpg using single bash command from imagemagick for simplicity:
>> mogrify -format jpg \*.*

## Data Splits:
Based on Andrew Ng tutorials, the splitting is highly dependent on the size of data, so in this task, I used a validation/ train split equals to 0.2 where I only have 110 training images which is relatively small, so validating over 20% of the data to ensure good standing in the learning process. Will be good for this amount of data.

## Learning Graph:
 ![Learning graph](/images/graph.png)
- Convergance, bias and variance:
The model converges starting to reach good convergance after the second 5 epochs. Since the training curve is still good for our variance and not it’s not high and relatively low, and since the training error is reaching very low losses at the end of training, that indicates the bias is not high as well and is relatively low. Therefore we can say we are on the safe side according to the table shown in the lectures where dev same as train in addition we have low training error  Low Bias, Low Variance.


## Confusion metrix
- on training data:
 
![Learning graph](/images/matrix1.png)
 
- on test data:

![Learning graph](/images/matrix2.png)

There are no errors according to the confusio matrixs, we are at the best case where its generelizing very good and assigning each person to his class correctly. The results was as expected on training and testing since we are very confident of the VGG16 which is extracting very well the representations we want for our classifier, were we’re getting 2622 feature (embedding for each person) that is very well pretrained and wanted to predict and is learning from it to predict only 11 classes. Moreover the usage of face croppings with some faces with smile, some being sad and some wearing eyeglasses, force the learning structure to somehow avoid overfitting to specific person pose and make it more robust.
Some additional improvements that could be done is to increase the vary the poses of each person, their hair, front and side views, in addition to their facial expressions which might also further make our model detect the person even with more different poses or views.

## Reference:
[1] Simonyan, Karen & Zisserman, Andrew. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv 1409.1556.
