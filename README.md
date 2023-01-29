# 5th_team_project with CNN
## Detection of Covid-19 & Pheumonia by analyzing Chest X-Ray images (ResNet50)

### Problem statement
- In this case study, we will assume that we work as s Deep Learning Consultant.
We have been hired by a hospital in downtown Toronto and we have been tasked to automate the process of detecting and classifying chest disease and reduce the cost and time of detection.
The team has collected extensive X-Ray data and they approached us to develop a model that could detect and classify the diseases in less than 1 minute.
- We have been provided with 532 images that belong to 4 classes:
1. Healthy
2. Covid-19
3. Bacterial Pneumonia
4. Viral Pneumonia
- In this case study we will implement transfer learning technique.

### Transfer Learning with ResNet50
Since the data set we are using is small, one way to address this problem of missing data in a given area is to use data from a similar area, a technique known as transfer learning (TL) (Kermany et al., 2018). In short, TL means using what is learned from one task and applying that to another task without learning from scratch.

We will use ResNet in our model for this project.

ResNet, due to its architecture, does not allow vanishing gradient to occur: the skip connections do not allow it as they act as gradient super-highways, allowing it to flow without being altered by a large magnitude.

source: https://medium.com/towards-data-science/the-annotated-resnet-50-a6c536034758

1. We are going to initialize the ResNet50 model and use weight = 'imagenet'. ImageNet is a large visual database designed for use in visual object recognition software research. It contains over 1 million high-resolution tagged images in approximately 1000 classes. We are going to use its weights to get a base from which we can start training the model.
2. Since we are doing transfer learning, we are not going to include the top layers of the ResNet50 model. So include_top = False to exclude the final Dense layer. Replacing the top layer with custom layers allows us to use ResNet50 as a feature extractor in a transfer learning workflow.
3. Define input_shape = (256, 256, 3)
4. Add our own layers on the top of ResNet50:

GlobalAveragePooling2D: designed to replace fully connected layers. It takes the average of each feature map.

Dropout to prevent overfitting by reducing the number of neurons.

1-fully-connected or 2-fully-connected and 3-fully-connected layers followed by output layer with activation softmax since this is a multi-class classification problem.

The model has around 25.6 million parameters.

### Conclusion
In our project, we propose a transfer learning-based method for Covid-19, Bacterial Pneumonia, Viral Pneumonia and Normal lungs detection via X-Ray images. The method uses the ResNet50 model architecture and weights pretrained on the popular ImageNet dataset. Modification of the network’s output was made to take the final diagnosis decisions. We unfreezed last 20 layers of ResNet50 and searched for the optimal hyperparameters and network architecture through loops.

The model implemented along with reducing learning rate and tuning hyperparameter proves to be efficient on a large and complex problem of X-Ray image data set.

First, we checked performance with 1-fully-connected layer, then with 2, 3 and 4-fully-connected layers, and with only output layer.

Models showed almost similiar results above 70%, but the best performance was shown by a network with 2-fully connected layers with accuracy 78% and f1-score on Covid-19: 90%, Normal lungs: 78%, Viral pneumonia: 71%, Bacterial pneumonia: 70%.

* Model with 1-fully connceted layer showed accuracy score at 75% 
* Model with 3-fully connceted layer showed accuracy score at 78% 
* Model with 4-fully connceted layer showed accuracy score at 73% 
* Model with output layer showed accuracy score at 75%

The best hyperparameters for transfer learning with ResNet50 in our project were:

- 2 Dense layers
- GlobalAveragePooling2D
- Adagrad optimizer
- callbacks: early stopping & reducing learning rate
- dropout

Further, in the next work, we will test the dataset with Xception network.
