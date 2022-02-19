Learn to use PyTorch with an actual example, whoop!

# Colab Notebook

[Link to Colab Notebook](https://colab.research.google.com/drive/1lh6QDjsKYFrXO2l3tRfW8qW0Z2WfCtj7)

# Processing

Data augmentation by way of **random horizontal flips** is used. Pixels are **normalized** by taking mean and standard deviation to be 0.286 and 0.126, respectively. This is approximately the average value of pixels in the training set. Data is then **converted to tensors** for modeling.

# Modeling

Since the images are small in size (28x28), I do not see the need for a large model a.k.a. **transfer learning** from pretrained models (typically taking in 224x224 images) such as ResNet. Thus, I trained a convolutional neural network with architecture **similar to LeNet-5**; convolutional layers, max-pooling, and fully-connected layers at the end. Dropouts are utilised to reduce overfitting. The final model size with the setup as is, is **4.36MB** with **a little over 1 million parameters**, which can be considered very lightweight.

# Training

Train/val split is **80/20**. Optimizer choice is **Adam** and **SGD**, with default being Adam. While SGD will require a little bit of tuning depending on the other hyperparameter choices, Adam performs better on default parameters. Will be optimized based on **cross entropy loss**. Dataset seems to be balanced enough to not require customised losses (e.g. focal loss).

Batch size of **16/32** is preferred. One epoch will run in about **20/15 seconds** with Colab Tesla T4 GPU for batch size 16/32, due to difference in vectorization efficiency. Batch size 16 tends to produce better results. This is a tradeoff to be decided by the **use case**.

The best models will be **automatically saved**, and the training can be chosen to terminate by means of **early stopping** on the validation set loss (can be changed to validation accuracy too).

# Results

With the current setup, accuracy of a single model **consistently beat 91.3%** on the test set. With some luck in the initialization, it may reach close to 92%. In comparison to 92% accuracy from ResNet18 more than 10 times its size [as tested by Kyriakos Efthymiadis](https://github.com/kefth/fashion-mnist), it is indeed an admirable result and efficient training for such a small network. With ensemble of similar models of even, the same architecture, we can reach better accuracy. This, in PyTorch API is a simple population of saved parameters from multiple trained models onto the network. However, I decided not to do so as to keep it quick and simple.

# Utilities

The class **NetworkTrainer** allows to instantiate objects that captures the dynamics of the training
process. The attribute **results** provides the losses/accuracies by epoch in the format of a Pandas DataFrame. In production, we might want to save these results to **monitor** the model training process. However, to keep the submission simple, I decided to skip this idea. The method **visualize_results** shows the trajectory of the loss and accuracy of the model as it iterates through multiple epochs.

# Others

**Ensembling** multiple trained models can be done if the use case allows a lot of training time. I expect the results to be better than a single trained model, especially if the errors are not very correlated.

A simple implementation of ensembling with uncorrelated model, Random Forest is explored at the end of the notebook. This is advised against, though, as the model size of Random Forest is much larger than the network.

A **warning** might pop up, as the latest PyTorch version (installed on Colab) has an issue regarding compatibility with Tesla T4 GPU on Colab. Issue was raised just 4 days ago. If Colab assigns Tesla T4 GPU for you, this warning might show. I noticed no abnormal behaviour and training goes as expected, but in your case it might be different. [Link to issue](
https://discuss.pytorch.org/t/pytorch-1-6-tesla-t4-with-cuda-capability-sm-75-is-not-compatible/91003).

Packages used are:
* os
* time
* glob
* copy
* pickle
* numpy
* pandas
* matplotlib
* torch
* torchvision

Which are either part of the Python standard library, basic data science stack, and a deep learning framework.
