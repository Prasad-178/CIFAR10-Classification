# Image Classification - CIFAR 10
The CIFAR 10 dataset consists of a training set of 50,000 images and a test set of 10,000 images, and is split into 10 classes.
Classification was done using two models and 30 epochs. 

The first one is a basic CNN involving some convolution blocks, batch norm, dropout, dense and max pooling blocks.

The second model is a CNN with residual connections, which helps in reducing the number of parameters and mitigating the potential problem of vanishing/exploding gradients.

# Basic CNN Model
Params - 2,397,226

Training set accuracy - 93.8%

Validation set accuracy - 85.4%

Test set accuracy - 86.3%

# CNN With Residual Blocks
Params - 66,986

Training set accuracy - 89.8%

Validation set accuracy - 77.6%

Test set accuracy - 79.1%

# Conclusion

Clearly the second model performs much better, achieving almost the same accuracy with much lesser number of parameters. This shows the efficiency of ResNets and Residual Blocks in CNN applications.