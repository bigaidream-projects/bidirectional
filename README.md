Bidirectional Propagation
==========================================================================================

This repo contains code to reproduce experiments in paper
"Bidirectional Backpropagation: Towards Biologically Plausible Error Signal Transmission in Neural Networks"
(https://arxiv.org/abs/1702.07097)

This code and readme is copied and modified based on https://github.com/anokland/dfa-torch (Direct Feedback Alignment using Torch)
Supported datasets are {Cifar10, MNIST}

## Dependencies
* Torch (http://torch.ch)
* "DataProvider.torch" (https://github.com/eladhoffer/DataProvider.torch) for DataProvider class.
* "cudnn.torch" (https://github.com/soumith/cudnn.torch) for faster training. Can be avoided by changing "cudnn" to "nn" in models.
* "dpnn" (https://github.com/Element-Research/dpnn) for maxnorm constraints on weights
* "unsup" (https://github.com/koraykv/unsup) for whitening of data
* "mnist" (https://github.com/andresy/mnist) for MNIST data set

To install all dependencies (assuming torch is installed) use:
```bash
luarocks install https://raw.githubusercontent.com/eladhoffer/eladtools/master/eladtools-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/eladhoffer/DataProvider.torch/master/dataprovider-scm-1.rockspec
luarocks install dpnn
luarocks install unsup
luarocks install https://raw.github.com/andresy/mnist/master/rocks/mnist-scm-1.rockspec
```

## Training
To train and evaluate the Bidirectional Direct Feedback Alignment model on MNIST and CIFAR10 datasets,
```lua
th Main.lua -dataset MNIST -network mlp.lua -LR 1e-4
```
or,
```lua
th Main.lua -dataset Cifar10 -network conv.lua -LR 5e-5
```

## Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|modelsFolder     |  ./Models/           | models Folder
|network          |  mlp.lua             | model file - must return valid network.
|criterion        |  bce                 | criterion, ce(cross-entropy) or bce(binary cross-entropy)
|eps              |  0                   | adversarial regularization magnitude (fast-sign-method a.la Goodfellow)
|dropout          |  0                   | 1=apply dropout regularization
|batchnorm        |  0                   | 1=apply batch normalization
|nonlin           |  tanh                | nonlinearity (tanh,sigm,relu)
|num_layers       |  2                   | number of hidden layers (if applicable)
|num_hidden       |  800                 | number of hidden neurons (if applicable)
|bias             |  1                   | 0=do not use bias
|rfb_mag          |  0                   | random feedback magnitude, 0=uniform distribution in [-1/sqrt(fanout),1/sqrt(fanout)], X=uniform distribution in [-X,X], X=1 works fine with SGD
|LR               |  0.0001              | learning rate
|LRDecay          |  0                   | learning rate decay (in # samples)
|weightDecay      |  0                   | L2 penalty on the weights
|momentum         |  0                   | momentum
|batchSize        |  64                  | batch size
|optimization     |  rmsprop             | optimization method(sgd,rmsprop,adam etc)
|epoch            |  300                 | number of epochs to train (-1 for unbounded)
|epoch_step       |  -1                  | learning rate step, -1 for no step, 0 for auto, >0 for multiple of epochs to decrease
|gradient         |  dfa                 | gradient for learning, bp(back-prop), fa(feedback-alignment) or dfa(direct feedback-alignment)
|maxInNorm        |  400                 | max norm on incoming weights
|maxOutNorm       |  400                 | max norm on outgoing weights
|accGradient      |  0                   | 1=accumulate normal and adversarial gradient (if eps>0)
|threads          |  8                   | number of threads
|type             |  cuda                | float or cuda
|devid            |  1                   | device ID (if using CUDA)
|load             |  none                | load existing net weights
|save             |  time-identifier     | save directory
|dataset          |  MNIST               | dataset - Cifar10, Cifar100, STL10, SVHN, MNIST
|datapath         |  ./Datasets/         | data set directory
|normalization    |  scale               | scale(between 0 and 1), simple(whole sample,mean=0,std=1), channel(by image channel), image(mean and std images)
|format           |  rgb                 | rgb or yuv
|whiten           |  false               | whiten data
|augment          |  false               | augment training data
|preProcDir       |  ./PreProcData/      | data directory for pre-processing (means,Pinv,P)
|validate         |  false               | use validation set for testing instead of test set
|visualize        |  0                   | 1=visualizing results
