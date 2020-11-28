# Deep learning with transfer functions: new applications in system identification
 

This repository contains the Python code to reproduce the results of the paper "Transfer functions and deep learning with dynoNet : new applications in system
identification" by Dario Piga, Marco Forgione, and Manas Mejari.

We describe the linear dynamical operator as a differentiable layer compatible with back-propagation-based training. 
The operator is parametrized as a rational transfer function and thus can represent an infinite impulse response (IIR)
filtering operation, as opposed to the Convolutional layer of 1D-CNNs that is equivalent to finite impulse response (FIR) filtering.

In the dynoNet architecture (already introduced [here](https://github.com/forgi86/dynonet)), linear dynamical operators are combined with static (i.e., memoryless) non-linearities which can be either elementary
activation functions applied channel-wise; fully connected feed-forward neural networks; or other differentiable operators. 

In this work, we show how to non-standard learning problems may be tackled using the differentiable 
transfer function block, namely:

* Learning with quantized measurements
* Learning in the presence of colored noise

# Folders:
* [torchid](torchid_nb):  PyTorch implementation of the linear dynamical operator (aka G-block in the paper) used in dynoNet
* [examples](examples): examples using dynoNet for system identification 
* [util](util): definition of metrics R-square, RMSE, fit index 

Two [examples](examples) discussed in the paper are:

* [Parallel Wiener-Hammerstein](examples/ParWH): A circuit with a two-branch parallel Wiener-Hammerstein structure. Experimental dataset from http://www.nonlinearbenchmark.org
* [WH](examples/WH): A circuit with Wiener-Hammerstein structure. Experimental dataset from http://www.nonlinearbenchmark.org


For the [WH2009](examples/WH2009) example, the main scripts are:

 *  ``WH2009_train.py``: Training of the dynoNet model
 *  ``WH2009_test.py``: Evaluation of the dynoNet model on the test dataset,  computation of metrics.
  
Similar scripts are provided for the other examples.

NOTE: the original data sets are not included in this project. They have to be manually downloaded from
http://www.nonlinearbenchmark.org and copied in the data sub-folder of the example.
# Software requirements:
Simulations were performed on a Python 3.7 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * pytorch (version 1.4)
 
These dependencies may be installed through the commands:

```
conda install numpy scipy pandas matplotlib
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
