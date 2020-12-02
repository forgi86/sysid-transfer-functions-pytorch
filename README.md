# Deep learning with transfer functions: new applications in system identification
 

This repository contains the Python code to reproduce the results of the paper "Deep learning with transfer functions: new applications in system identification" by Dario Piga, Marco Forgione, and Manas Mejari.

We describe a linear dynamical operator described in terms of a rational transfer function, endowed with a well-defined and efficient back-propagation behavior for
automatic derivatives computation. In the dynoNet architecture (already introduced [here](https://github.com/forgi86/dynonet)), linear dynamical operators are combined with static (i.e., memoryless) non-linearities which can be either elementary
activation functions applied channel-wise; fully connected feed-forward neural networks; or other differentiable operators. 

In this work, we use the differentiable transfer function operator to tackle
other challenging problems in system identification. In particular:

* Learning in the presence of colored noise (prediction error minimization method)
* Learning with quantized output measurements (maximum likelihood estimation method)

# Folders:
* [torchid](torchid_nb):  PyTorch implementation of the linear dynamical operator (aka G-block in the paper) used in dynoNet
* [examples](examples): examples using dynoNet for system identification 
* [util](util): definition of metrics R-square, RMSE, fit index 

Two [examples](examples) discussed in the paper are:

* [WH2009](examples/WH): A circuit with Wiener-Hammerstein structure. Experimental dataset from http://www.nonlinearbenchmark.org
* [Parallel Wiener-Hammerstein](examples/ParWH): A circuit with a two-branch parallel Wiener-Hammerstein structure. Experimental dataset from http://www.nonlinearbenchmark.org


For the [WH2009](examples/WH2009) example, the main scripts are:

 *  ``WH2009_train_colored_noise_PEM.py``: Training of a dynoNet model with the prediction error method in presence of colored noise
 *  ``WH2009_test.py``: Evaluation of the dynoNet model on the original test dataset,  computation of metrics, plots.
  
For the [Parallel Wiener-Hammerstein](examples/ParWH) example, the main scripts are:

 *  ``parWH_train_quant_ML.py``: Training of a dynoNet model with maximum likelihood in presence of quantized measurements
 *  ``parWH_test.py``: Evaluation of the dynoNet model on the original test dataset,  computation of metrics, plots.


NOTE: the original data sets are not included in this project. They have to be manually downloaded from
http://www.nonlinearbenchmark.org and copied in the data sub-folder of the example.
# Software requirements:
Simulations were performed on a Python 3.7 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * numba
 * pytorch (version 1.6)
 
These dependencies may be installed through the commands:

```
conda install numpy scipy pandas numba matplotlib
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
