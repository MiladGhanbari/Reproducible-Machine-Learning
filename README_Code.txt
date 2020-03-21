Libraries Used: Scikit-Learn, OS, Numpy, pickle, keras, math, tensorflow


Instructions:
Below is the instructions in order to obtain the result shown in the report. 


Run CIFAR10.py and CIFAR100.py for reproducing the results reported in Table 1. Pay attention to the comments in the code. 
The default value for reduction is 0 and the bottleneck is set to False. 
These parameters should be set before runing:
	-- depth
	-- growth_rate
	-- reduction
	-- bottleneck
	-- dropout_rate : 0.0 for data augmentation and 0.2 otherwise

For approach A, B and C run CIFAR10_A.py, CIFAR10_B.py, CIFAR10_C.py respectively.

Notice: make sure that you have two folders named logs and weights in the directory where CIFAR10.py and CIFAR100.py exist.
	make sure that al .py files are in the same directory.