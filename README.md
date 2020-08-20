# Estimation-Theory
 This contains the course project codes for EE5111 at IIT-Madras


The code contains the linear version of the BCDL code for the paper:
```
Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian compression for deep learning." Advances in Neural Information Processing Systems. 2017.
```

Although the backbone of the code is the same as in the official repo, some aspects of the code have been modified


For getting pre-trained model, run: 

```
python mnist_nn.py
```

This will save a pretrained neural network that will be used by the bayesian network (BN) for initialization.

For running the BN with random initialization, run:
```
python example.py
```

For running the same BN initialized with pretrained weights, run:
```
python example.py --load_pretrained
```


Additionally, you can prefix the above command with 'CUDA_VISIBLE_DEVICES=X' to select the Xth GPU on your system. 
For example, running the following command will train the BN initialized with pretrained weights on GPU 1:

```
CUDA_VISIBLE_DEVICES=1 python example.py --load_pretrained
```

To play around with the code and the trained models on CPU, use the notebook ```BCDL_playground.ipynb```. 

