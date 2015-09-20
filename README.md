# Highway Networks

This is sample code for convolutional **Highway Networks**, implemented in Caffe checked out at [this](https://github.com/BVLC/caffe/tree/e20498ebf985322bab2f4f28f0f6365ecde80c29) state. It runs only on the NVIDIA GPUs and requires NVIDIA's cuDNN v2. The original Caffe README is reproduced below the line.

Highway Networks utilize the idea of *information highways*, in turn inspired by LSTM networks [[1](http://www.bioinf.at/publications/older/2604.pdf), [2](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.5709&rep=rep1&type=pdf), [3](http://arxiv.org/abs/1503.04069)].
Our experiments on highway networks show that when designed correctly, neural networks with tens, even hundreds of layers can be trained directly with stochastic gradient descent, thereby providing a promising solution to the vanishing gradient problem. More information is available on the [Project Website](http://people.idsia.ch/~rupesh/very_deep_learning/).

Highway Networks were introduced in the following paper:

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Highway Networks. arXiv preprint [arXiv:1505.00387](http://arxiv.org/abs/1505.00387).

Followed by a more detailed report:

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. arXiv preprint [arXiv:1507.06228](http://arxiv.org/abs/1507.06228).

## Data

You can get preprocessed datasets (CIFAR-10/100 were global contrast normalized and padded with 8 pixels each side) at the links below.
Caffe's data generation scripts do not generate validation sets. The provided data below includes splits into training and validation sets.

[CIFAR-10](https://www.dropbox.com/s/r9zuhhhii4uzi24/cifar10-gcn-leveldb-splits.tar.bz2?dl=0) ~ 2.27 GB

[CIFAR-100](https://www.dropbox.com/s/w2qywjihzr7avfa/cifar100-gcn-leveldb-splits.tar.bz2?dl=0) ~ 2.27 GB

[MNIST](https://www.dropbox.com/s/3q04bu5cz9mha52/mnist-splits.tar.bz2?dl=0) ~ 20 MB


## Examples

See examples and sample log outputs in examples/highways. You probably need to adjust the paths to the datasets in the network definition files in order to train networks.

## Citation

Please cite us if you use this code:

    @article{srivastava2015highway,
        title={Training Very Deep Networks},
        author={Srivastava, Rupesh Kumar and Greff, Klaus and Schmidhuber, J{\"u}rgen},
        journal={arXiv preprint arXiv:1507.06228},
        year={2015}
    }

----


# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
