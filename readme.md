# Model-Agnostic Meta-Learning

MAML is a model-agnostic optimization-based meta learning algorithm. It meta-trains a model to learn a parameter initialization such that it can be fine-tuned to a different task in a single gradient update.

This repository implements second-order MAML on the omniglot dataset

## Requirements
* PyTorch
* OpenCV
* Numpy
* Tqdm

## Usage
1. Download the Omniglot Dataset's [images_background.zip](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip) and [images_evaluation.zip](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip) splits here.
2. Unzip the files in `omniglot/` directory.
3. Run the [train.py](train.py) script to start the training with default options. Run `python train.py -h` to get a description of the arguments.
4. For evaluation, run [evaluate.py](evaluate.py) script.
5. To make predictions on new data, refer [Test.ipynb](Test.ipynb).
6. Alternatively, <a href="https://colab.research.google.com/drive/1AXYhvkXV9ZaAXioWSrV9Sam706dZaAsL?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## References
* https://github.com/oscarknagg/few-shot
* Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks https://arxiv.org/abs/1703.03400