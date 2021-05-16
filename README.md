# Gestures Classification

## Introduction
Classification of Human Gestures using an in-house dataset collected from the students of my class, using different models like AlexNet, VGG16, ResNet pre-trained on ImageNet using PyTorch and Transfer Learning.

## Overview of the Repository
In this repo, you'll find :
* `dataset`: pictures of my classmates thumbs labelled `up` or `down`. 
* `cat`: pictures of different kinds of cats to test pre-trained VGG on ImageNet.
* `imagenet_labels.py`: ImageNet labels for 1000 class.
* `like_classifier.py`: thumbs classification for thumbs up/down using VGG-16.
* `VGG_classifier_cats.py`: cats classifier using VGG-16.

## Getting Started
1.  Clone repo: `git clone https://github.com/HusseinLezzaik/Gestures-Classification.git`
2.  Install dependencies:
    ```
    conda create -n gestures-classification python=3.8
    conda activate gestures-classification
    pip install -r requirements.txt
    ```
3. Run `like_classifier.py` for classifying thumbs as up/down, you can play with the test set by yourself ;)
4. Run `VGG_classifier_cats.py` to discover the different kinds of cats on our planet.

## Contact
* Hussein Lezzaik : hussein dot lezzaik at gmail dot com
