---
layout: post
title: "Bootstrap Your Own Latent: Self-Supervised Learning Without 
Contrastive Learning"
author: Pedro Tajia
tags: [Self Supervised, Deep Learning]
image: /assets/bootstrap-your-own-latent/BYOL-Architecture.png
---

## Introduction
Supervised Learning are top solution to train Convolutional Neural Network (CNN), which are model made to solve computer vision like self-driving car, security, automatization, etc. As these computer vision tasks increases the need to improve these systems also emerges.

One of the main ways to improve these systems is through having more data and bigger model, but unfortunately by using supervised learning these solutions need labeled data (information that have been classified by humans), **which makes them expensive and difficult to get**.

The bottleneck that limits the process of training is the process itself. Since supervised learning is dependent on label data which limits the amount of information the model are trained on. This represents a big limitation on the training of these model and a waste of unlabeled data. Current research on Self-Supervised Learning has able to **train model without label data and take out the need of using Contrastive Learning** which are the most common way to train model with Self-Supervised Learning. 

## Self-Supervised Learning
Self-Supervised learning gain popularity on the training of Large Language models, by having the model learn from many unlabeled data which make the model learn the general structure of the words and their meaning. With that general knowledge then these pre-trained model are use for transfer learning to solve more specific take. 

### Transfer Learning
>Transfer learning is the process of using a model like CNNs trained on a large corpus of labeled data to gain a general structure and meaning of images and then use it to solve a more specific task. The use of large label data is to ensure the model learn a broad variety of images. Normal transfer learning is used when there is a limited amount of data, limited computational power or the improvement of performance. Even do it seems that transfer learning can solve the problem of using datasets with small label data still it do not. Since a key component of using transfer learning is when implemented, the data need to similar or close to the large corpus of data was used train the model.

Since more research have been done on training CNNs with self-supervised, there has been new approaches to make able CNN learn from unlabeled data. One of these approaches was introduced by the paper called [bootstrap your own latent](https://arxiv.org/pdf/2006.07733) where demonstrates that is not necessary of use contractive learning approach for a self-supervised setting.

## Contrastive Learning
The idea of contrastive learning is to make a model learn an embedding space that captures the essential information about its inputs including their structure and semantics. This is done by training the model to output vector representations that are close for similar examples and farther apart when there are different examples. 
> Example:
> The model $f_\Theta$ have an image input of 224 pixels by 224 pixels which have $[224*224] = 50,176 dimensions$. The model outputs a vector that represent the input of $1,000 dimensions$ which occupies 50 times less space than the image with almost the same information.

To archive this normally is use an anchor example (image as a reference), a positive example (image closely related to the anchor example) and a negative example (an image that is not related to the anchor example).
> Example: 
> Imagine three images a dog, cat and a watermelon. The **anchor example $x^a$**(dog), **positive example $x^+$** (cat) and the **negative example $x^-$** (watermelon). The task is to create a model that discriminate between animals and non-animals. 

<!-- The model is train to generate **vector representation $z$** that have a compress and essential information about its input -->

<!-- There are many ways of implementing transfer learning. First is **Fine-tuning** which retrain the model on the chosen layer for the selected dataset. -->


