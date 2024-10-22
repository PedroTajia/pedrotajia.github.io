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

Since more research have been done on training CNNs with self-supervised, 
<!-- There are many ways of implementing transfer learning. First is **Fine-tuning** which retrain the model on the chosen layer for the selected dataset. -->


