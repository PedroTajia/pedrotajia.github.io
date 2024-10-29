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

### Contrastive Learning
The idea of contrastive learning is to make a model learn an embedding space that captures the essential information about its inputs including their structure and semantics.  

**Example:**
The model $f_\Theta$ w have an image input of 224 pixels by 224 pixels which have $[24*24] = 576$ dimensions. The model outputs a vector that represent the input of $16$ dimensions which occupies 36 times less space than the image with almost the same information.

This is done by training the model to output vector representations that are close for similar examples and farther apart when there are different examples. To training the model is used three types example, the anchor example (image as a reference), a positive example (image closely related to the anchor example) and a negative example (an image that is not related to the anchor example).

**Example:**
Imagine the task to create a model that discriminate between animals and non-animals. The inputs for the model will be an image of a dog, cat and a watermelon. The **anchor example $x^a$**(dog), **positive example $x^+$** (cat) and the **negative example $x^-$** (watermelon). The model which have a CNN denoted as $f_\Theta$ (CNN are the one that gets the structure and meaning of the image) and a projection $g_\Theta$ (a projection head is applied to map the representations of $f_\Theta $ to its loss function). When the image of a dog and a cat is imputed to the model it should outputs similar vector representations
![Example of similar example](/assets/bootstrap-your-own-latent/CL-Explication-positive.svg)


And vice-versa when the negative example is inputted to the model the vector representation is completely different and far from the representation of the anchor image.
![Example of different example](/assets/bootstrap-your-own-latent/CL-Explication-negative.svg)

The paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)(SimCLR) is the foundation on implementing contrastive methods for self-supervised learning. In the paper SimCLR was introduced a loss called **NT-Xent** which was originally inspired on the **InfoNCE** just having $\tau$ temperature variable as a modification.

**InfoNCE**
<span style="font-size: 1em;">$\ell_{i,j} = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$</span>

<!-- **NT-Xent**
<span style="font-size: 2em;">$\ell_{i,j} = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$</span> -->

The InfoNCE loss will enforce $x^a$ and $x^+$ to be similar pairs and also enforce pairs that are different. The sim(.) function is a similarity metric which measure how similar is a vector against others. This metric is used to minimize the difference between positive pairs $(x^a, x^x)$ and maximize the distance between negative pairs $(x^a, x^-)$

In summary, we can think a contrastive task as trying to generate similar representation for positive examples and different for negative examples.

## Bootstrap Your Own Latent: BYOL
In contractive learning positive example are easy to obtain, but negative examples are hard to get. Positive examples can be just a modified version of the anchor image. Negative is difficult to obtain because we to define what this is different to an anchor and have enough similarity to make a challenging to the model without human intervention.

For this reason there is research of self-supervised learning without contrastive learning. This is difficult because there is a need for negative example, if not what can stop the model of generating the same vector representation for the anchor example and positive example which is called *collapse*. With negative examples the model is force to learn meaning representations for its inputs.

In order to make **BYOL** archive self-supervised learning without contrastive methods it needs many innovative things.  
BYOL have two neural networks, named as *online* and *target* networks that are able to interact to each other.
The model is trained the online network to predict the target network representation with the same image using different augmented views.





 