---
layout: post
title: "Bootstrap Your Own Latent: Self-Supervised Learning Without 
Contrastive Learning"
author: Pedro Tajia
tags: [Self Supervised, Deep Learning]
image: /assets/bootstrap-your-own-latent/BYOL-Architecture.png
---

## Introduction

Supervised Learning is the top solution to train Convolutional Neural Network (CNN), which is the model made to solve computer vision like self-driving car, security, automatization, etc. As these computer vision tasks increase, the need to improve these systems also emerges.

One of the main ways to improve these systems is through having more data and bigger models, but unfortunately by using supervised learning these solutions need labeled data (information that has been classified by humans), **which makes them expensive and difficult to get**.

The bottleneck that limits the process of training is the process itself. Since supervised learning is dependent on label data which limits the amount of information the model are trained on. This represents a big limitation on the training of these models and a waste of unlabeled data. Current research on Self-Supervised Learning is able to **train model without label data and take out the need of using Contrastive Learning** which is the most common way to train model with Self-Supervised Learning. 

## Self-Supervised Learning
Self-Supervised learning gained popularity on the training of Large Language models, by having the model learn from many unlabeled data which make the model learn the general structure of the words and their meaning. With that general knowledge then these pre-trained models are use for transfer learning to solve more specific take. 

### Transfer Learning
>Transfer learning is the process of using a model like CNNs trained on a large corpus of labeled data to gain a general structure and meaning of images and then use it to solve a more specific tasks. The use of large label data is to ensure the model learns a broad variety of images. Normally transfer learning is used when there is a limited amount of data, limited computational power or the improvement of performance. Even do it seems that transfer learning can solve the problem of using datasets with small label data still it does not. Since a key component of using transfer learning is its implementation, the data needs to be similar or close to the large corpus of data that was used to train the model.

Since more research has been done on training CNNs with self-supervised learning, there has been new approaches to make able CNN learn from unlabeled data. One of these approaches was introduced by the paper called [bootstrap your own latent](https://arxiv.org/pdf/2006.07733) where demonstrates that is not necessary of use contrastive learning approach for a self-supervised setting.

### Contrastive Learning
The idea of contrastive learning is to make a model learn an embedding space that captures the essential information about its inputs including their structure and semantics.  

**Example:**
The model $f_\theta$ w have an image input of 224 pixels by 224 pixels which have $[24*24] = 576$ dimensions. The model outputs a vector that represents the input of $16$ dimensions which occupies 36 times less space than the image with almost the same information.

This is done by training the model to output vector representations that are close for similar examples and farther apart when there are different examples. To train the model three type exampled we use: the anchor example (image as a reference), a positive example (image closely related to the anchor example) and a negative example (an image that is not related to the anchor example).

**Example:**
Imagine the task to create a model that discriminate between animals and non-animals. The inputs for the model will be an image of a dog, cat and a watermelon. The **anchor example $x^a$**(dog), **positive example $x^+$** (cat) and the **negative example $x^-$** (watermelon). The model which has a CNN denoted as $f_\theta$ (CNN is the one that gets the structure and meaning of the image) and a projection $g_\theta$ (a projection head is applied to map the representations of $f_\theta$ to its loss function). When the image of a dog and a cat is imputed to the model it should output similar vector representations
![Example of similar example](/assets/bootstrap-your-own-latent/CL-Explication-positive.svg)


And vice-versa when the negative example is inputted to the model the vector representation is completely different and far from the representation of the anchor image.
![Example of different example](/assets/bootstrap-your-own-latent/CL-Explication-negative.svg)

The paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)(SimCLR) is the foundation on implementing contrastive methods for self-supervised learning. In the paper SimCLR was introduced a loss called **NT-Xent** which was originally inspired on the **InfoNCE** just having $\tau$ temperature variable as a modification.

**InfoNCE**
<span style="font-size: 1em;">$\ell_{i,j} = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$</span>

<!-- **NT-Xent**
<span style="font-size: 2em;">$\ell_{i,j} = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$</span> -->

The InfoNCE loss will enforce $x^a$ and $x^+$ to be similar pairs and also enforce pairs that are different. The sim(.) function is a similarity metric which measures how a vector is similar against others. This metric is used to minimize the difference between positive pairs $(x^a, x^x)$ and maximize the distance between negative pairs $(x^a, x^-)$

In summary, we can think of contrastive tasks as trying to generate similar representation for positive examples and different for negative examples.

## Bootstrap Your Own Latent: BYOL
In contrastive learning positive examples are easy to obtain, but negative examples are difficult to get. Positive examples can be just a modified version of the anchor image. Negative examples can be difficult to get because we to define what this is different to an anchor and have enough similarity to make a challenging to the model without human intervention.

For this reason there is research of self-supervised learning without contrastive learning. This is difficult because there is a need for negative example, if not what can stop the model of generating the same vector representation in contrast to an anchor example and positive example which is called *collapse*. For negative examples the model is forced to learn meaning representations for its inputs.

In order to understand how **BYOL** archive self-supervised learning without contrastive methods let's explore the main components of this self-supervised learning framework.

BYOL have two neural networks, named as *online* and *target* networks that are able to interact to each other.
The model is trained by the online network to predict the target network representation with the same image using different augmented views.

![First augmentation Example](/assets/bootstrap-your-own-latent/Augmentation_1.svg)

To generate this augmented views, we create 2 distortionated copies form an input image, by applying two sets of data augmentation operations. The transformation includes 

>* random cropping: a random patch of the image is selected, with an area uniformly sampled between 8% and 100% of that of the original image, and an aspect ratio logarithmically sampled between 3/4 and 4/3. This patch is then resized to the target size of 224 × 224 using bicubic interpolation;
>* Random horizontal flip: optional left-right flip;
>* color jittering: the brightness, contrast, saturation and hue of the image are shifted by a uniformly random offset applied on all the pixels of the same image. The order in which these shifts are performed is randomly selected for each patch;
>* color dropping: an optional conversion to grayscale. When applied, output intensity for a pixel (r,g,b) corresponds to its luma component, computed as $0.2989r+ 0.5870g+ 0.1140b$;
>* Gaussian blurring: for a 224 × 224 image, a square Gaussian kernel of size 23 ×23 is used, with a standard deviation uniformly sampled over $[0.1,2.0]$;
>* solarization: an optional color transformation $x→x·1{x<0.5}+ (1−x)·1{x≥0.5}$ for pixels with values in $[0,1]$.

*Credits: [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/pdf/2006.07733)*


These augmentations double the examples, if we have a batch of 32 images, we end up with 64 images per batch.  
![Second augmentation Example](/assets/bootstrap-your-own-latent/Augmentation_conbination.svg)

Data augmentation is used to force the model to generate representations that have the meaning of the input independent of the distortions applied in the input.

The online network have parameters $\theta$ updated by back propagation and is made from three components: an encoder $f_{\theta}$, projector $g_{\theta}$ and predictor $q_{\theta}$. The target network have an encoder $f_{\xi}$ and projector $g_{\xi}$. The parameters $\xi$ of the target network are not updated by back propagation, but instead the model is updated by exponential moving average of the online parameters $\theta$. The parameters of the target network can be seen a **smoothed version** of the online network.

<span style="font-size: 1em;">${\xi}\leftarrowtail{\tau}{\xi}+(1-\tau){\theta}$</span>
 
> $\tau$ is the decay rate $T\in[0, 1]$

The representation head uses a ResNet-50 for $f_{\theta}$ and {$f_{\xi}$}. The ResNet-50 receives the augmented image of size (224, 224, 3) and output a vector representation or a vector embedding of 2048-dimensional for the online network $y_{\theta}$ and for the target network $y_{\xi}^{'}$. Then a projection head $g$ receives the vector $y$ and produces the final output for the target network $sq(z_{\xi}^{'})$. $sg$ means stop gradient, which the parameters $\xi$ for the target network will not be updated by back-propagation. The output $z_{\theta}$ of the projection head $g_{\theta}$ of the online network is inputted to the prediction head $q_{\theta}$ which produces the final output $q_{\theta}(z_{\theta})$ of the online network. The projection and prediction head consist of a linear layer with an input shape of 2048-dimensions and output size of 4096 followed by **batch normalization**, a non-linear function (ReLU) and a final layer with output of dimension 256.
![Image of the architecture of BYOL, image from the original paper](/assets/bootstrap-your-own-latent/BYOL-Architecture.png)

BYOL is train to minimizes the similarity loss between $q_{\theta}(z_{\theta})$ and $sq(z_{\xi}^{'})$. The loss function is defined as:

$\mathcal{L}_{\theta, \xi} \triangleq \left\| q_{\theta}(z_0) - z'_{\xi} \right\|_2^2 = 2 - 2 \cdot \frac{\langle q_{\theta}(z_0), z'_{\xi} \rangle}{\| q_{\theta}(z_0) \|_2 \cdot \| z'_{\xi} \|_2}$

$q_{\theta}(z_0)$ and $z'_{\xi}$ are normalized to be unit vectors, $\hat{q}_{\theta}(z_0) \triangleq \frac{q_{\theta}(z_0)}{\| q_{\theta}(z_0) \|_2} \quad \text{and} \quad \hat{z}'_{\xi} \triangleq \frac{z'_{\xi}}{\| z'_{\xi} \|_2}$. Then is applied a mean squared error between the normalized outputs of the online and target networks.


