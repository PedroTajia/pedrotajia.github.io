---
layout: post
title: "Contrastive Deep Explanations (In Progress)"
author: Pedro Tajia
tags: [Explainable AI, Deep Learning]
image: /assets/contrastive-deep-explanations/proposed-approach.png
---
<script type="text/javascript" async
     src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# 1. Introduction 

Since the creation Deep Learning researches have the curiosity of knowing what happen inside these deep learning model that in everyday we use. Even do many people use deep learning in their daily basis like grammar checker, self-driving cars, weather prediction or in more niche topic like cancer detection or prediction of protein's 3D structure from amino acid sequence, etc., nobody really knows how these models work internally. In cases where deep learning is use on medicine or in self-driving cars is important to know the reasons of model decision, for example knowing why did not the model chose prediction **B** over the prediction **A**. 

Note: I will interchangeably use model or a neural network with the same meaning.

In this article I will explain the paper [CDeepEx: Contrastive Deep Explanations](https://rlair.cs.ucr.edu/papers/docs/cdeepex.pdf) where proposed a method capable of answer the question of *Why did you not choose answer B over A*. This paper provided an overview of the concepts captured by a network.

![Contrastive Example](/assets/contrastive-deep-explanations/Contrastive_example.svg)
**Figure 1**: This is an example of a classifier trained on the MNIST dataset, to predict the digit that appears in the image. The input of the model is an image that have a number **3**, the model predict with a confidence of *72%*. In the paper solve the question of *Why the classifier predicts the number **3** instead of the number **9*** by showing how the image 3 can be change in order to make the classifier predict the number 9.

# 2. CDeepEx: Contrastive Deep Explanations
To archive the question of *Why did you not choose answer B over A*, the proposed  method in this paper uses generative latent-space model. These are just Generative Adversarial Networks (GAN) or autoencoder (in this paper are the Variational autoencoder), where learns the latent-space to generate data. The latent-space can view as the compress or the essential information that compose the data. In this paper [CDeepEx: Contrastive Deep Explanations](https://rlair.cs.ucr.edu/papers/docs/cdeepex.pdf) the latent-space is view as a bridge between the network and human understanding of the data. The idea is to use a GAN or variational autoencoder (VAE) to create a latent-space for later be used to generate images that can explain the reasoning of a model, like a classifier of numbers or faces. 

Note: There is a big variety of GAN the specific one used is this paper is wasserstein GAN (WGAN) that give a more stability when training and yields better generated images than a vanilla GAN. I will use interchangeably GAN and WGAN

![GAN-VAE](/assets/contrastive-deep-explanations/GAN-VAE.svg)
**Figure 2**: These are two generative models (a) is a variational autoencoder (VAE), an image is inputted on the encoder and with the **code** generates a latent representation where the input (image) is transformed into a lower dimension that have equivalent information of the input. The decoder use this latent representation to do a reconstruction that is similar to the input. (b) uses a random noise to do generations of images that look similar to the data. The discrimination predict how real an image it is.

Note: The latent representation is just a point in the latent space, the job of a GAN or VAE create this latent space which contains the information the model learn from the data.

The **code** and the **random noise** can be view as the latent space. Since the **code** and the **random noise** comes from a normal distribution is possible to sample a point from normal distribution and inputted in the decoder for the VAE and the generator for the GAN to generate an image. The only part that is going to use to generate the explications is the **decoder** for the VAE and the **generator** for the GAN.

In order to generate explication is use a generator (a network that generate natural images) and the discriminator (the classifier of interest). The image $\mathcal{I}$ is inputted in the discriminator network $D$ to produce $\mathcal{y}_{true}$, the class label of interest will be denoted as $\mathcal{y}_{probe}$. With this we can formulate the question of *Why did $D$ produce label $\mathcal{y}_{true}$ and not label $\mathcal{y}_{probe}$ for the input $\mathcal{I}$?*.

To generate explanation:
![Algorithm_1-Algorithm_2](/assets/contrastive-deep-explanations/Generate_explanation.png)

On these algorithms, the first step is to train a generator $G$ that giving a latent representation with real-values of size $k$ outputs an image with size $n$. After the generator is trained, we need to find a representation $z_0$ that inputted to $G$ generates a similar image of $\mathcal{I}$. $z_0$ is sampled from a normal distribution $\mathcal{N}(0,1)$ with mean 0 with variance of 1. There will be loop until the generated image $G(z)$ is close to the image $\mathcal{I}$. Inside the loop, $z_0$ is updated by gradient decent. 
<!-- The gradient $\nabla_z$ is calculated $loss$ with respect to $z_0$, the loss can be obtained from l2 distance or binary cross entropy for images. $z_0$ is subtracted by the gradient $\nabla_z$ multiplied by a learning rate $\eta$ to get the new $z_0$.  -->
After finding the right latent representation that $G(z_0)$ will generate a similar image of $\mathcal{I}$, we get $\Delta_{z0}$ as the difference between $G$ and $\mathcal{I}$.

To find $z_e$:
![Getting_Ze](/assets/contrastive-deep-explanations/Getting_Ze.png)

After having a $z_e$ that have the minimum l2 squared distance with respect to $z$ between $z$ and $z_0$, and is in between the constraints is performed the difference between $G(z_0) - G(z_e)$

![proposed_method](/assets/contrastive-deep-explanations/Proposed_approach.svg)






<!-- $\mathcal{I}_{z,z_0}$ = $G(z)+\Delta_{z_0}$ and $llh(f, y)$ is the log-likelihodd of class $y$ for the output $f$. The first constraint  -->





<!-- 1. Learn a function $G$: $\R^{k} \rightarrow \R^{n}$
1. Find a latent representation for input $\mathcal{I}$
   1. **procedure** Learn $\mathcal{z}_0 (G, \mathcal{I}, \eta, loss(.))$
     $z_0 \sim \ni$ -->
   

     


