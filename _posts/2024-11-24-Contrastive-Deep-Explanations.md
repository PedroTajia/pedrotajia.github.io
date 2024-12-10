---
layout: post
title: "Contrastive Deep Explanations (In Progress)"
author: Pedro Tajia
tags: [Explainable AI, Deep Learning]
image: /assets/contrastive-deep-explanations/Preview.svg
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

In order to generate explication is use a generator (a network that generate natural images) and the discriminator (the classifier of interest). The image $\mathcal{I}$ is inputted in the discriminator network $D$ to produce 
$y_{true}$, 
the class label of interest will be denoted as 
$y_{probe}$. 
With this we can formulate the question of 
*Why did $D$ produce label $y_{true}$ and not label $y_{probe}$ for the input $\mathcal{I}$?*.

To generate explanation:
![Algorithm_1-Algorithm_2](/assets/contrastive-deep-explanations/Generate_explanation.png)

On these algorithms, the first step is to train a generator $G$ that giving a latent representation with real-values of size $k$ outputs an image with size $n$. After the generator is trained, we need to find a representation $z_0$ that inputted to $G$ generates a similar image of $\mathcal{I}$. $z_0$ is sampled from a normal distribution $\mathcal{N}(0,1)$ with mean 0 with variance of 1. There will be loop until the generated image $G(z)$ is close to the image $\mathcal{I}$. Inside the loop, $z_0$ is updated by gradient decent. 
<!-- The gradient $\nabla_z$ is calculated $loss$ with respect to $z_0$, the loss can be obtained from l2 distance or binary cross entropy for images. $z_0$ is subtracted by the gradient $\nabla_z$ multiplied by a learning rate $\eta$ to get the new $z_0$.  -->
After finding the right latent representation that $G(z_0)$ will generate a similar image of $\mathcal{I}$, we get $\Delta_{z0}$ as the difference between $G$ and $\mathcal{I}$.

To find $z_e$:
![Getting_Ze](/assets/contrastive-deep-explanations/Getting_Ze.png)

After having a $z_e$ that have the minimum l2 squared distance with respect to $z$ between $z$ and $z_0$, and is in between the constraints is performed the difference between $G(z_0) - G(z_e)$

![proposed_method](/assets/contrastive-deep-explanations/Proposed_approach.svg)
**Figure 3**: This is a different way to the working the algorithm 1 and 2.



### Different approach
The methods suggested work good on the MNIST dataset showing the transformation need for *Image A* classified as the *Number 8* to be classified as the *Number 3*. In the experiment is show different pair of number and the transformation need to be classified into different class.
![Figure2_mnist_experiment](/assets/contrastive-deep-explanations/Figure2_mnist_experiment.png)

For my approach instead of representing the transformation as red or blue for regions that should be added and regions that should be removed respectively, the transformations are represented as a timeline of all the transformation that have past the *Image 9* to converted into *Image 3*.
![New proposed_method](/assets/contrastive-deep-explanations/New_Approach.svg)
**Figure 4**: The framework to solve the problem with a different view. (a) Use a VAE (Decoder) or GAN (Generator) to generate images, use a *image 9* from the dataset MNISt, the latent vector z is updates to be close to this image, having $z_0$. (b) The updated latent vector $z_0$ as an image is classified as *class 9*, the latent vector $z_0$ is updated again to get $z_e$ which its generated image is predicted by the classifier as the *class 3* in the process to update $z_0$ from *Image 9* to *Image 3* is where is got these sequence of transformations. 

**Figure 4**: Part A
```python
def learn_z0(G, I, lr, loss_fn, epochs, z0):
    # G is the generator
    # I is the image from a dataset
    # lr: Learning rate for the optimizer (default: 0.0005)
    # z0 the random initialized latent vector
    # set the optimizer "Adam" to optimize the loss with respect to the latent variable z0
    optimizer = optim.Adam([z0], lr=lr)
    
    # Iterate over epochs
    for _ in range(epochs):

        # set the calculation of the gradients for z0 
        z0.requires_grad = True

        # generate the image from z0
        G_z0 = G(z0) 

        # Calculate the loss (the loss is the norm l2 squared)
        loss = loss_fn(G_z0, I) 

        # backpropagate the error and get the gradients
        loss.backward()

        # update z0
        optimizer.step()
    return z0, 
```
**Figure 4**: Part B
```python
def learn_ze(G, D, epochs, z, y, lr=5e-4, some_pixel_threshold=5):
    # G: Generator model
    # D: Discriminator model
    # epochs: Number of training iterations
    # z: Latent vector (input noise for the generator)
    # y: Target labels for the discriminator's output
    # lr: Learning rate for the optimizer (default: 0.0005)

    # some_pixel_threshold: Threshold for pixel difference to store generated images (default: 5)

    # Ensure the latent vector requires gradient computation
    z.requires_grad = True

    # Initialize the Adam optimizer to update the latent vector z
    optimizer = optim.Adam([z], lr=lr)

    # List to store generated images that meet the pixel difference criterion
    grid_images = []

    # Variable to store the previously generated image for comparison
    prev_stored_image = None

    # Training loop over the specified number of epochs
    for _ in range(epochs):
        # Generate an image from the latent vector z and pass it through the discriminator
        D_z, G_z_resize = discriminator_gen(G, D, z)

        # Compute the cross-entropy loss between the discriminator's output and the target labels
        loss = F.cross_entropy(D_z, y)

        # Backpropagate the loss to compute gradients
        loss.backward()

        # Update the latent vector z using the optimizer
        optimizer.step()

        # Check if there's a previously stored image to compare with
        if prev_stored_image is not None:
            # Calculate the pixel-wise difference between the current and previous images
            pixel_diff = torch.norm(G_z_resize - prev_stored_image).item()
            # If the difference exceeds the threshold, store the current image
            if pixel_diff > some_pixel_threshold:
                grid_images.append(G_z_resize[0].detach().cpu().permute(1, 2, 0))
                # Update the previous stored image to the current one
                prev_stored_image = G_z_resize.clone().detach()
        else:
            # If no previous image exists, store the current image
            grid_images.append(G_z_resize[0].detach().cpu().permute(1, 2, 0))
            # Set the previous stored image to the current one
            prev_stored_image = G_z_resize.clone().detach()

    # Return the optimized latent vector and the list of stored images
    return z, grid_images
```


I believe this approach is useful to understand how the network classifier goes through the latent space to find the image the outputs the correct label. Instead of depending on two variables to change an *image A* to an *image B*, is shown a sequence of transformation applied to *image A* to become *image B*.




<!-- $\mathcal{I}_{z,z_0}$ = $G(z)+\Delta_{z_0}$ and $llh(f, y)$ is the log-likelihodd of class $y$ for the output $f$. The first constraint  -->





<!-- 1. Learn a function $G$: $\R^{k} \rightarrow \R^{n}$
1. Find a latent representation for input $\mathcal{I}$
   1. **procedure** Learn $\mathcal{z}_0 (G, \mathcal{I}, \eta, loss(.))$
     $z_0 \sim \ni$ -->
   

     


