# Style-Transfer
Implement Style Transfer with Pytorch

Style transfer is about to generate a fusion image by combining a content image and a style image. The objects and arrangements in the new image are recognized from the content image, while the color, texture and other styles are extracted from the style image.

Below are two excellent illustration on what style transfer is and how amazing it is.

![Style Transfer](https://www.researchgate.net/profile/Jeremiah_Johnson18/publication/330828467/figure/fig1/AS:721849084297216@1549113641325/Two-examples-of-image-style-transfer-generated-using-the-neural-style-algorithm-of-Gatys.ppm)

We will use a pre-trained VGG19 network to accomplish this task. We will extract the output from the second convolutional layer in the fourth stack of the network as the content representation of the image and the outputs from the first convolutional layer from all 5 stacks as the style representation because we want to include all kinds of styles features from the image.

## VGG19: 

This network accepts a colored image as input, and passes it through a series of convolutional and max pooling layers. followed by three fully connected layers to classify the image. 

![VGG19](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/vgg19.png)

Conv 1_1 is the first convolutional layer in the first stack. 
Conv 2_1 is the first convolutional layer in the second stack. 
The deepest convolutional layer in the network is Conv5_4 (the fourth convolutional layer in the fifth stack)

Style Transfer aims to separate the content and style of an image. Given one style image and one content image, we aim to create a new fusion image:

* The objects and their arrangements are similar to that of the content image
* The style, color, and textures are similar to that of the style image. 

When the network sees the content image, it will go through the feed-forward process until it gets to a convolutional layer that is deep enough. The output of this layer will be the content representation of the input image. When it sees the style image, it will extract different features form multiple layers that represent the style of that image. 

### Target Image Initialization
We start our target image as a copy of the content image, and keeps manipulate and update it so that its content is close to our content image and style close to our style image. 

### Content Representation:

The content representation of an image is the output from the second convolutional layer in the fourth stack(Conv 4_2).

We will compare the content representation of the target image with the content representation of the content image. We want these two representation to be close. 
To formalize this comparison, we define a content loss that calculates the difference between these two representations.

- Content Loss:

 - Mean squared difference between pixel values

 - ![Content Loss](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/Content%20Loss.png)

 - This measures how far away these two representations are from each other. As we try to create the best image, the goal is to minimize this loss. 

 - Here we are not using VGG19 as a classifier, we are using it as a feature extractor. We are not training CNN at all, we are using backpropagation to update the target image    until its content representation matches the content representation of our content image and its style representation matches the style representation of the style image. 

### Style Representation:

Looking at the correlation between features in individual layers of the VGG19 network, in other words, looking at how similar between the features in individual layers are. 
If the correlation between one feature map with others are high, it means this feature has been captured by lots of feature maps, it can be a shape, texture or color. This feature is thought of as part of that image's style. 

By including multiple layers of different sizes, we can obtain a multiscale representation of the input image, which captures both large and small style features. The correlation at each layer are given by a Gram Matrix. 

First we vectorize each feature map as a row: (8*4*4 feature maps in this layer)

![Vectorize](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/vectorization.png)

Multiply the vectorized feature maps with its transpose to get the gram matrix:

![Gram Matrix](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/gram%20matrix.png)

This final gram matrix will show the correlation between feature maps in one layer, and the dimension is only dependent on the number of feature maps in the convolutional layer instead of input image. 

### Style Loss:

Mean squared difference between the gram matrices of style image and the gram matrices of the target image. All five pairs(Conv 1_1 up to 5_1) are computed at each layer in the predefined list. 

![Style Loss](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/Style%20Loss.png)

*a* represents the number of values in each layer. *i* represents the layer, and *w* is the specified style weights that determines how much effect each layer's representation will have on our final image. We will only update the target image pixel values to minimize this loss after some number of iterations. 

### Total Loss:

![Total Loss](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/Total%20Loss.png)

Use typical backpropogation and optimization to reduce this total loss by updating the target image pixel values to match our desired content and styles. 

- Balance the weights between content loss and style loss:

 - These two losses are very different from each other, and we would like to take both into account fairly equally.
 
 -![Loss Weights](https://github.com/udAAbu/Style-Transfer/blob/master/note%20images/Loss%20weights.png)

 - We would like a appropriate ratio of a/b. If beta is too high, the images will be mostly styles without contents, while if the beta is too low, the images will be contents without much stylist effect. We need to change this weights for different style and content images. 

As the CNN goes deeper, the network cares about more the specific contents of the image rather than any detail about the texture and the color of pixels. Later layer of a network are sometimes referred to as a content representation of an image. 
