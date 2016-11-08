% Tweaking the AlexNet
% Amit Sahu; Ayushman Dash;
  John Gamboa; Vitor Rey
% November 8, 2016

# Introduction

The [AlexNet] [@krizhevsky2012imagenet] is a prominent CNN model that known to
have produced the best results on the ImageNet Large Scale Visual Recognition
Challenge in 2012, with an improvement of around 10% on the error rate in
comparison to the second best model (see the [LSVRC-2012 results] for more
information).

Figure **??** shows an overview of the model. The first convolutional layer with
is composed by 96 filter maps of size $11 \times 11 \times 3$. The AlexNet uses
a stride $4$. This followed by a contrast normalization layer and a pooling layer
of size $3 \times 3$, with a stride of $2 \times 2$.

The second convolutional layer is composed of 48 convolutional kernels of size
$5 \times 5$, again followed by a contrast normalization layer and a pooling
layer with the same parameters as the first one.

A third convolutional layer follows, composed by 384 filter maps
of size $3 \times 3 \times 256$, followed by another one composed by 384 kernels
of size $3 \times 3 \times 192$. Finally, one last convolutional layer consisting
of 256 kernels of size $3 \times 3 \times 192$ completes the convolutional
part of the network.

On top of the aforementioned structure, 2 fully connected layers with 4096
neurons each are followed by the final 1000-neurons output layer. The network is
trained with backpropagation, and a dropout of 0.5 is used in the two last
4096 neurons fully connected layer. All layers use ReLU units.

[LSVRC-2012 results]: http://image-net.org/challenges/LSVRC/2012/results.html

# Tweaks

As a baseline, a model similar to the original AlexNet was trained. To facilitate
visualiations, we use 100 filter maps in the first and second layers instead of
the original 96 filter maps of the first layer and 48 of the second layer.
Additionally, in the fourth layer, we use 384 kernels of size $3 \times 3 \times
384$, and in the fifth layer we use 256 kernels of size $3 \times 3 \times 384$.
In what follows, we call this model the "original-AlexNet" (despite its
misleading name).

We wanted to compare speed of convergence of the original AlexNet with that of
an AlexNet using $tanh$ units. In what follows, we call this model the
"$tanh$-AlexNet".

Additionally, we trained a much smaller model to investigate how better the
results of AlexNet would be compared to this cheapily trainable model. In what
follows, we call it "small-AlexNet". (describe its exact size).

Figure **??** shows the evolution of the loss of the original AlexNet.

[AlexNet]: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

# Discussion

The tahn was worse...

The small was not that terrible...

# References

