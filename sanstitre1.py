# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:51:23 2016

@author: xg02389
"""

# Let's load and process the dataset
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.transformers.image import MinMaxImageDimensions
from fuel.transformers import Cast


# Load the training set
train = DogsVsCats(('train',), subset=slice(0, 20000))

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream.default_stream constructor will turn our
# 8-bit images into floating-point decimals in [0, 1].
stream = DataStream.default_stream(
    train,
    iteration_scheme=ShuffledScheme(train.num_examples, 32)
)

scaled_stream = MinMaxImageDimensions(stream, 128, which_sources=('image_features',))

# Our images are of different sizes, so we'll use a Fuel transformer
# to take random crops of size (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    scaled_stream, (128, 128), which_sources=('image_features',))

float32_stream = Cast(cropped_stream, dtype='float32', which_sources=('image_features',))

# Create the Theano MLP
import theano
from theano import tensor
import numpy

rng = numpy.random.RandomState(23455)

X = tensor.tensor4('image_features')
T = tensor.lmatrix('targets')

w_shp = (8, 3, 4, 4)
w_bound = numpy.sqrt(3 * 4 * 4)
W = theano.shared( 
        numpy.asarray(
            rng.uniform(
            low=-1.0 / w_bound,
            high=1.0 / w_bound,
            size=w_shp),
        dtype=X.dtype), name ='W')
b_shp = (8,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=X.dtype), name ='b')


V = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(8000000, 2)), 'V')
c = theano.shared(numpy.zeros(2))
params = [W, b, V, c]

conv_out = tensor.nnet.conv.conv2d(X, W)
H = tensor.nnet.relu(conv_out + b.dimshuffle('x', 0, 'x', 'x'))


Y = tensor.nnet.softmax(tensor.dot(H.reshape((8000000,32)), V) + c)

loss = tensor.nnet.categorical_crossentropy(Y, T.flatten()).mean()

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

algorithm = GradientDescent(cost=loss, parameters=params,
                            step_rule=Scale(learning_rate=0.1))

# We want to monitor the cost as we train
loss.name = 'loss'
extensions = [TrainingDataMonitoring([loss], every_n_batches=1),
              Printing(every_n_batches=1)]

main_loop = MainLoop(data_stream=cropped_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()