"""
Pixel RNN on MNIST
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.register_crash_notifier()
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import numpy
numpy.random.seed(123)
import random
random.seed(123)

import theano
import theano.tensor as T
import lib
import lasagne
import scipy.misc

import time
import functools
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.data_utils import get_CIFAR10_data

MODEL = 'pixel_rnn' # either pixel_rnn or pixel_cnn

# Dataset
HEIGHT = 32
WIDTH = 32
N_CHANNELS = 3
N_LABELS = 256

# Hyperparams
BATCH_SIZE = 100
DIM = 64 # Model dimensionality.
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Other constants
GEN_SAMPLES = True # whether to generate samples during training (generating samples takes WIDTH*HEIGHT*N_CHANNELS full passes through the net)
N_EPOCHS = 50

lib.utils.print_model_settings(locals().copy())

def relu(x):
    # Using T.nnet.relu gives me NaNs. No idea why.
    return T.switch(x > lib.floatX(0), x, lib.floatX(0))

def Conv2D(name, input_dim, output_dim, filter_size, inputs, mask_type=None, he_init=False):
    """
    inputs.shape: (batch size, height, width, input_dim)
    mask_type: None, 'a', 'b'
    output.shape: (batch size, height, width, output_dim)
    """
    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters_init = uniform(
        1./numpy.sqrt(input_dim * filter_size * filter_size),
        # output dim, input dim, height, width
        (output_dim, input_dim, filter_size, filter_size)
    )

    if he_init:
        filters_init *= lib.floatX(numpy.sqrt(2.))

    if mask_type is not None:
        filters_init *= lib.floatX(numpy.sqrt(2.))

    filters = lib.param(
        name+'.Filters',
        filters_init
    )

    if mask_type is not None:
        mask = numpy.ones(
            (output_dim, input_dim, filter_size, filter_size), 
            dtype=theano.config.floatX
        )
        center = filter_size//2
        for i in xrange(filter_size):
            for j in xrange(filter_size):
                    if (j > center) or (j==center and i > center):
                        mask[:, :, j, i] = 0.
        for i in xrange(N_CHANNELS):
            for j in xrange(N_CHANNELS):
                if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                    mask[
                        j::N_CHANNELS,
                        i::N_CHANNELS,
                        center,
                        center
                    ] = 0.

        filters = filters * mask

    # conv2d takes inputs as (batch size, input channels, height, width)
    inputs = inputs.dimshuffle(0, 3, 1, 2)
    result = T.nnet.conv2d(inputs, filters, border_mode='half', filter_flip=False)

    biases = lib.param(
        name+'.Biases',
        numpy.zeros(output_dim, dtype=theano.config.floatX)
    )
    result = result + biases[None, :, None, None]

    return result.dimshuffle(0, 2, 3, 1)

def Conv1D(name, input_dim, output_dim, filter_size, inputs, apply_biases=True):
    """
    inputs.shape: (batch size, height, input_dim)
    output.shape: (batch size, height, output_dim)
    * performs valid convs
    """
    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters = lib.param(
        name+'.Filters',
        uniform(
            1./numpy.sqrt(input_dim * filter_size),
            # output dim, input dim, height, width
            (output_dim, input_dim, filter_size, 1)
        )
    )

    # conv2d takes inputs as (batch size, input channels, height[?], width[?])
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1, inputs.shape[2]))
    inputs = inputs.dimshuffle(0, 3, 1, 2)
    result = T.nnet.conv2d(inputs, filters, border_mode='valid', filter_flip=False)

    if apply_biases:
        biases = lib.param(
            name+'.Biases',
            numpy.zeros(output_dim, dtype=theano.config.floatX)
        )
        result = result + biases[None, :, None, None]

    result = result.dimshuffle(0, 2, 3, 1)
    return result.reshape((result.shape[0], result.shape[1], result.shape[3]))

def Skew(inputs):
    """
    input.shape: (batch size, HEIGHT, WIDTH, dim)
    """
    buffer = T.zeros(
        (inputs.shape[0], inputs.shape[1], 2*inputs.shape[2] - 1, inputs.shape[3]),
        theano.config.floatX
    )

    for i in xrange(HEIGHT):
        buffer = T.inc_subtensor(buffer[:, i, i:i+WIDTH, :], inputs[:,i,:,:])

    return buffer

def Unskew(padded):
    """
    input.shape: (batch size, HEIGHT, 2*WIDTH - 1, dim)
    """
    return T.stack([padded[:, i, i:i+WIDTH, :] for i in xrange(HEIGHT)], axis=1)

def DiagonalLSTM(name, input_dim, inputs):
    """
    inputs.shape: (batch size, height, width, input_dim)
    outputs.shape: (batch size, height, width, DIM)
    """
    inputs = Skew(inputs)

    input_to_state = Conv2D(name+'.InputToState', input_dim, 4*DIM, 1, inputs, mask_type='b')

    batch_size = inputs.shape[0]

    c0_unbatched = lib.param(
        name + '.c0',
        numpy.zeros((HEIGHT, DIM), dtype=theano.config.floatX)
    )
    c0 = T.alloc(c0_unbatched, batch_size, HEIGHT, DIM)

    h0_unbatched = lib.param(
        name + '.h0',
        numpy.zeros((HEIGHT, DIM), dtype=theano.config.floatX)
    )
    h0 = T.alloc(h0_unbatched, batch_size, HEIGHT, DIM)

    def step_fn(current_input_to_state, prev_c, prev_h):
        # all args have shape (batch size, height, DIM)

        # TODO consider learning this padding
        prev_h = T.concatenate([
            T.zeros((batch_size, 1, DIM), theano.config.floatX), 
            prev_h
        ], axis=1)
        state_to_state = Conv1D(name+'.StateToState', DIM, 4*DIM, 2, prev_h, apply_biases=False)

        gates = current_input_to_state + state_to_state

        o_f_i = T.nnet.sigmoid(gates[:,:,:3*DIM])
        o = o_f_i[:,:,0*DIM:1*DIM]
        f = o_f_i[:,:,1*DIM:2*DIM]
        i = o_f_i[:,:,2*DIM:3*DIM]
        g = T.tanh(gates[:,:,3*DIM:4*DIM])

        new_c = (f * prev_c) + (i * g)
        new_h = o * T.tanh(new_c)

        return (new_c, new_h)

    outputs, _ = theano.scan(
        step_fn,
        sequences=input_to_state.dimshuffle(2,0,1,3),
        outputs_info=[c0, h0]
    )
    all_cs = outputs[0].dimshuffle(1,2,0,3)
    all_hs = outputs[1].dimshuffle(1,2,0,3)

    return Unskew(all_hs)

def DiagonalBiLSTM(name, input_dim, inputs):
    """
    inputs.shape: (batch size, height, width, input_dim)
    inputs.shape: (batch size, height, width, DIM)
    """
    forward = DiagonalLSTM(name+'.Forward', input_dim, inputs)
    backward = DiagonalLSTM(name+'.Backward', input_dim, inputs[:,:,::-1,:])[:,:,::-1,:]
    batch_size = inputs.shape[0]
    backward = T.concatenate([
        T.zeros([batch_size, 1, WIDTH, DIM], dtype=theano.config.floatX),
        backward[:, :-1, :, :]
    ], axis=1)

    return forward + backward

def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (numpy.random.uniform(size=images.shape) < images).astype('float32')

def save_image(img, filename):
    # Make sure num_channels is the last axis
    if img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)

    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')
    plt.savefig(filename)

def save_images(X, save_path):
    """
    Input:
    - X: array of images of shape (num_images, num_channels, H, W)
         or (num_images, H, W, num_channels)
    """
    
    # Make sure num_channels is the last axis
    if X.shape[-1] != 3:
        X = X.transpose(0, 2, 3, 1)

    print 'Saving images'
    for i, img in enumerate(X):
        filename = save_path + str(i)
        save_image(img, filename)

def generate_and_save_samples(X, y, save_path, mode='test'):
    samples = numpy.copy(X)
    missing_start = int(HEIGHT * 0.25)  # 8
    missing_end = int(HEIGHT * 0.75)    # 24

    for i in xrange(missing_start, missing_end):
        for j in xrange(missing_start, missing_end):
            for k in xrange(N_CHANNELS):
                next_sample = sample_fn(samples)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    # Save
    if X.shape[0] > 10:
        # Pick 10 images randomly
        i = numpy.random.randint(0, num_test, 10)
        save_images(samples[i, :, :, :], save_path)
        if mode == 'test':
            save_images(X[i, :, :, :], save_path + '_missing')
            save_images(y[i, :, :, :], save_path + '_ori')
    else:
        save_images(samples, save_path)
        if mode == 'test':
            save_images(X, save_path + 'missing')
            save_images(y, save_path + 'ori')

    # Compute RMSE
    rmse = numpy.sqrt(numpy.mean(numpy.square(samples - y)))

    return rmse

def make_minibatch(X_train, y_train, batch_size):
    # Make a minibatch of training data
    num_train = X_train.shape[0]
    batch_mask = numpy.random.choice(num_train, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    return X_batch, y_batch

if len(sys.argv) < 2:
    print 'Usage: python baseline.py use_small_data'

print 'Compiling network'

# inputs.shape: (batch size, height, width, channels)
inputs = T.tensor4('inputs')

output = Conv2D('InputConv', N_CHANNELS, DIM, 7, inputs, mask_type='a')

if MODEL=='pixel_rnn':

    output = DiagonalBiLSTM('LSTM1', DIM, output)
    output = DiagonalBiLSTM('LSTM2', DIM, output)

elif MODEL=='pixel_cnn':
    # The paper doesn't specify how many convs to use, so I picked 4 pretty
    # arbitrarily.
    for i in xrange(4):
        output = Conv2D('PixelCNNConv'+str(i), DIM, DIM, 3, output, mask_type='b', he_init=True)
        output = relu(output)

output = Conv2D('OutputConv1', DIM, DIM, 1, output, mask_type='b', he_init=True)
output = relu(output)

output = Conv2D('OutputConv2', DIM, DIM, 1, output, mask_type='b', he_init=True)
output = relu(output)

if N_CHANNELS > 1:
    # Softmax layer
    output = Conv2D('OutputConv3', DIM, N_LABELS * N_CHANNELS, 1, output, mask_type='b')
    output = output.reshape((-1, HEIGHT, WIDTH, N_LABELS, N_CHANNELS)) # (batch_size, 32, 32, 256, 3)
    output = output.dimshuffle(0, 1, 2, 4, 3) # (batch_size, 32, 32, 3, 256)
    output = T.nnet.softmax(output.reshape((-1, N_LABELS))) # (batch_size * 32 * 32 * 3, 256)
    
    # Loss
    cost = T.mean(T.nnet.categorical_crossentropy(output, inputs.flatten().astype('int64')))
    
    # Prediction
    prediction = T.argmax(output, axis=1).reshape((-1, HEIGHT, WIDTH, 3))    

else:
    # Sigmoid layer
    output = Conv2D('OutputConv3', DIM, 1, 1, output, mask_type='b')
    output = T.nnet.sigmoid(output)

    # Loss
    cost = T.mean(T.nnet.binary_crossentropy(output, inputs))

    # Prediction
    prediction = binarize(output)


params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.utils.print_params_info(params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=1e-3)

print 'Compiling functions'
train_fn = theano.function(
    [inputs],
    cost,
    updates=updates,
    on_unused_input='warn'
)

eval_fn = theano.function(
    [inputs],
    cost,
    on_unused_input='warn'
)

sample_fn = theano.function(
    [inputs],
    prediction,
    on_unused_input='warn'
)

print 'Loading data'
data = get_CIFAR10_data(mode=1)
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

X_train = data['X_train']
y_train = data['y_train']
small_X = X_train[:2]
small_y = y_train[:2]
X_test = data['X_test'][:100]
y_test = data['y_test'][:100]

use_small_data = bool(int(sys.argv[1]))
if use_small_data:
    X_train = X_train[:100]
    y_train = y_train[:100]
    print 'Using small data'

num_train = X_train.shape[0]

# Test on small train data while training to monitor progress
# and save results here
train_path = 'datasets/results/train/'
try:
    os.makedirs(train_path)
except OSError as exception:
    pass

print "Training!"
total_time = 0.
start_time = time.time()

num_iters = num_train * N_EPOCHS / BATCH_SIZE
for itr in xrange(num_iters):
    _, images = make_minibatch(X_train, y_train, BATCH_SIZE)
    new_cost = train_fn(images)

    # Print training progress every 10 iters    
    if itr % 10 == 0:
        epoch = itr * BATCH_SIZE / num_train
        total_time = time.time() - start_time
        print "epoch:{}\ttotal iters:{}\ttrain cost:{}\ttotal time:{}".format(
            epoch,
            itr,
            new_cost,
            total_time
        )
    
    # Test on small train data every 100 iters
    if itr % 100 == 0:
        tag = train_path + "itr{}".format(itr)
        if GEN_SAMPLES:
            generate_and_save_samples(small_X, small_y, tag, mode='train')
        lib.save_params('params_{}.pkl'.format(tag))

print 'Testing'
test_path = 'datasets/results/test/'
try:
    os.makedirs(train_path)
except OSError as exception:
    pass
rmse = generate_and_save_samples(X_test, y_test, test_path, mode='test')
