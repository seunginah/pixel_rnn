import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.data_utils import get_CIFAR10_data

# Filter size
N_HIDDEN = 128 #512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# Number of epochs to train the net
NUM_EPOCHS = 50

SAVE_DIR = 'datasets/results/'

def load_data(small_data=False, verbose=False):
    """
    Output: A dictionary with the following keys and value shapes
    if small_data is True:
        X_train (100, 32, 32, 3)
        y_train (100, 32, 32, 3)
        X_val (10, 32, 32, 3)
        y_val (10, 32, 32, 3)
    else (small_data is False):
        X_train (49000, 32, 32, 3)
        y_train (49000, 32, 32, 3)
        X_val (1000, 32, 32, 3)
        y_val (1000, 32, 32, 3)
        X_test (1000, 32, 32, 3)
        y_test (1000, 32, 32, 3)

    where X is the image y with the center pixels zeroed out
    """
    print 'Loading data'
    data = get_CIFAR10_data(mode=1)
    if small_data:
        num_train = 100
        data = {
          'X_train': data['X_train'][:num_train],
          'y_train': data['y_train'][:num_train],
          'X_val': data['X_val'][:(num_train / 10)],
          'y_val': data['y_val'][:(num_train / 10)],
        }

    if verbose:
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape

    return data

def make_minibatch(X_train, y_train, batch_size):
    # Make a minibatch of training data
    num_train = X_train.shape[0]
    batch_mask = np.random.choice(num_train, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    return X_batch, y_batch

def save_image(img, filename):
    # Make sure num_channels is the last axis
    if img.shape[-1] != 3:
        img.transpose(0, 2, 3, 1)

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
        X.transpose(0, 2, 3, 1)

    num_images, H, W, num_channels = X.shape
    if num_images > 3:
        # Pick 3 images randomly
        i = np.random.randint(0, num_images, 3)
        X = X[i, :, :, :]

    print 'Saving images'
    for i, img in enumerate(X):
        filename = save_path + str(i)
        save_image(img, filename)

def test(X_test, y_test, predict, save_path=None):
    """
    Input: X_test (num_test, num_channels, H, W)
    """
    num_test, num_channels, H, W = X_test.shape
    
    # Start and end indices of the missing region
    start = int(H * 0.25)  # 8
    end = int(H * 0.75)    # 24
    missing_len = start - end
    
    # Make sure the center pixels of X_test are completely missing
    assert np.all(X_test[:, :, start:end, start:end] == 0), \
           'X_test missing region is not all zeros'

    # Predict missing pixels
    X_test[:, :, start:end, start:end] = predict(X_test)

    # Save predicted images
    if save_path is not None:
        save_images(X_test, save_path)

    # Compute RMSE
    rmse = np.sqrt(np.mean(np.square(X_test - y_test)))
    
    return rmse

def main(batch_size, seed, use_small_data, use_mse):
    # Set seed for reproducibility if provided
    if seed is not None:
        lasagne.random.set_rng(np.random.RandomState(1))

    # Load data
    data = load_data(small_data=use_small_data)
    X_train = data['X_train'].transpose(0, 3, 1, 2) # (num_train, num_channels, H, W)
    y_train = data['y_train'].transpose(0, 3, 1, 2) # (num_test, num_channels, H, W)
    print 'X_train.shape:', X_train.shape
    print 'y_train.shape:', y_train.shape

    # Images to check training progress on
    small_X = X_train[:2]
    small_y = y_train[:2]
    save_images(small_X, SAVE_DIR + 'train/missing')
    save_images(small_y, SAVE_DIR + 'train/ori')

    # Define network
    print 'Compiling network'
    num_train, num_channels, H, W = X_train.shape
    missing_start = int(H * 0.25)  # 8
    missing_end = int(H * 0.75)    # 24
    missing_len = missing_end - missing_start
    out_dim = 3 * np.square(missing_len)

    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, num_channels, H, W))

    # First hidden layer
    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in, num_filters=N_HIDDEN, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform())

    print lasagne.layers.get_output_shape(l_conv1)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2, 2))

    # Second hidden layer
    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1, num_filters=N_HIDDEN, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform())
    print lasagne.layers.get_output_shape(l_conv2)

    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2, 2))
    print lasagne.layers.get_output_shape(l_pool2)

    # Matrix of target pixels (batch_size, out_dim)
    target_output = T.imatrix('target_output')

    if use_mse:
        # Output layer
        l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_pool2, p=.5), num_units=out_dim,
            W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.rectify)
    
        # Get output from l_out (batch_size, out_dim)
        network_output = lasagne.layers.get_output(l_out)
    
        # Prediction
        prediction = network_output.reshape((-1, num_channels, missing_len, missing_len))
    
        # Compute mean-squared error loss
        loss = T.mean(np.square(network_output - target_output))
    else:
        # Output layer
        l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_pool2, p=.5), num_units=out_dim * 256,
            W = lasagne.init.HeUniform(), nonlinearity=None)

        # Get output from l_out (batch_size, out_dim * 256)
        network_output = lasagne.layers.get_output(l_out)

        # Softmax layer  (batch_size * 3 * missing_len * missing_len, 256)
        network_output = T.nnet.softmax(network_output.reshape((-1, 256)))

        # Prediction
        prediction = T.argmax(network_output, axis=1).reshape((-1, num_channels, missing_len, missing_len))

        # Compute categorical cross-entropy loss
        loss = T.mean(T.nnet.categorical_crossentropy(network_output, target_output.flatten().astype('int64')))
        
    # Compute gradients
    params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE)

    # Define Theano helper functions
    print 'Compiling functions'
    train = theano.function([l_in.input_var, target_output],
        loss, updates=updates, allow_input_downcast=True)
    predict = theano.function([l_in.input_var],
        prediction, allow_input_downcast=True)

    # Train network
    print 'Training'
    try:
        num_iters = num_train * NUM_EPOCHS / batch_size
        for itr in xrange(num_iters):
            X, y = make_minibatch(X_train, y_train, batch_size)
            y = y[:, :, missing_start:missing_end, missing_start:missing_end].reshape((batch_size, -1))
            new_loss = train(X, y)

            # Check training progress at the beginning of every epoch
            #if itr % (num_train / batch_size) == 0:
            if itr % 100 == 0:
                # Test current network on very small training data
                print 'Testing current network on very small training data'
                rmse = test(np.copy(small_X), np.copy(small_y), predict, SAVE_DIR + 'train/itr' + str(itr) + '_')
                
                # Print training progress (loss)
                print 'Epoch {} loss = {} rmse = {}'. \
                      format(itr * batch_size / num_train, new_loss, rmse)
                    
    except KeyboardInterrupt:
        pass

    # Test network
    #test(data['X_test'])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python main.py use_small_data'
        sys.exit()

    batch_size = 100
    seed = None
    use_small_data = bool(int(sys.argv[1]))
    use_mse = bool(int(sys.argv[2]))

    main(batch_size, seed, use_small_data, use_mse)
