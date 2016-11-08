"""Builds AlexNet.

Summary of available functions:
 # gen_model(images): for generating AlexNet
 # get_softmax_layer:  To initialize the softmax layer
 # gen_fully_connected_layer: To get a fully connected layer
 # gen_pool_layer: Get a max pooling layer
 # gen_norm_layer: get a normalization layer
 # gen_conv_layer: Generate a convolution layer

This code Generate AlexnNet and trains with CIFAR-10 dataset. The idea was to
train the network and play around with the hyperparameters and see restuls if the
for of losses, gradients, and image filters

A lot of this code has been inspired or taken from https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10.py
"""


import os
import re
import sys
import tarfile
import time, datetime
import tensorflow as tf
import numpy as np
import math
from six.moves import urllib
from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.models.image import cifar10 as cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train/',
	"""Directory where to write event logs """
	"""and checkpoint.""")

tf.app.flags.DEFINE_string('activation', 'relu',
	"""Directory where to write event logs """
	"""and checkpoint.""")

tf.app.flags.DEFINE_string('eval_dir', './results/',
	"""Directory where to write event logs """
	"""and checkpoint.""")

tf.app.flags.DEFINE_float('learning_rate', 0.001,
	"The rate with which the weights are reduced during optimization.")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
	"""Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_examples', 10000,
	"""Number of examples to run.""")

tf.app.flags.DEFINE_integer('model', 5,
	"An integer to create a model architecture. defaults to 1 which is a small "
	"alexnet.")

tf.app.flags.DEFINE_integer('steps_per_checkpoint', 1000,
	"""Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
	"""Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('train', False,
	"""Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('eval', False,
	"""Whether to log device placement.""")

#FLAGS.batch_size = 100
FLAGS.data_dir = 'data/'
print(os.path.dirname(os.path.realpath(__file__)))
FLAGS.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),FLAGS.data_dir)
#FLAGS.batch_size = 100
#FLAGS.use_fp16 = False

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.01  # Learning rate decay factor.
INITIAL_LEARNING_RATE = FLAGS.learning_rate  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


# In[3]:

def maybe_download_and_extract():
	"""Download and extract the tarball from Alex's website."""
	dest_directory = FLAGS.data_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
															 float(
																 count *
																 block_size) /
															 float(
																 total_size) *
															 100.0))
			sys.stdout.flush()

		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
			_progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size,
			'bytes.')
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)


# In[4]:

maybe_download_and_extract()


# In[5]:

def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
	  x: Tensor
	Returns:
	  nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _filter_image_summary(name, x, grid_size):
	grid_x = grid_y = int(math.sqrt(grid_size))   # to get a square grid for 64
	# conv1 features
	grid = put_kernels_on_grid (x, grid_y, grid_x)
	tf.image_summary(name+'/filters', grid, max_images=1)

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''
    Taken from https://gist.github.com/kukuruza/03731dc494603ceab0c5

    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable

	Returns:
	  Variable Tensor
	"""
	with tf.device('/gpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer,
			dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
	  name: name of the variable
	  shape: list of ints
	  stddev: standard deviation of a truncated Gaussian
	  wd: add L2Loss weight decay multiplied by this float. If None, weight
	      decay is not added for this Variable.

	Returns:
	  Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		#tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
		tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


# kernel and stride are 4D tensors.
def gen_conv_layer(prev, name, kernel, stride, filter_img=False):
	with tf.variable_scope(name) as scope:
		kernel_conv = _variable_with_weight_decay('weights',
			shape=kernel,
			stddev=5e-2,
			wd=0.0)
		conv = tf.nn.conv2d(prev, kernel_conv, stride, padding='SAME')
		biases = _variable_on_cpu('biases', [kernel[-1]],
			tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		if FLAGS.activation == 'sigmoid':
			conv = tf.nn.sigmoid(bias, name=scope.name)
		elif FLAGS.activation == 'relu':
			conv = tf.nn.relu(bias, name=scope.name)
		elif FLAGS.activation == 'tanh':
			conv = tf.nn.tanh(bias, name=scope.name)
		else:
			conv = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv)
		if filter_img:
			_filter_image_summary(name, kernel_conv, kernel[-1])

	return conv


def gen_norm_layer(prev, name, window_size=4):
	norm = tf.nn.lrn(prev, window_size, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
		name=name)
	_activation_summary(norm)
	return norm

# In[8]:

def gen_pool_layer(prev, name, ksize, strides, padding='SAME'):
	pool = tf.nn.max_pool(prev, ksize=ksize, strides=strides, padding=padding,
		name=name)
	_activation_summary(pool)
	return pool


# In[9]:

def gen_fully_connected_layer(prev, name, shape, do_reshape=False):
	with tf.variable_scope(name) as scope:
		if do_reshape:
			prev = tf.reshape(prev, [FLAGS.batch_size, -1])
			shape[0] = prev.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=shape,
			stddev=0.04, wd=0.004)

		biases = _variable_on_cpu('biases', [shape[-1]],
			tf.constant_initializer(0.1))
		fc = tf.nn.relu(tf.matmul(prev, weights) + biases, name=scope.name)
		_activation_summary(fc)
	return fc


# In[10]:

def get_softmax_layer(prev, name, size):
	with tf.variable_scope(name) as scope:
		weights = _variable_with_weight_decay('weights', size,
			stddev=1 / float(size[0]), wd=0.0)
		biases = _variable_on_cpu('biases', [size[-1]],
			tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(prev, weights), biases,
			name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear


# In[11]:

def gen_model_1(images):

	images = tf.image.resize_images(images, [244, 244],

		method=tf.image.ResizeMethod.BICUBIC)
	conv1 = gen_conv_layer(images, 'conv1', [11, 11, 3, 100], [1, 4, 4, 1],
		filter_img=True)
	norm1 = gen_norm_layer(conv1, 'norm1')
	pool1 = gen_pool_layer(norm1, 'pool1', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv2 = gen_conv_layer(pool1, 'conv2', [5, 5, 100, 256], [1, 1, 1, 1])
	norm2 = gen_norm_layer(conv2, 'norm2')
	pool2 = gen_pool_layer(norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 256, 384], [1, 1, 1, 1])
	conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 384], [1, 1, 1, 1])
	conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 384, 256], [1, 1, 1, 1])

	pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')
	fcn1 = gen_fully_connected_layer(pool3, 'fc1', [None, 4096], do_reshape=True)
	fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [4096, 4096], do_reshape=False)
	fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [4096, 1000], do_reshape=False)

	smx = get_softmax_layer(fcn3, 'softmax', [1000, 10])
	return smx

def gen_model_2(images):

	images = tf.image.resize_images(images, [244, 244],

		method=tf.image.ResizeMethod.BICUBIC)
	conv1 = gen_conv_layer(images, 'conv1', [11, 11, 3, 100], [1, 4, 4, 1],
		filter_img=True)
	norm1 = gen_norm_layer(conv1, 'norm1')
	pool1 = gen_pool_layer(norm1, 'pool1', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	'''
	conv2 = gen_conv_layer(pool1, 'conv2', [5, 5, 64, 192], [1, 1, 1, 1])
	norm2 = gen_norm_layer(conv2, 'norm2')
	pool2 = gen_pool_layer(norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')
	'''
	conv3 = gen_conv_layer(pool1, 'conv3', [3, 3, 100, 384], [1, 1, 1, 1])
	conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 384], [1, 1, 1, 1])
	conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 384, 256], [1, 1, 1, 1])

	pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')
	fcn1 = gen_fully_connected_layer(pool3, 'fc1', [None, 4096], do_reshape=True)
	fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [4096, 4096], do_reshape=False)
	fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [4096, 1000], do_reshape=False)

	smx = get_softmax_layer(fcn3, 'softmax', [1000, 10])
	return smx

def gen_model_3(images):

	images = tf.image.resize_images(images,  [244, 244],
		method=tf.image.ResizeMethod.BICUBIC)
	conv1 = gen_conv_layer(images, 'conv1', [11, 11, 3, 100], [1, 4, 4, 1],
		filter_img=True)
	norm1 = gen_norm_layer(conv1, 'norm1')
	pool1 = gen_pool_layer(norm1, 'pool1', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv2 = gen_conv_layer(pool1, 'conv2', [5, 5, 100, 256], [1, 1, 1, 1])
	norm2 = gen_norm_layer(conv2, 'norm2')
	pool2 = gen_pool_layer(norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	'''
	conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 256, 384], [1, 1, 1, 1])
	conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 384], [1, 1, 1, 1])
	conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 384, 256], [1, 1, 1, 1])

	pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')
	'''
	fcn1 = gen_fully_connected_layer(pool2, 'fc1', [None, 4096], do_reshape=True)
	fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [4096, 4096], do_reshape=False)
	fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [4096, 1000], do_reshape=False)

	smx = get_softmax_layer(fcn3, 'softmax', [1000, 10])
	return smx

def gen_model_4(images):
	images = tf.image.resize_images(images, [244, 244],
		method=tf.image.ResizeMethod.BICUBIC)
	conv1 = gen_conv_layer(images, 'conv1', [11, 11, 3, 100], [1, 4, 4, 1],
		filter_img=True)
	norm1 = gen_norm_layer(conv1, 'norm1')
	pool1 = gen_pool_layer(norm1, 'pool1', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv2 = gen_conv_layer(pool1, 'conv2', [5, 5, 100, 256], [1, 1, 1, 1])
	norm2 = gen_norm_layer(conv2, 'norm2')
	pool2 = gen_pool_layer(norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 256, 384], [1, 1, 1, 1])
	conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 384], [1, 1, 1, 1])
	conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 384, 256], [1, 1, 1, 1])

	pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	fcn1 = gen_fully_connected_layer(pool3, 'fc1', [None, 4096], do_reshape=True)
	'''
	fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [4096, 4096], do_reshape=False)
	'''
	fcn3 = gen_fully_connected_layer(fcn1, 'fc3', [4096, 1000], do_reshape=False)

	smx = get_softmax_layer(fcn3, 'softmax', [1000, 10])
	return smx

def gen_model_small(images):

	images = tf.image.resize_images(images, [244, 244],
		method=tf.image.ResizeMethod.BICUBIC)
	conv1 = gen_conv_layer(images, 'conv1', [11, 11, 3, 64], [1, 4, 4, 1],
		filter_img=True)
	norm1 = gen_norm_layer(conv1, 'norm1')
	pool1 = gen_pool_layer(norm1, 'pool1', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv2 = gen_conv_layer(pool1, 'conv2', [5, 5, 64, 192], [1, 1, 1, 1])
	norm2 = gen_norm_layer(conv2, 'norm2')
	pool2 = gen_pool_layer(norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')

	conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 192, 384], [1, 1, 1, 1])
	conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 256], [1, 1, 1, 1])
	conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 256, 256], [1, 1, 1, 1])

	pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
		padding='VALID')
	fcn1 = gen_fully_connected_layer(pool3, 'fc1', [None, 384], do_reshape=True)
	fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [384, 192], do_reshape=False)
	fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [192, 192], do_reshape=False)

	smx = get_softmax_layer(fcn3, 'softmax', [192, 10])
	return smx

def load_model(saver, sess, chkpnts_dir):
	ckpt = tf.train.get_checkpoint_state(chkpnts_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print("Loading previously trained model: {}".format(
			ckpt.model_checkpoint_path))
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print("Training with fresh parameters")
		sess.run(tf.initialize_all_variables())

def evaluate(dir):
	eval_data = FLAGS.data_dir
	images_eval, labels_eval = cifar10.cifar10.inputs(eval_data=eval_data)
	logits = None
	top_k_op = None
	if FLAGS.model == 1:
		logits_eval = gen_model_1(images_eval)
		top_k_op = tf.nn.in_top_k(logits_eval, labels_eval, 1)
	elif FLAGS.model == 2:
		logits_eval = gen_model_2(images_eval)
		top_k_op = tf.nn.in_top_k(logits_eval, labels_eval, 1)
	if FLAGS.model == 3:
		logits_eval = gen_model_3(images_eval)
		top_k_op = tf.nn.in_top_k(logits_eval, labels_eval, 1)
	if FLAGS.model == 4:
		logits_eval = gen_model_4(images_eval)
		top_k_op = tf.nn.in_top_k(logits_eval, labels_eval, 1)
	if FLAGS.model == 5:
		logits_eval = gen_model_small(images_eval)
		top_k_op = tf.nn.in_top_k(logits_eval, labels_eval, 1)

		# Create a saver.
	saver = tf.train.Saver(tf.all_variables())
	sess = tf.Session(config=tf.ConfigProto(
		log_device_placement=FLAGS.log_device_placement))
	checkpoints_folder = FLAGS.train_dir
	if not os.path.exists(checkpoints_folder):
		os.makedirs(checkpoints_folder)
	load_model(saver, sess, FLAGS.train_dir)

		# Start the queue runners.
	coord = tf.train.Coordinator()
	num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
	true_count = 0  # Counts the number of correct predictions.
	total_sample_count = num_iter * FLAGS.batch_size
	step = 0

	while step < num_iter and not coord.should_stop():

		predictions = sess.run([top_k_op])
		true_count += np.sum(predictions)
		if step % 100:
			step_precision = float(true_count) / float((step * FLAGS.batch_size))
			print('%i\t%s: precision @ 1 = %.3f' % (datetime.now(), step_precision, step))
		step += 1

	# Compute precision @ 1.
	precision = true_count / total_sample_count
	print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

	with open(dir) as myfile:
		myfile.write(str(datetime.now()) + "," + str(step) + "," + str(precision))

def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)

		# Get images and labels for CIFAR-10.
		images, labels = cifar10.cifar10.distorted_inputs()



		# Build a Graph that computes the logits predictions from the
		# inference model.


		# Calculate predictions.

		# Build a Graph that computes the logits predictions from the
		# inference model.
		#
		logits = None
		if FLAGS.model == 1:
			logits = gen_model_1(images)

		elif FLAGS.model == 2:
			logits = gen_model_2(images)

		if FLAGS.model == 3:
			logits = gen_model_3(images)

		if FLAGS.model == 4:
			logits = gen_model_4(images)

		if FLAGS.model == 5:
			logits = gen_model_small(images)

		# logits = cifar10.cifar10.inference(images)

		# Calculate loss.
		loss = cifar10.cifar10.loss(logits, labels)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = cifar10.cifar10.train(loss, global_step)

		# Create a saver.
		saver = tf.train.Saver(tf.all_variables())
		config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
		config.gpu_options.per_process_gpu_memory_fraction=1.0
		sess = tf.Session(config=config)
		summary_op = tf.merge_all_summaries()
		checkpoints_folder = FLAGS.train_dir
		if not os.path.exists(checkpoints_folder):
			os.makedirs(checkpoints_folder)
		load_model(saver, sess, FLAGS.train_dir)

		# Build the summary operation based on the TF collection of
		# Summaries.


		# Build an initialization operation to run below.
		#init = tf.initialize_all_variables()

		# Start running operations on the Graph.

		#sess.run(init)

		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)

		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		for step in range(FLAGS.max_steps):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time

			assert not np.isnan(
				loss_value), 'Model diverged with loss = NaN'

			if step % FLAGS.steps_per_checkpoint == 0:
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				format_str = (
					'%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
					'sec/batch)')
				print(format_str % (
					datetime.datetime.now(), step, loss_value,
					examples_per_sec, sec_per_batch))

			if step % FLAGS.steps_per_checkpoint == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			# Save the model checkpoint periodically.
			if step % FLAGS.steps_per_checkpoint == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir,
					'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)


# def prepare_data():


if __name__ == '__main__':
	real_eval_dir = os.path.join(FLAGS.eval_dir,FLAGS.train_dir)
	if not tf.gfile.Exists(FLAGS.eval_dir):
		os.makedirs(real_eval_dir)
	with open(real_eval_dir + FLAGS.train_dir + ".csv", "a") as myfile:
		myfile.write(str("timestamp,step,value"))
	if not tf.gfile.Exists(FLAGS.train_dir):
		os.makedirs(FLAGS.train_dir)
	if not tf.gfile.Exists(FLAGS.train_dir):
		os.makedirs(FLAGS.train_dir)
	if FLAGS.train:
		train()
	elif FLAGS.eval:
		evaluate(real_eval_dir + FLAGS.train_dir + ".csv")
