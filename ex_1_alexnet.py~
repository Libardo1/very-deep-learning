
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import gzip
import os
import re
import sys
import tarfile
import time

import tensorflow as tf

from six.moves import urllib

from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.models.image import cifar10 as cifar10




# In[2]:

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
"""Whether to log device placement.""")


#tf.app.flags.DEFINE_integer('batch_size', 128,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")
#tf.app.flags.DEFINE_boolean('use_fp16', False,
#                            """Train the model using fp16.""")




# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

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
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
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

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
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
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


# In[6]:

# kernel and stride are 4D tensors.
def gen_conv_layer(prev, name, kernel, stride):
    with tf.variable_scope(name) as scope:
        kernel_conv = _variable_with_weight_decay('weights',
                                         shape=kernel,
                                         stddev=5e-2,
                                         wd=0.0)
        conv = tf.nn.conv2d(prev, kernel_conv, stride, padding='SAME')
        biases = _variable_on_cpu('biases', [ kernel[-1]], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv)
    return conv


# In[7]:

def gen_norm_layer(prev, name, window_size = 4):
    norm = tf.nn.lrn(prev, window_size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    _activation_summary(norm)
    return norm


# In[8]:

def gen_pool_layer(prev, name, ksize, strides, padding = 'SAME'):
    pool = tf.nn.max_pool(prev, ksize=ksize, strides=strides, padding=padding, name=name)
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
        
        biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(prev, weights) + biases, name=scope.name)
        _activation_summary(fc)
    return fc


# In[10]:

def get_softmax_layer(prev, name, size):
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', size,
                                              stddev=1/float(size[0]), wd=0.0)
        biases = _variable_on_cpu('biases', [size[-1]],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(prev, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


# In[11]:

def gen_model(images):
    images = tf.image.resize_images(images,[244,244],method=tf.image.ResizeMethod.BICUBIC)
    conv1 = gen_conv_layer(images, 'conv1', [11, 11, 3, 64], [1, 4, 4, 1])
    norm1 = gen_norm_layer(conv1, 'norm1')
    pool1 = gen_pool_layer(norm1, 'pool1', [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
    
    conv2 = gen_conv_layer(pool1, 'conv2', [5, 5, 64, 192], [1, 1, 1, 1])
    norm2 = gen_norm_layer(conv2, 'norm2')
    pool2 = gen_pool_layer(norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
    
    conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 192, 384], [1, 1, 1, 1])
    conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 256], [1, 1, 1, 1])
    conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 256, 256], [1, 1, 1, 1])

    pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
    fcn1 = gen_fully_connected_layer(pool3, 'fc1', [None,384], do_reshape=True)
    fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [384,192], do_reshape=False)
    fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [192,192], do_reshape=False)
    
    smx = get_softmax_layer(fcn3, 'softmax', [192,10])
    return smx

# In[ ]:




# In[ ]:




# In[12]:


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #
    logits = gen_model(images) #cifar10.cifar10.inference(images)
    #logits = cifar10.cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


# In[ ]:


#cifar10.maybe_download_and_extract()
if tf.gfile.Exists('/tmp/cifar10_train'):
    tf.gfile.DeleteRecursively('/tmp/cifar10_train')  #FLAGS.train_dir)
    tf.gfile.MakeDirs('/tmp/cifar10_train')#FLAGS.train_dir)
    train()


# In[ ]:




# In[ ]:



