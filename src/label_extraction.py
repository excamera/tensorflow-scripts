from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tarfile
import os
import sys
import uuid
import logging
import time
from random import randint
import subprocess as sp

import numpy as np
from six.moves import urllib
import tensorflow as tf
import tensorflow.models.image.imagenet.classify_image as ic

FLAGS = tf.app.flags.FLAGS

###
#  Runs inference on an image.
#
#  Args:
#    image: Image file name.
#
#  Returns:
#    Nothing
###
def run_top_k_predictions_on_image(image):

  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  ic.create_graph()

  with tf.Session() as sess:
    # Convert the PNG image to a height x width x 3 (channels) Numpy array, 
    # for example using PIL, then feed the 'DecodeJpeg:0' tensor:
    image_content = Image.open(image)
    image_array = np.array(image_content)[:, :, 0:3]  # Select RGB channels only.

    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_array})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = ic.NodeLookup()
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))


# Set Logging
sys.path.append(".")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get Image
if len(sys.argv) >= 1:
    image_path = sys.argv[1]
else:
    print ("Pass the path to an Image file")
    sys.exit(1)

# Run Inference on Image
print (run_top_k_predictions_on_image(image_path))
