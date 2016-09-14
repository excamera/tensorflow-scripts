#
# This script works only for PNG Files
# If you need JPG/JPEG images, the decode
# function has to be changed
#

import tensorflow as tf
import sys

image_path     = sys.argv[1]
filename_queue = tf.train.string_input_producer([image_path])
reader         = tf.WholeFileReader()
key, value     = reader.read(filename_queue)
my_img         = tf.image.decode_png(value)
init_op        = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1):
    image = my_img.eval()

  print(image.shape)
  Image.show(Image.fromarray(np.asarray(image)))

  coord.request_stop()
  coord.join(threads)
