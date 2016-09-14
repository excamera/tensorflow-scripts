from __future__ import print_function
from tensorflow.models.image.imagenet.classify_image import maybe_download_and_extract
from tensorflow.models.image.imagenet.classify_image import run_inference_on_image

import os
import sys
import uuid
import logging
import time
from random import randint
import subprocess as sp

# Set Logging
sys.path.append(".")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get Image
image_path = sys.argv[1]

# Run Inference on Image
(infer,times) = run_inference_on_image(image_path)

# Write to the output_file
output_file = sys.argv[2]
open("/tmp/" + output_file, "w+").write(infer)
