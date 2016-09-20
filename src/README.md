# Label Extraction
 Tensorflow has a tensor to decode JPEG images and run image classification on it. We are interested in doing the 
 same for PNG Images. I have customized the Image Classification (classify_image.py) under models/image.imagenet to
 work with PNG Images. 
 
 1. Convert the PNG Image to a numpy array with 3 channels (RGB)
 2. Change the tensor from DecodeJpeg/contents:0 to DecodeJpeg:0 so that the tensor can work on the numpy array
 3. Run the tensor and get the predictions
 4. Output the score
