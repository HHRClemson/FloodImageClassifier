# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 02:16:22 2021

@author: jpall
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
from pathlib import Path
import tensorflow as tf 
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import PIL
from PIL import Image
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
print(PIL.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def download_images():
    base_url = 'C:/Users/jpall/OneDrive/Desktop/8050/flood_images'
    #base_url = 'https://drive.google.com/drive/folders/1bJ4DYMMPCgliwdNhU-1dDKQ9_7Wp565l?usp=sharing'
    #base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['flood_75.jpg', 'flood_580.jpg', 'flood_536.jpg', 'flood_66.jpg','flood_162.jpg','flood_181.jpg', 'flood_172.jpg']
    image_paths = []
    for filename in filenames:
        print("at\n")
        #image_path = tf.keras.utils.get_file(fname=filename,
                                            #origin=base_url + filename,
                                            #untar=False)
        image_path = pathlib.Path(os.path.join(base_url, filename))
        image_paths.append(str(image_path))
    return image_paths

IMAGE_PATHS = download_images()
print(IMAGE_PATHS)


PATH_TO_SAVED_MODEL = "C:/Users/jpall/tensorflow/workspace/training_demo/exported-models/my_model_2/saved_model"
PATH_TO_LABELS = "C:/Users/jpall/tensorflow/workspace/training_demo/annotations/label_map.pbtxt"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL,tags=None)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    im =Image.open(path)
    (im_width, im_height) = im.size
    #print(im_width)
    #return np.array(im.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    return np.array(Image.open(path)),im_width,im_height

for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')
    img_path = Path(image_path)
    #print(img_path.exists())
    #print(os.path.basename(img_path).split(".")[0])
    image_name = os.path.basename(img_path).split(".")[0]
    image_np,image_np_width,image_np_height = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    #print(detections.keys())
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          groundtruth_box_visualization_color='white',
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    
    #print(detections['detection_boxes'])
    #print(detections['detection_anchor_indices'])
    #print(detections['detection_classes'])
    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh=.5
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name = category_index[detections['detection_classes'][i]]['name']
            print ("This box is gonna get used", boxes[i], detections['detection_classes'][i],class_name)
            ymin, xmin, ymax, xmax = boxes[i]
            x_up = int(xmin*image_np_width)
            y_up = int(ymin*image_np_height)
            x_down = int(xmax*image_np_width)
            y_down = int(ymax*image_np_height)
            arr = image_np[y_up:y_down,x_up:x_down]
            cv2.imwrite("C:/Users/jpall/OneDrive/Desktop/8050/models/training_demo/crops/box{}_{}.jpg".format(image_name,class_name),arr)
    plt.figure()
    plt.imshow(image_np_with_detections)
    #plt.imsave(image_name, image_np_with_detections)
    plt.figure(figsize=(90,75))
    plt.show()
    print('Done')
plt.show()
