# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:21:12 2021

@author: jpall
"""

import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib
matplotlib.use('TkAgg')
import PIL
import pathlib
import time
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
print(PIL.__version__)
from IPython.display import display

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def download_images():
    base_url = 'C:/Users/jpall/OneDrive/Desktop/8050/flood_images'
    #base_url = 'https://drive.google.com/drive/folders/1bJ4DYMMPCgliwdNhU-1dDKQ9_7Wp565l?usp=sharing'
    #base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['flood_75.jpg', 'flood_580.jpg','flood_536.jpg', 'flood_66.jpg','flood_162.jpg','flood_181.jpg', 'flood_172.jpg']
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


PATH_TO_SAVED_MODEL = "C:/Users/jpall/OneDrive/Desktop/8050/models/training_demo/inference_graph/saved_model"
PATH_TO_LABELS = "C:/Users/jpall/OneDrive/Desktop/8050/models/training_demo/training/labelmap.pbtxt"

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

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]
  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  need_detection_key = ['detection_classes','detection_boxes','detection_masks','detection_scores']
  output_dict = {key: output_dict[key][0, :num_detections].numpy()
               for key in need_detection_key}
  output_dict['num_detections'] = num_detections
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  # Handle models with masks:
  if 'detection_masks' in output_dict:
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          tf.convert_to_tensor(output_dict['detection_masks']), output_dict['detection_boxes'],
          image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure()
  plt.imshow(image_np)
  #plt.imsave(image_name, image_np_with_detections)
  plt.figure(figsize=(90,75))
  plt.show()
  print('Done')
  

for image_path in IMAGE_PATHS:
  show_inference(detect_fn, image_path)