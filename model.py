import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import cv2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

from object_detection.utils import ops, label_map_util, visualization_utils

# MobileNet Object Detection
DETECTION_URL = "https://tfhub.dev/tensorflow/efficientdet/d3/1"
DETECTION_IMAGE_RES = 512
detection_model = hub.load(DETECTION_URL)
detection_labels_path = "mscoco_label_map.pbtxt"

# MobileNet Classification
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
CLASSIFIER_IMAGE_RES = 224
classifier_model = keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(CLASSIFIER_IMAGE_RES, CLASSIFIER_IMAGE_RES, 3))
])
classifier_labels_path = keras.utils.get_file("ImageNetLabels.txt","https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
imagenet_labels = np.array(open(classifier_labels_path).read().splitlines())

def isShark(cropped_shark_img):
  sharks = ["tiger shark", "hammerhead", "great white shark", "killer whale"]
  temp_result = classifier_model.predict(cropped_shark_img[np.newaxis, ...])
  predicted_class = np.argmax(temp_result[0], axis=-1)
  predicted_class_name = imagenet_labels[predicted_class]
  return predicted_class_name in sharks

# Video
video_path = "Shark on the Road.mp4"
video = cv2.VideoCapture(video_path)

while video.isOpened():
  ret, image = video.read()

  if not ret:
    break

  image_height, image_width = image.shape[:2]
  shark_img = Image.fromarray(image).resize((DETECTION_IMAGE_RES, DETECTION_IMAGE_RES))
    
  
  # Turn into numpy array
  shark_img = np.array(shark_img)

  # Predict the image, returns a vector of the probabilities
  detection_model_output = detection_model(shark_img[np.newaxis, ...])

  class_ids = detection_model_output["detection_classes"]


  num_detections = int(detection_model_output.pop("num_detections"))
  detection_model_output = {key:value[0, :num_detections].numpy() 
                  for key,value in detection_model_output.items()}
  detection_model_output['num_detections'] = num_detections
  detection_model_output["detection_classes"] = detection_model_output["detection_classes"].astype(np.int64)

  detection_shark_img = np.copy(shark_img)
  bbox_coords = []

  for i in range(num_detections):
    temp_bbox = detection_model_output["detection_boxes"][i]
    img_y, img_x, img_h, img_w = [int(i * DETECTION_IMAGE_RES) for i in temp_bbox]

    cropped_shark_img = Image.fromarray(shark_img[img_y:img_h,img_x:img_w]).resize((CLASSIFIER_IMAGE_RES, CLASSIFIER_IMAGE_RES))
    cropped_shark_img = np.array(cropped_shark_img) / 255.0
    if isShark(cropped_shark_img):

      # This is an additional id in mscoco_label_map.pbtxt:
      '''
      item {
        name: "m/0abcde"
        id: 91
        display_name: "shark"
      }
      '''
      detection_model_output["detection_classes"][i] = 91
      _ = visualization_utils.visualize_boxes_and_labels_on_image_array(
      detection_shark_img,
      np.array([detection_model_output["detection_boxes"][i]]),
      np.array([detection_model_output["detection_classes"][i]]),
      np.array([detection_model_output["detection_scores"][i]]),
      label_map_util.create_category_index_from_labelmap(detection_labels_path, use_display_name=True),
      instance_masks=detection_model_output.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=4,
      min_score_thresh=0.08
      )
  detection_shark_img = np.array(Image.fromarray(detection_shark_img).resize((image_width, image_height)))
  cv2.imshow("Shark on the Road", detection_shark_img)

  cv2.waitKey(1) & 0xff

cv2.destroyAllWindows()
video.release()
