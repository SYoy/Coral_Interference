import re
import os
import cv2

import numpy as np


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def annotate_objects(annotator, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * int(os.environ.get("NET_WIDTH", default=640)))
    xmax = int(xmax * int(os.environ.get("NET_WIDTH", default=640)))
    ymin = int(ymin * int(os.environ.get("NET_HEIGHT", default=480)))
    ymax = int(ymax * int(os.environ.get("NET_HEIGHT", default=480)))

    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box([xmin, ymin, xmax, ymax])
    annotator.text([xmin, ymin],
                   '%s\n%.2f' % (labels[obj['class_id']], obj['score']))


def drawBoundingBox(th, input_image, result, labels):
  for obj in result:
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * int(os.environ.get("NET_WIDTH", default=640)))
    xmax = int(xmax * int(os.environ.get("NET_WIDTH", default=640)))
    ymin = int(ymin * int(os.environ.get("NET_HEIGHT", default=480)))
    ymax = int(ymax * int(os.environ.get("NET_HEIGHT", default=480)))

    conf = obj['score']
    # print(conf)
    label = labels[obj['class_id']]
    if conf < th:
      continue

    cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 6)

    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)

    _x1 = xmin
    _y1 = ymin  # +int(labelSize[0][1]/2)
    _x2 = _x1 + labelSize[0][0]
    _y2 = _y1 - int(labelSize[0][1])

    cv2.rectangle(input_image, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(input_image, label, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

  return input_image