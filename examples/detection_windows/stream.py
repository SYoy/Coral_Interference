from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imutils
import argparse
import time
import os
import cv2

from helper_fcts import load_labels, detect_objects, drawBoundingBox

import tensorflow as tf
from imutils.video import WebcamVideoStream
from imutils.video import FPS

os.environ["CAMERA_WIDTH"] = "640"
os.environ["CAMERA_HEIGHT"] = "480"

os.environ["NET_WIDTH"] = "300"
os.environ["NET_HEIGHT"] = "300"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', default=r"D:\Raspberry Pi\Coral_Interference\examples\detection_windows\models\detect.tflite")
    parser.add_argument(
        '--labels', help='File path of labels file.', default=r"D:\Raspberry Pi\Coral_Interference\examples\detection_windows\models\coco_labels.txt")
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.4)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = tf.lite.Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    _cur_fps = 0

    # loop over some frames...this time using the threaded stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=int(os.environ.get("CAMERA_WIDTH", default=640)), height=int(os.environ.get("CAMERA_HEIGHT", default=480)))
        start_time = time.monotonic()
        frame = frame[(int(int(os.environ.get("CAMERA_HEIGHT", default=480))/2) - int(input_height/2)):(int(int(os.environ.get("CAMERA_HEIGHT", default=480))/2) + int(input_height/2)),
                (int(int(os.environ.get("CAMERA_WIDTH", default=640))/2) - int(input_width/2)):(int(int(os.environ.get("CAMERA_WIDTH", default=640))/2) + int(input_width/2)), :]
        results = detect_objects(interpreter, frame, args.threshold)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        frame = drawBoundingBox(th=args.threshold, input_image=frame, result=results, labels=labels)

        frame = cv2.putText(frame, "raw FPS: {:3.1f}".format(_cur_fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255))
        frame = cv2.putText(frame, "Detection Time [ms]: {:3.2f}".format(elapsed_ms), (0, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255))
        cv2.imshow("Webcam Stream w/o Detections", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # update the FPS counter
        fps.update()
        _cur_fps = fps.fps()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()
