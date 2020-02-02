from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
_cur_fps = 0

# loop over some frames...this time using the threaded stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=480)
    frame = cv2.putText(frame, "raw FPS: {:3.1f}".format(_cur_fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
