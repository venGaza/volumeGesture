from cv2 import cv2
import time
import numpy as np
import handTrackingModule as htm
from math import hypot
from subprocess import call

# TODO: 
#   - Responsive volume meter
#   - Windows support
#   - Add prepatory/end gesture

CAM_WIDTH, CAM_HEIGHT = 640, 480

record = False
if record:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
previousTime = 0
detector = htm.HandDetector(detectionCon=0.80)

while True:
    success, img = cap.read()

    if success:
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img, draw=False)
        if landmarkList:
            _, thumb_x, thumb_y = landmarkList[4]
            _, index_x, index_y = landmarkList[8]

            center_x, center_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2

            # Add shapes to key features (thumb and index)
            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (center_x, center_y), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)

            distance = int(hypot(index_x-thumb_x, index_y-thumb_y))

            if distance < 50:
                cv2.circle(img, (center_x, center_y), 10, (255, 255, 255), cv2.FILLED)
            
            minVal, maxVal = 10, 250
            volume = int(np.interp(distance, [10,150], [0,100]))

            if volume < 10:
                volume = 0
            if volume > 90:
                volume = 100
            
            # macOS system call to adjust the volume
            call([f"osascript -e 'set volume output volume {volume}'"], shell=True)
            
            # Volume meter
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            volume_value = int(np.interp(volume, (0, 100), (400, 150)))
            cv2.rectangle(img, (50, volume_value), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{volume}%', (20, 450), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0), 1)

    # Output FPS
    currentTime = time.time()
    fps = int(1 / (currentTime-previousTime))
    previousTime = currentTime
    cv2.putText(img, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_PLAIN,
                1, (255, 0, 0), 1)
    
    # Output image
    if record: 
        out.write(img)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

if record: 
        out.release()
cap.release()
cv2.destroyAllWindows()