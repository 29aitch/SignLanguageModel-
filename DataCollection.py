import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# For hand cam window
offset = 20
# For white image
imgSize = 300

# file directory ( change the name to respective letter data)
folder = "Data/C"

# To count how many image is saved
count = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # This shows the only hand cam
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Multiply 255 so that the color value is 1 * 255 to becomes white
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # Crop the window so that it is in the middle
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        # Putting the hand window onto img white
        # imgCropShape = imgCrop.shape
        # Setting the height and width
        # white[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        # Setting the aspect Ratio so that the height/width don't exceed the window (will goes blank)
        aspRatio = h / w
        if aspRatio > 1:
            constant = imgSize / h
            # make sure it always goes higher
            wCal = math.ceil(constant * w)
            img_Resize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = img_Resize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            # Setting the height and width
            # white[0:imgResizeShape[0], 0:imgResizeShape[1]] = img_Resize
            imgWhite[:, wGap:wCal + wGap] = img_Resize

        else:
            constant = imgSize / w
            # make sure it always goes higher
            hCal = math.ceil(constant * h)
            img_Resize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = img_Resize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            # Setting the height and width
            imgWhite[hGap:hCal + hGap, :] = img_Resize

        cv2.imshow("ImageCrop:", imgCrop)
        cv2.imshow("ImageWhite:", imgWhite)

    # This shows the whole cam
    cv2.imshow("Image:", img)
    # cv2.waitKey(1)
    key = cv2.waitKey(1)

    # Press "s" to save
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)