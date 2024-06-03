import pickle
from skimage.measure import compare_ssim
import argparse
import imutils

import cv2
import sys
import numpy as np
newfile = 'test.txt'
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])


grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")


val1=format(score)
print("VALUE (1.0) IS ORIGINAL OR FORGERY, WHAT'S YOURS ?: {}", val1)



thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)



for c in cnts:

	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

with open(newfile, "ab") as fi:
  # dump your data into the file
  pickle.dump(val1, fi)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh ", thresh)
cv2.waitKey(0)
sys.stdout.close()
