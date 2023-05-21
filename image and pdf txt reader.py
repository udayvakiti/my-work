# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:02:04 2023

@author: udaykiranreddyvakiti
"""

import cv2
import pytesseract
import sys

# Load the image or PDF file
image = cv2.imread(r"C:\Users\udaykiranreddyvakiti\Task1\1.jpg")
# Check if the image was correctly read
if image is None:
    print("Failed to read image")
    sys.exit()

# Print the shape of the image
print(image.shape)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to binarize the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Run Py-tesseract to extract the text from the image
text = pytesseract.image_to_string(thresh)

# Print the extracted text
print(text)