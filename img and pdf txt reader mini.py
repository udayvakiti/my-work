# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:23:02 2023

@author: udaykiranreddyvakiti
"""

import cv2
import pytesseract
import sys
import os
import pdf2image
import numpy as np

# List of file names
files = [r"C:\Users\udaykiranreddyvakiti\Task1\1.jpg", r"C:\Users\udaykiranreddyvakiti\Task1\2.jpg", r"C:\Users\udaykiranreddyvakiti\Task1\3.jpg",r"C:\Users\udaykiranreddyvakiti\Task1\table.pdf" ]

# Loop through each file
for file in files:
    # Load the image or PDF file
    image = cv2.imread(file)

    # Check if the file was correctly read
    if image is None:
        # If the file is a PDF, extract the images
        if file.endswith('.pdf'):
            # Use pdf2image to extract the images from the PDF
            images = pdf2image.convert_from_path(file)
            # Loop through each image
            for image in images:
                # Convert the image to grayscale
             # Convert the PIL image to an OpenCV image
             opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

# Convert the OpenCV image to grayscale
             gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to binarize the image
             thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Run Py-tesseract to extract the text from the image
             text = pytesseract.image_to_string(thresh)

# Print the extracted text
             print(text)
        else:
            print(f"Failed to read {file}")
            sys.exit()
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's thresholding to binarize the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Run Py-tesseract to extract the text from the image
        text = pytesseract.image_to_string(thresh)
        # Print the extracted text
        print(text)