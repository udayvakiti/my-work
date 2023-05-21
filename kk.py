# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:24:02 2023

@author: udaykiranreddyvakiti
"""
import numpy as np
import cv2
import pytesseract
import os
import sys
from pdf2image import convert_from_path

def process_image_pdf(image_path):
    # Read the image/pdf
    if image_path.endswith(".pdf"):
        pages = convert_from_path(image_path)
        for page in pages:
            # Convert the PDF page to a numpy array
            image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            # Pre-process the image for OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            # Run OCR on the image
            text = pytesseract.image_to_string(gray)
            print("OCR Output:")
            print(text)
    else:
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to read image")
            sys.exit()
        # Pre-process the image for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        # Run OCR on the image
        text = pytesseract.image_to_string(gray)
        print("OCR Output:")
        print(text)

if __name__ == "__main__":
    # List of image/pdf paths
    image_pdf_paths = [
       r"C:\Users\udaykiranreddyvakiti\Task1\1.jpg", r"C:\Users\udaykiranreddyvakiti\Task1\2.jpg", r"C:\Users\udaykiranreddyvakiti\Task1\3.jpg",r"C:\Users\udaykiranreddyvakiti\Task1\table.pdf"
    ]
    for image_pdf_path in image_pdf_paths:
        print("Processing:", image_pdf_path)
        process_image_pdf(image_pdf_path)
        print("=" * 50)