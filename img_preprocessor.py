"""
Claire Liu, Yu-Jing Wei
-----
Image Preprocessing Pipeline for Deep Learning Inference.

This script standardizes raw images for MNIST-style digit recognition and 
Greek letter classification. It automatically scans input directories, 
applies class-specific transformations, and saves processed results 
to a 'processed' subdirectory.
"""

import os
import argparse
import cv2
import numpy as np


class ImagePreprocessor:
    """
    Utility class for orchestrating batch image transformations.
    Supports specialized pipelines for digit OCR and symbolic recognition.
    """
    def __init__(self):
        # Support common image extensions
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    def process_digit(self, img):
        """
        Transforms raw images into MNIST-compatible format (28x28, centered).
        Includes inversion, thresholding, distance transform, and smoothing.
        """
        # 1. Background-Foreground Inversion
        img = cv2.bitwise_not(img)

        # 2. Clean binary mask extraction
        _, img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # 3. Distance Transform to simulate handwriting stroke pressure
        dist = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
        img_grad = dist.astype(np.uint8)

        # 4. Dimension scaling (28x28) and anti-aliasing
        img_resized = cv2.resize(img_grad, (28, 28), interpolation=cv2.INTER_CUBIC)
        img_final = cv2.GaussianBlur(img_resized, (3, 3), 0)
        return img_final

    def process_greek(self, img):
        """
        Standardizes Greek letter images to 128x128 for Transfer Learning.
        """
        img_final = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        return img_final

    def run(self, mode, input_dir):
        """
        Scans the provided directory for images and executes the pipeline.
        """
        if not os.path.isdir(input_dir):
            print(f"Error: '{input_dir}' is not a valid directory.")
            return

        # Gather all valid image files in the directory
        image_files = [f for f in os.listdir(input_dir) 
                       if f.lower().endswith(self.valid_extensions)]
        
        if not image_files:
            print(f"No images found in: {input_dir}")
            return

        # Create 'processed' subdirectory inside the input directory
        target_dir = os.path.join(input_dir, 'processed')
        os.makedirs(target_dir, exist_ok=True)

        print(f"Starting [{mode.upper()}] mode processing in: {input_dir}")

        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            
            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                print(f"   - Skipping unreadable file: {filename}")
                continue

            # Execute logic based on mode
            if mode == 'digit':
                result = self.process_digit(img)
            else:
                result = self.process_greek(img)

            # Save to the processed folder
            output_path = os.path.join(target_dir, filename)
            cv2.imwrite(output_path, result)
            print(f"   - Processed: {filename}")

        print(f"Done! All images saved to: {target_dir}")


def main():
    """Main entry point to parse directory-based arguments."""
    parser = argparse.ArgumentParser(description="Batch Image Preprocessor for Digit and Greek Letters")
    parser.add_argument('--mode', type=str, required=True, choices=['digit', 'greek'], 
                        help="Mode: 'digit' (28x28) or 'greek' (128x128)")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the directory containing raw images")

    args = parser.parse_args()

    processor = ImagePreprocessor()
    processor.run(args.mode, args.input)


if __name__ == "__main__":
    main()
