"""
Claire Liu, Yu-Jing Wei
-----
This file contains the ImagePreprocessor class, which provides methods
to preprocess images for digit and Greek letter recognition tasks.
The preprocessing steps include color inversion, binarization, distance
transformation, resizing, and blurring for digit images, while Greek
letter images are simply resized. The processed images are saved in a
'processed' subdirectory within the original image's directory.
"""
import cv2
import os
import glob
import numpy as np
import argparse


class ImagePreprocessor:
    """Helper class to preprocess images for digit and Greek letter
    recognition tasks."""
    def __init__(self):
        # No global output folder needed; it will be determined per image
        pass

    def process_digit(self, img):
        """
        Process images for MNIST format: 28x28, black background, 
        white text with a realistic handwritten gradient.
        """
        # 1. Invert colors (assuming input is black text on white paper)
        # MNIST models expect a black background (0) and white stroke (255)
        img = cv2.bitwise_not(img)

        # 2. Binarize to extract the clean shape of the digit
        _, img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # 3. Distance Transform: center of the stroke becomes bright, edges become dark
        # This simulates the pressure of a pen/marker
        dist = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
        img_grad = dist.astype(np.uint8)

        # 4. Resize to 28x28 and apply Blur to soften edges (mimicking MNIST dataset)
        img_resized = cv2.resize(img_grad, (28, 28), interpolation=cv2.INTER_CUBIC)
        img_final = cv2.GaussianBlur(img_resized, (3, 3), 0)
        return img_final

    def process_greek(self, img):
        """
        Process images for Greek dataset: Simply resize to 128x128.
        The actual inversion and cropping will be handled by GreekTransform in the model script.
        """
        img_final = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        return img_final

    def run(self, mode, input_pattern):
        """Run the preprocessing pipeline based on the specified mode
        and input pattern."""
        # Get all file paths matching the input pattern
        image_files = glob.glob(input_pattern)
        if not image_files:
            print(f"No files found for pattern: {input_pattern}")
            return

        for img_path in image_files:
            # Read image in grayscale mode
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                print(f"Skipping: Could not read {img_path}")
                continue

            # --- Dynamic Directory Logic ---
            # 1. Get the parent directory of the current image
            original_dir = os.path.dirname(img_path)
            # 2. Create a 'processed' folder inside that parent directory
            target_dir = os.path.join(original_dir, 'processed')
            os.makedirs(target_dir, exist_ok=True)

            # --- Execute Processing Based on Mode ---
            if mode == 'digit':
                result = self.process_digit(img)
            else: # mode == 'greek'
                result = self.process_greek(img)

            # --- Save File ---
            base_name = os.path.basename(img_path)
            output_path = os.path.join(target_dir, base_name)
            cv2.imwrite(output_path, result)
            print(f"[{mode.upper()}] Saved to: {output_path}")


def main():
    """Main function to parse command line arguments and
    run the image preprocessor."""
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Image Preprocessing Tool for Digit and Greek Letter Recognition")
    parser.add_argument('--mode', type=str, required=True, choices=['digit', 'greek'], 
                        help="Mode: 'digit' for 28x28 MNIST style, 'greek' for 128x128")
    parser.add_argument('--input', type=str, required=True,
                        help="Input path pattern (e.g., './data/test/*.png')")

    args = parser.parse_args()

    # Initialize and run the processor
    processor = ImagePreprocessor()
    processor.run(args.mode, args.input)


if __name__ == "__main__":
    main()
