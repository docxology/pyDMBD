#!/usr/bin/env python3
"""
Simple script to display DMBD result images using matplotlib
"""

import sys
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

def display_image(image_path):
    """Display an image using matplotlib"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return False
    
    try:
        # Read and display image
        img = imread(image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"Error displaying image: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_result.py <path_to_image>")
        print("Example: python view_result.py gb_improved_v2_outputs/dmbd_results_t15.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = display_image(image_path)
    sys.exit(0 if success else 1) 