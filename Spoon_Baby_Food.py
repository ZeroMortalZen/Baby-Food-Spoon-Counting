#!/usr/bin/env python
# coding: utf-8

# In[19]:


import zipfile
import os

# Replace 'your_folder.zip' with the actual name of your uploaded ZIP file
zip_file_path = 'C:/Users/kelly/babyfood.zip'
target_dir = 'NewBabyFood'  # Name of the destination folder

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

# Optionally, you can remove the ZIP file after extraction
# os.remove(zip_file_path)


# In[20]:


import os
print(os.getcwd()) # to get the current working directory


# In[21]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Directory path
IMAGE_DIR = 'C:/Users/kelly/NewBabyFood/babyfood'

# Define the lower and upper boundaries for the "red" in the HSV color space
RED_LOWER = np.array([0, 70, 50])
RED_UPPER = np.array([10, 255, 255])

# Average number of red pixels for one spoon.
AVERAGE_SPOON_PIXELS = 18000

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    return mask

def count_red_pixels(image):
    mask = preprocess_image(image)
    n_red_pixels = cv2.countNonZero(mask)
    return n_red_pixels

def estimate_spoons(n_red_pixels):
    n_spoons = round(n_red_pixels / AVERAGE_SPOON_PIXELS)
    return n_spoons

def main():
    for image_name in os.listdir(IMAGE_DIR):
        image = cv2.imread(os.path.join(IMAGE_DIR, image_name))
        n_red_pixels = count_red_pixels(image)
        n_spoons = estimate_spoons(n_red_pixels)

        if n_spoons == 1:
            label = "Pass"
            color = (0, 255, 0)  # Green color in BGR
        else:
            label = "Fail"
            color = (0, 0, 255)  # Red color in BGR

        position = (10, 30)
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        print(f'Image: {image_name}, Estimated Number of Spoons: {n_spoons}, Label: {label}')

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
    main()


# In[ ]:




