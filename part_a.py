
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_histogram(image, num_bins=256):
    # Compute the histogram of the input image
    histogram, _ = np.histogram(image, bins=num_bins, range=(0, num_bins))
    return histogram

def calculate_cumulative_histogram(histogram):
    # Compute the cumulative histogram
    cumulative_histogram = np.cumsum(histogram)
    return cumulative_histogram

def otsu_threshold(image):
    histogram = calculate_histogram(image)
    num_pixels = image.shape[0] * image.shape[1]

    # Calculate the probability of each intensity level
    probability = histogram.astype(np.float32) / num_pixels

    # Calculate cumulative probabilities
    cumulative_probability = calculate_cumulative_histogram(probability)

    # Calculate the cumulative means
    cumulative_means = np.cumsum(probability * np.arange(0, 256))

    # Calculate the global mean (mean of the entire image)
    global_mean = cumulative_means[-1]

    # Initialize variables for the maximum inter-class variance and threshold value
    max_variance = 0
    threshold = 0

    for t in range(256):
        # Calculate the class probabilities and means for both classes (background and foreground)
        prob_background = cumulative_probability[t]
        prob_foreground = 1.0 - prob_background

        if prob_background == 0 or prob_foreground == 0:
            continue

        mean_background = cumulative_means[t] / prob_background
        mean_foreground = (global_mean - cumulative_means[t]) / prob_foreground

        # Calculate the inter-class variance
        variance = prob_background * prob_foreground * (mean_background - mean_foreground)**2

        # Update the threshold and maximum variance if a higher variance is found
        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold

def apply_threshold(image, threshold):
    # Binarize the image based on the given threshold
    binary_image = (image > threshold).astype(np.uint8) * 255
    return binary_image

def apply_threshold_opencv(image, threshold):
    # Binarize the image using OpenCV's threshold function
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

if __name__ == "__main__":
    # Replace 'path_to_image' with the actual path to your image file
    image = cv2.imread('eggs.jpg', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = otsu_threshold(gray_image)
    binary_image_custom = apply_threshold(gray_image, threshold_value)
    binary_image_opencv = apply_threshold_opencv(gray_image, threshold_value)

    # Display the original, custom threshold, and OpenCV threshold images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary_image_custom, cmap='gray')
    plt.title('Custom Threshold')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image_opencv, cmap='gray')
    plt.title('OpenCV Threshold')
    plt.axis('off')

    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def connected_component_labeling(binary_image):
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=int)
    label_counter = 1

    def dfs(row, col, current_label):
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            labels[r, c] = current_label

            for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and binary_image[nr, nc] == 1 and labels[nr, nc] == 0:
                    stack.append((nr, nc))

    for r in range(height):
        for c in range(width):
            if binary_image[r, c] == 1 and labels[r, c] == 0:
                dfs(r, c, label_counter)
                label_counter += 1

    return labels

# Read binary image (0 for background, 1 for foreground) using OpenCV
image = cv2.imread('eggs.jpg', cv2.IMREAD_GRAYSCALE)
binary_image = (image > 128).astype(np.uint8)

# Perform Connected Component Labeling using custom implementation
labels_custom = connected_component_labeling(binary_image)

# Perform Connected Component Labeling using OpenCV
_, labels_opencv, _, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

# Display the images using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original Binary Image
axes[0].imshow(binary_image, cmap='gray')
axes[0].set_title('Original Binary Image')
axes[0].axis('off')

# Labeled Image - Custom Implementation
axes[1].imshow(labels_custom, cmap='jet')
axes[1].set_title('Connected Component Labeling (Custom)')
axes[1].axis('off')

# Labeled Image - OpenCV
axes[2].imshow(labels_opencv, cmap='jet')
axes[2].set_title('Connected Component Labeling (OpenCV)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def binary_erosion(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    padded_image = np.pad(image, ((k_height//2, k_height//2), (k_width//2, k_width//2)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for r in range(height):
        for c in range(width):
            if np.all(padded_image[r:r+k_height, c:c+k_width] * kernel):
                result[r, c] = 1

    return result

def binary_dilation(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    padded_image = np.pad(image, ((k_height//2, k_height//2), (k_width//2, k_width//2)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for r in range(height):
        for c in range(width):
            if np.any(padded_image[r:r+k_height, c:c+k_width] * kernel):
                result[r, c] = 1

    return result

# Read binary image (0 for background, 1 for foreground) using OpenCV
image = cv2.imread('eggs.jpg', cv2.IMREAD_GRAYSCALE)
binary_image = (image > 128).astype(np.uint8)

# Define the kernel for erosion and dilation (a 3x3 square kernel)
kernel = np.ones((3, 3), dtype=np.uint8)

# Perform binary image erosion
eroded_image = binary_erosion(binary_image, kernel)

# Perform binary image dilation
dilated_image = binary_dilation(binary_image, kernel)

# Display the images using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original Binary Image
axes[0].imshow(binary_image, cmap='gray')
axes[0].set_title('Original Binary Image')
axes[0].axis('off')

# Eroded Image
axes[1].imshow(eroded_image, cmap='gray')
axes[1].set_title('Eroded Image')
axes[1].axis('off')

# Dilated Image
axes[2].imshow(dilated_image, cmap='gray')
axes[2].set_title('Dilated Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()

