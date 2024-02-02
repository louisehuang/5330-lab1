import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')
def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(img_gray, kernel_size=(3, 3)):
    return cv2.GaussianBlur(img_gray, kernel_size, 0)

def edge_detection(img_blurred, low_threshold, high_threshold):
    return cv2.Canny(img_blurred, low_threshold, high_threshold)

def morphological_dilate(edges, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(edges, kernel, iterations=iterations)

def morphological_eroded(edges, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(edges, kernel, iterations=iterations)

def auto_find_seed_point(image):
    '''using histogram  to finds the max appeared intensity
    in the upper part of the image. This will be the sky intensity value'''
    top_half = image[:int(image.shape[0] * 0.5), :]

    histogram = cv2.calcHist([top_half], [0], None, [256], [0, 256])

    max_intensity = np.argmax(histogram)
    # Find the max_appeared intensity value of the top half of the image
    rows, cols = np.where(top_half == max_intensity)
    if len(rows) > 0 and len(cols) > 0:
        return (rows[0], cols[0])
    else:
        return None

def region_growing(img, seed, region_threshold):
    # Parameters for region growing, up, down, left, right
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    segmented_img = np.zeros_like(img, dtype=np.uint8)  # To hold the segmented region
    stack = [seed]  # Stack for pixel traversal

    # Seed intensity
    seed_intensity = int(img[seed[0], seed[1]])

    # Loop until the stack is exhausted
    while stack:
        px, py = stack.pop()

        for dx, dy in neighbors:
            nx, ny = px + dx, py + dy

            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                # Pixel difference
                diff = abs(int(img[nx, ny]) - seed_intensity)

                if diff < region_threshold and segmented_img[nx, ny] == 0:
                    segmented_img[nx, ny] = 255
                    stack.append((nx, ny))

    return segmented_img



def display_images(original, dilated, sky_mask):
    #Displays the original image, dilated image, and sky mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(dilated, cmap='gray')
    plt.title('Dilated')

    plt.subplot(1, 3, 3)
    plt.imshow(sky_mask, cmap='gray')
    plt.title('Sky Mask')

    plt.show()

def process_image(img_path):
    region_threshold = 50
    img = cv2.imread(img_path)

    assert img is not None, f"File '{img_path}' could not be read. Check if the file exists."

    img_gray = convert_to_gray(img)
    blurred_image = apply_gaussian_blur(img_gray, kernel_size=(5, 5))

    edges = edge_detection(blurred_image,20, 65)

    dilated_image = morphological_dilate(edges)

    # Find the seed point for region growing
    seed_point = auto_find_seed_point(dilated_image)

    assert seed_point is not None, f"Failed to find seed point for image: {img_path}"

    # Apply region growing with a higher region threshold
    sky_mask = region_growing(dilated_image, seed_point, region_threshold)

    # Color the sky area in green
    img[sky_mask != 0] = [0, 255, 0]

    # Display the images
    display_images(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dilated_image, sky_mask)

def main():
    folder_path = '/content/drive/MyDrive/Colab Notebooks'
    imgs = os.listdir(folder_path)

    for filename in imgs:
        img_path = os.path.join(folder_path, filename)
        process_image(img_path)

main()
