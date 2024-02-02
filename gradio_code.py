import gradio as gr
import cv2
import numpy as np


# Gradio URL: https://981409f34ee40c2a3e.gradio.live

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(img_gray, kernel_size=(3, 3)):
    return cv2.GaussianBlur(img_gray, kernel_size, 0)

def edge_detection(img_blurred, low_threshold, high_threshold):
    return cv2.Canny(img_blurred, low_threshold, high_threshold)

def morphological_dilate(edges, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(edges, kernel, iterations=iterations)

def auto_find_seed_point(image):
    top_half = image[:int(image.shape[0] * 0.5), :]
    histogram = cv2.calcHist([top_half], [0], None, [256], [0, 256])
    max_intensity = np.argmax(histogram)
    rows, cols = np.where(top_half == max_intensity)
    if len(rows) > 0 and len(cols) > 0:
        return (rows[0], cols[0])
    else:
        return None

def region_growing(img, seed, region_threshold):
    # Parameters for region growing, up, down, left, right
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # To hold the segmented region
    segmented_img = np.zeros_like(img, dtype=np.uint8)
    # Stack for pixel traversal
    stack = [seed]

    # Seed intensity
    seed_intensity = int(img[seed[0], seed[1]])

    #until the stack is empty
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

def process_image(img):
    region_threshold = 20

    img_gray = convert_to_gray(img)
    blurred_image = apply_gaussian_blur(img_gray, kernel_size=(5, 5))
    edges = edge_detection(blurred_image, 20, 65)
    dilated_image = morphological_dilate(edges)

    seed_point = auto_find_seed_point(dilated_image)
    assert seed_point is not None, "Failed to find seed point for image."

    sky_mask = region_growing(dilated_image, seed_point, region_threshold)
    img[sky_mask != 0] = [0, 255, 0]


    return img,sky_mask

demo = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs=["image","image"],
    live=True,
)

demo.launch()
