import cv2
import numpy as np
from skimage import feature, filters
from .metrics import bhattacharyya_distance, cosine_similarity
import matplotlib.pyplot as plt
from skimage import exposure


def grayscale_histogram(img1:np.ndarray, img2:np.ndarray):
    hist1 = hist_normalize(img1)
    hist2 = hist_normalize(img2)
    distance = bhattacharyya_distance(hist1, hist2)
    return distance


def RGB_histogram(img1:np.ndarray, img2:np.ndarray):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    distance = bhattacharyya_distance(hist1, hist2)
    return distance


def HSV_histogram(img1:np.ndarray, img2:np.ndarray):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [180, 256, 256], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [180, 256, 256], [0, 180, 0, 256, 0, 256])
    distance = bhattacharyya_distance(hist1, hist2)
    return distance


def hist_normalize(img:np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    normalized_hist = hist / hist.sum()
    return normalized_hist


def compute_hog(image, orientations=8, cell_size=16, cells_per_block=1, block_norm='L2-Hys'):
    hog_features = feature.hog(image, orientations=orientations, pixels_per_cell=(cell_size, cell_size),
                               cells_per_block=(cells_per_block, cells_per_block), block_norm=block_norm)
    return hog_features


def hog_similarity(img1:np.ndarray, img2:np.ndarray, visualise=False):
    hog_features1 = compute_hog(img1)
    hog_features2 = compute_hog(img2)

    hog_features1 = hog_features1.reshape(1, -1)
    hog_features2 = hog_features2.reshape(1, -1)
    if visualise:
        # Plot both images
        plt.figure(figsize=(8, 4))

        plt.subplot(2, 2, 1)
        plt.imshow(img1, cmap='gray')
        plt.title('Image 1')

        plt.subplot(2, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title('Image 2')

        # Plot HOG features for Image 1
        plt.subplot(2, 2, 3)
        hog_image1 = exposure.rescale_intensity(feature.hog(img1, visualize=True)[1], in_range=(0, 10))
        plt.imshow(hog_image1, cmap='gray')
        plt.title('HOG Image 1')

        # Plot HOG features for Image 2
        plt.subplot(2, 2, 4)
        hog_image2 = exposure.rescale_intensity(feature.hog(img2, visualize=True)[1], in_range=(0, 10))
        plt.imshow(hog_image2, cmap='gray')
        plt.title('HOG Image 2')

        plt.show()

    return cosine_similarity(hog_features1, hog_features2)

def compute_gradient_histograms(image):
    # Compute gradients
    gradient_x = filters.sobel_h(image)
    gradient_y = filters.sobel_v(image)

    # Compute gradient magnitudes
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    # Compute gradient orientations
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Ensure all angles are positive
    gradient_direction[gradient_direction < 0] += 360
    # Create histograms
    hist_magnitude, _ = np.histogram(gradient_magnitude, bins=256, range=[0, 256])
    hist_direction, _ = np.histogram(gradient_direction, bins=36, range=[0, 360])

    hist_magnitude = hist_magnitude / hist_magnitude.sum()
    hist_direction = hist_direction / hist_direction.sum()
    return hist_magnitude, hist_direction

def gradient_magnitude_and_direction(img1:np.ndarray, img2:np.ndarray, visualise=False):
    hist_magnitude1, hist_direction1 = compute_gradient_histograms(img1)
    hist_magnitude2, hist_direction2 = compute_gradient_histograms(img2)
    hist_magnitude1_flat = hist_magnitude1.reshape(1, -1)
    hist_magnitude2_flat = hist_magnitude2.reshape(1, -1)
    hist_direction1_flat = hist_direction1.reshape(1, -1)
    hist_direction2_flat = hist_direction2.reshape(1, -1)
    magnitude_distance = bhattacharyya_distance(hist_magnitude1_flat, hist_magnitude2_flat)
    direction_distance = bhattacharyya_distance(hist_direction1_flat, hist_direction2_flat)
    distance = (magnitude_distance + direction_distance) / 2
    if visualise:
        # Plot both images
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 4, 1)
        plt.imshow(img1, cmap='gray')
        plt.title('Image 1')

        plt.subplot(2, 4, 5)
        plt.imshow(img2, cmap='gray')
        plt.title('Image 2')

        # Plot histograms for Image 1
        plt.subplot(2, 4, 2)
        plt.bar(range(len(hist_magnitude1)), hist_magnitude1)
        plt.title('Magnitude Histogram 1')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')

        plt.subplot(2, 4, 6)
        plt.bar(range(len(hist_direction1)), hist_direction1)
        plt.title('Direction Histogram 1')
        plt.xlabel('Direction')
        plt.ylabel('Frequency')

        # Plot histograms for Image 2
        plt.subplot(2, 4, 3)
        plt.bar(range(len(hist_magnitude2)), hist_magnitude2)
        plt.title('Magnitude Histogram 2')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')

        plt.subplot(2, 4, 7)
        plt.bar(range(len(hist_direction2)), hist_direction2)
        plt.title('Direction Histogram 2')
        plt.xlabel('Direction')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    return distance


def compute_lbp_histogram(image):

    # Compute LBP features
    lbp = feature.local_binary_pattern(image, P=8, R=2, method='uniform')
    # Compute LBP histogram
    hist, _ = np.histogram(lbp, bins=16, range=[0, 10], density=True)
    hist = hist / hist.sum()
    return hist, lbp


def lbp_distance(img1, img2, visualise=False):
    # Reshape histograms to be 2D arrays
    hist1, lbp1 = compute_lbp_histogram(img1)
    hist2, lbp2 = compute_lbp_histogram(img2)

    hist1 = hist1.reshape(1, -1)
    hist2 = hist2.reshape(1, -1)

    distance = bhattacharyya_distance(hist1, hist2)
    if visualise:
        # Plot both images
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 3, 1)
        plt.imshow(img1, cmap='gray')
        plt.title('Image 1')

        plt.subplot(2, 3, 2)
        plt.imshow(lbp1, cmap='gray')
        plt.title('LBP Image 1')

        plt.subplot(2, 3, 3)
        plt.bar(range(len(hist1.flatten())), hist1.flatten())
        plt.title('LBP Histogram 1')
        plt.xlabel('Pattern')
        plt.ylabel('Frequency')

        plt.subplot(2, 3, 4)
        plt.imshow(img2, cmap='gray')
        plt.title('Image 2')

        plt.subplot(2, 3, 5)
        plt.imshow(lbp2, cmap='gray')
        plt.title('LBP Image 2')

        plt.subplot(2, 3, 6)
        plt.bar(range(len(hist2.flatten())), hist2.flatten())
        plt.title('LBP Histogram 2')
        plt.xlabel('Pattern')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    return distance
