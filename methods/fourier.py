import numpy as np
import matplotlib.pyplot as plt
from .metrics import METRICS


def fourier_similarity(img1:np.ndarray, img2:np.ndarray, metric='ssim'):
    magnitude_spectrum1 = fourier_spectrum(img1)
    magnitude_spectrum2 = fourier_spectrum(img2)
    print(magnitude_spectrum1.dtype)
    similarity = METRICS[metric](magnitude_spectrum1, magnitude_spectrum2)
    return similarity


def fourier_spectrum(img:np.ndarray):
    f_transform = np.fft.fft2(img)
    f_spectrum = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_spectrum)
    return magnitude_spectrum


def visualize(img:np.ndarray, spectrum:np.ndarray):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(np.log(spectrum + 1), cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
