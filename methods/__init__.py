from .fourier import fourier_similarity
from .metrics import mse_similarity, ssim_similarity, METRICS
from .statistical import stat_similarity
from .mahotas import haralick_similarity
from .histogram import hog_similarity, gradient_magnitude_and_direction, lbp_distance, \
                grayscale_histogram, RGB_histogram, HSV_histogram
from .img2img import gabor_similarity, law_similarity
