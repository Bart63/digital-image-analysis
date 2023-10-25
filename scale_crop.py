import numpy as np
import cv2


def get_sizes(img1:np.ndarray, img2:np.ndarray):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    return h1, w1, h2, w2


def resize_width(img1, img2):
    _, w1, h2, w2 = get_sizes(img1, img2)
    new_h2 = int(h2 * w1/w2) # scale height of img2
    img2 = cv2.resize(img2, (new_h2, w1))
    return img1, img2


def crop_height(img1, img2):
    h1, _, h2, _ = get_sizes(img1, img2)
    center_y = h2//2
    top_y = center_y - h1//2
    bottom_y = top_y + h1
    img2 = img2[top_y:bottom_y, :]
    return img1, img2


def width_process(img1:np.ndarray, img2:np.ndarray):
    ## Get sizes
    _, w1, _, w2 = get_sizes(img1, img2)
    
    ## If the same width, skip
    if w1 == w2:
        return img1, img2
    
    ## Make image 2 wider than image 1
    is_img1_wider = w1 > w2
    if is_img1_wider:
        img1, img2 = img2, img1

    ## Resize
    img1, img2 = resize_width(img1, img2)
    ## Return to the previous order
    if is_img1_wider:
        img1, img2 = img2, img1

    ## Return images
    return img1, img2


def height_process(img1:np.ndarray, img2:np.ndarray):
    ## Get sizes
    h1, _, h2, _ = get_sizes(img1, img2)

    ## If the same height, skip
    if h1 == h2:
        return img1, img2
    
    is_img1_higher = h1 > h2
    if is_img1_higher:
        img1, img2 = img2, img1

    ## Crop height 
    img1, img2 = crop_height(img1, img2)

    ## Return to the previous order
    if is_img1_higher:
        img1, img2 = img2, img1
    return img1, img2


def scale_crop(img1:np.ndarray, img2:np.ndarray):
    """Scale width, crop height"""

    ## Transpose if the difference of heights is smaller than the difference of widths
    h1, w1, h2, w2 = get_sizes(img1, img2)
    transposed = abs(h1-h2) < abs(w1-w2)
    if transposed:
        img1, img2 = img1.swapaxes(1, 0), img2.swapaxes(1, 0)

    ## Scale width (axis where the difference is smallest)
    img1, img2 = width_process(img1, img2)

    ## Crop height
    img1, img2 = height_process(img1, img2)

    if transposed:
        img1, img2 = img1.swapaxes(1, 0), img2.swapaxes(1, 0)
    
    return img1, img2
