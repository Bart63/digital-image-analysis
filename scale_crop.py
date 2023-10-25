from PIL import Image


def get_sizes(img1:Image.Image, img2:Image.Image):
    w1, h1 = img1.size
    w2, h2 = img2.size
    return h1, w1, h2, w2


def resize_width(img1:Image.Image, img2:Image.Image):
    _, w1, h2, w2 = get_sizes(img1, img2)
    new_h2 = int(h2 * w1 / w2)
    img2 = img2.resize((w1, new_h2))
    return img1, img2


def crop_height(img1:Image.Image, img2:Image.Image):
    h1, _, h2, _ = get_sizes(img1, img2)
    center_y = h2 // 2
    top_y = center_y - h1 // 2
    bottom_y = top_y + h1
    img2 = img2.crop((0, top_y, img2.width, bottom_y))
    return img1, img2


def width_process(img1:Image.Image, img2:Image.Image):
    _, w1, _, w2 = get_sizes(img1, img2)

    if w1 == w2:
        return img1, img2

    is_img1_wider = w1 > w2
    if is_img1_wider:
        img1, img2 = img2, img1

    img1, img2 = resize_width(img1, img2)

    if is_img1_wider:
        img1, img2 = img2, img1
    return img1, img2


def height_process(img1:Image.Image, img2:Image.Image):
    h1, _, h2, _ = get_sizes(img1, img2)

    if h1 == h2:
        return img1, img2

    is_img1_higher = h1 > h2
    if is_img1_higher:
        img1, img2 = img2, img1

    img1, img2 = crop_height(img1, img2)

    if is_img1_higher:
        img1, img2 = img2, img1
    return img1, img2


def scale_crop(img1:Image.Image, img2:Image.Image):
    h1, w1, h2, w2 = get_sizes(img1, img2)
    transposed = abs(h1 - h2) < abs(w1 - w2)

    if transposed:
        img1 = img1.transpose(Image.Transpose.ROTATE_90)
        img2 = img2.transpose(Image.Transpose.ROTATE_90)

    img1, img2 = width_process(img1, img2)
    img1, img2 = height_process(img1, img2)

    if transposed:
        img1 = img1.transpose(Image.Transpose.ROTATE_270)
        img2 = img2.transpose(Image.Transpose.ROTATE_270)
    return img1, img2
