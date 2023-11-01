def filter_images(ds, indexes=[]):
    return [
        (img, idx)
        for img, idx in ds
        if idx in indexes or (not indexes) # if indexes is empty, return all
    ]
