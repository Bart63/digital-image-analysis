# Digital Image Analysis
Project: Similar image search

## Dataset
__ImageNetV2__ - dataset contains new test data for the image benchmark. Divided in 3 overlapping sets with 10,000 images each (10 images per class). Data collection process for ImageNetV2 was developed in such a way so that the resulting distribution is as similar as possible to the original ImageNet dataset. 

__Database images__: from __TopImages__.

__Query images__: from merging __Threshold0.7__ with __MatchedFrequency__ and deleting identical images from __TopImages__. 

Only a subset of classes (and images for __Query images__) was chosen where images are similar to each other and the class object is big enough.

## Preprocessing
1. __Scale and crop__: Scale down dimension with the lowest difference and center crop.
2. __Convert to colorspace__: grayscale or HSV. 

## Methods

### Metrics
- [x] MSE
- [x] SSIM
- [x] PSNR
- [x] Cosine similarity
- [x] Bhattacharyya distance

### Image2Histogram methods
- [x] Grayscale/Color histogram
- [x] Edge-based description
- [x] Local Binary Pattern

### Image2Vector methods
- [x] Histogram of Oriented Gradients
- [x] Statistical (mean, moments, entropy, energy)
- [x] Haralick

### Image2Image methods
- [x] Fourier spectrum
- [x] Laws’ Texture Energy Measures
- [x] Gabor Filters

## Evaluation
__Top 10 accuracy__: if any image from top 10 belongs to the class of the query, we count it as a success.
