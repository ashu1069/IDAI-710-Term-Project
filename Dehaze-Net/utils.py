import numpy as np
import cv2

def rgb2gray(rgb):
    """Convert an RGB or grayscale image to grayscale."""
    if rgb.shape[2] == 3:
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale conversion
    elif rgb.shape[2] == 1:
        return rgb[:, :, 0]  # It's already a grayscale image, just remove the channel dimension
    else:
        raise ValueError("Invalid image format: image must have either 1 or 3 channels")

def ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = rgb2gray(img1).astype(np.float64)
    img2 = rgb2gray(img2).astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # Calculate local means
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
