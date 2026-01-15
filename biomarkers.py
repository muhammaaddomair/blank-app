import cv2
import numpy as np
from skimage.morphology import skeletonize

def binarize_vessels(rgb_img: np.ndarray) -> np.ndarray:
    """
    rgb_img: HxWx3 uint8 RGB
    returns: HxW binary mask (0/1) as uint8
    """
    green = rgb_img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return (binary > 0).astype(np.uint8)

def vessel_density(binary_mask: np.ndarray) -> float:
    return float(binary_mask.sum() / binary_mask.size)

def vessel_tortuosity(binary_mask: np.ndarray) -> float:
    """
    Lightweight tortuosity proxy from skeleton; stable for deployment.
    """
    if binary_mask.sum() == 0:
        return 0.0

    skel = skeletonize(binary_mask.astype(bool)).astype(np.uint8)
    length = float(skel.sum())
    count = float(np.count_nonzero(skel))
    return float(length / (count + 1e-6))

def fractal_dimension(binary_mask: np.ndarray) -> float:
    """
    Box-counting fractal dimension on the binary vessel mask.
    """
    Z = binary_mask.astype(np.uint8)
    if Z.sum() == 0:
        return 0.0

    n = min(Z.shape)
    Z = Z[:n, :n]

    max_pow = int(np.log2(n))
    if max_pow <= 1:
        return 0.0

    sizes = 2 ** np.arange(1, max_pow)

    def boxcount(A, k):
        S = np.add.reduceat(
            np.add.reduceat(A, np.arange(0, A.shape[0], k), axis=0),
            np.arange(0, A.shape[1], k),
            axis=1
        )
        return np.count_nonzero(S)

    counts = np.array([boxcount(Z, s) for s in sizes], dtype=np.float64)
    sizes = sizes.astype(np.float64)

    # Avoid log(0)
    mask = counts > 0
    if mask.sum() < 2:
        return 0.0

    coeffs = np.polyfit(np.log(sizes[mask]), np.log(counts[mask]), 1)
    return float(-coeffs[0])
