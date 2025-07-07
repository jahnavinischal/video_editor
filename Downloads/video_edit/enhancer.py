import cv2
import numpy as np

def enhance_frame(frame, denoise_val, sharpen_val, clahe_clip):
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, denoise_val, denoise_val, 7, 21)

    # Sharpening
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1, 8 + sharpen_val / 10.0, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=max(1.0, clahe_clip), tileGridSize=(8, 8))
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    enhanced = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
