import cv2
import numpy as np
import pywt


__all__=['FastRetinex','GammaCorrection','Dehaze']

def get_dark_channel(image, size=15):
    b, g, r = cv2.split(image)
    dark_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel

def get_atmosphere(image, dark_channel):
    h, w = image.shape[:2]
    image_size = h * w
    num_pixels = max(int(image_size / 1000), 1)
    dark_vec = dark_channel.ravel()
    indices = np.argsort(dark_vec)[image_size - num_pixels::]
    atmosphere = np.mean(image.reshape(image_size, 3)[indices], axis=0)
    return atmosphere

def get_transmission(image, atmosphere, omega=0.95, size=15):
    norm_image = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        norm_image[:, :, i] = image[:, :, i] / atmosphere[i]
    transmission = 1 - omega * get_dark_channel(norm_image, size)
    return transmission

def guided_filter(image, p, r, eps):
    mean_I = cv2.boxFilter(image, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(image * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(image * image, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * image + mean_b
    return q

def Dehaze(image, size=15, omega=0.95, t0=0.1, r=40, eps=1e-6,**kwargs):
    dark_channel = get_dark_channel(image/255., size)
    atmosphere = get_atmosphere(image/255., dark_channel)
    transmission = get_transmission(image/255., atmosphere, omega, size)
    transmission = guided_filter(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0, transmission, r, eps)

    transmission = np.maximum(transmission, t0)
    restored_image = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        restored_image[:, :, i] = (image[:, :, i] /255.- atmosphere[i]) / transmission + atmosphere[i]

    restored_image = np.clip(restored_image, 0, 1)
    return (restored_image * 255).astype(np.uint8)


def msr(image, scales, weights):
    result = np.zeros_like(image, dtype=np.float32)
    for i, sigma in enumerate(scales):
        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        result += weights[i] * (np.log1p(image) - np.log1p(gaussian))
    return result

def fast_msr(image, scales, weights):
    v_msr = msr(image, scales, weights)
    v_msr = cv2.normalize(v_msr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return v_msr

def apply_wavelet_transform(channel):
    coeffs2 = pywt.dwt2(channel, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, (LH, HL, HH)

def inverse_wavelet_transform(LL, coeffs):
    return pywt.idwt2((LL, coeffs), 'haar')



def GammaCorrection(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), table)


def FastRetinex(image):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # h, s, v = cv2.split(hsv)
    h,s,v = cv2.split(hsv)
    # Apply Haar DWT on the V channel
    v_LL, v_coeffs = apply_wavelet_transform(v)


    # Apply MSR to the LL part of the wavelet transform
    scales = [10, 60, 90, 160]
    weights = [0.25, 0.25, 0.25, 0.25]
    v_msr = fast_msr(v_LL, scales, weights)

    # Inverse wavelet transform
    v_enhanced = inverse_wavelet_transform(v_msr, v_coeffs)

    # Clip values to ensure they are in valid range
    v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)

    # Merge back the enhanced V channel with original H and S channels
    hsv_enhanced = cv2.merge([h,s,v_enhanced])
    rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    
    return rgb_enhanced