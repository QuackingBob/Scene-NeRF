import cv2
import numpy as np

def apply_gaussian_noise(image, num_times, mean=0, std=1):
    """
    Applies Gaussian noise to a grayscale image a specified number of times.
    """
    noisy_image = image.copy().astype(np.float64)
    height, width = image.shape[:2]

    for i in range(num_times):
        noise = np.random.normal(mean, std, size=(height, width))

        noisy_image = cv2.addWeighted(noisy_image, 0.75, noise, 0.25, 0)


    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image

im = cv2.imread('sampledepth.png', cv2.IMREAD_GRAYSCALE) / 255.0
im = cv2.resize(im, (500,500))
im = np.divide(1, 1 + np.exp(5 * (-im + 0.5)))
print(np.max(im))
print(np.min(im))
cv2.imshow(f'im', cv2.applyColorMap((im*255.0).astype(np.uint8), cv2.COLORMAP_PARULA))
cv2.waitKey(0)
# im = cv2.resize(im, dsize=im.shape, fx=0.5, fy=0.5)
# print(im.shape)
# for i in range(21):
#     im2 = apply_gaussian_noise(im, i)
#     cv2.imshow(f'im {i}', cv2.applyColorMap((im2*255.0).astype(np.uint8), colormap=np.random.randint(1, 20, size=1)[0]))
#     cv2.waitKey(0)