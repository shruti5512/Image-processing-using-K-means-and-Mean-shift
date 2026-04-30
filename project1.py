import numpy as np
import cv2
from sklearn.cluster import KMeans, MeanShift
from skimage import color
import matplotlib.pyplot as plt
from PIL import Image

# Helper function to crop image borders
def crop_image_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find the bounding box
    return image[y:y+h, x:x+w]

# Helper function for decorrelation stretch
def decorrelation_stretch(image):
    reshaped = image.reshape(-1, 3)
    mean = np.mean(reshaped, axis=0)
    cov = np.cov(reshaped, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    transform = eig_vecs / np.sqrt(eig_vals)
    stretched = np.dot((reshaped - mean), transform.T)
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched.reshape(image.shape)

# Function to apply segmentation
def segment_image(image, use_meanshift=False, include_position=False):
    # Convert to HSV color space for better segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    reshaped = hsv_image.reshape(-1, 3)

    if include_position:
        rows, cols, _ = hsv_image.shape
        positions = np.indices((rows, cols)).reshape(2, -1).T
        reshaped = np.hstack((reshaped, positions / np.array([rows, cols])))

    if use_meanshift:
        ms = MeanShift()
        ms.fit(reshaped)
        labels = ms.labels_
    else:
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(reshaped)

    segmented = labels.reshape(hsv_image.shape[:2])
    return segmented

# Load and process images
image_paths = [
    'C:/Users/pshru/OneDrive - University of Massachusetts/Desktop/project_ip/0073MR0003970000103657E01_DRCL.tif',
    'C:/Users/pshru/OneDrive - University of Massachusetts/Desktop/project_ip/0174ML0009370000105185E01_DRCL.tif',
    'C:/Users/pshru/OneDrive - University of Massachusetts/Desktop/project_ip/0617ML0026350000301836E01_DRCL.tif',
    'C:/Users/pshru/OneDrive - University of Massachusetts/Desktop/project_ip/1059ML0046560000306154E01_DRCL.tif'
]
for image_path in image_paths:
    # Load and preprocess image
    image = cv2.imread(image_path)
    cropped_image = crop_image_borders(image)
    stretched_image = decorrelation_stretch(cropped_image)

    # Apply segmentation (both K-means and Mean-shift)
    segmented_kmeans = segment_image(stretched_image, use_meanshift=False, include_position=True)
    segmented_meanshift = segment_image(stretched_image, use_meanshift=True, include_position=True)

    # Plot results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Original (Cropped)')
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_kmeans, cmap='jet')
    plt.title('K-means Segmentation')
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_meanshift, cmap='jet')
    plt.title('Mean-shift Segmentation')
    plt.show()
