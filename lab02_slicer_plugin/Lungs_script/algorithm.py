# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - [Lungs Segmentation](#toc1_)    
#   - [Imports](#toc1_1_)    
#   - [Load / save .nii files and Visualization](#toc1_2_)    
#   - [Binarize functions](#toc1_3_)    
#   - [Body Mask](#toc1_4_)    
#   - [Lungs segmentation](#toc1_5_)    
#   - [Compare with reference](#toc1_6_)    
#   - [Main](#toc1_7_)    
#   - [Run code](#toc1_8_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# # <a id='toc1_'></a>[Lungs Segmentation](#toc0_)
# The segmentation of lungs may, possibly, proceed directly in 3D as follows:
# 1. Run binarization of the CT image using a threshold of -320 HU – every voxel
# with HU lower than this threshold should receive label 1 (air label) and the
# remaining voxels should receive label 0
# 2. Use body mask to select only air regions within body
# 3. Design a sequence of morphological (and other appropriate) operations to fill
# the holes in the interior of lungs and to remove ‘air’ clusters which do not
# correspond to lungs (e.g. gas in bowels) – at the end one should be left with
# clusters which correspond only to airways
# 4. Use watershed from markers (scikit-image -> segmentation -> watershed) to
# extract the left and the right lung from the segmentation being the result of step
# (3) above. Before using watershed design a procedure for defining the three
# markers (marker of left lung, marker of right lung, marker of background).
# 5. To compare segmentation results with reference segmentations available at
# Lab One Drive use Dice coefficient and Hausdorff distance (find the definitions of
# these   quantities)   as   implemented   in   surface-distance   [package](https://github.com/google-deepmind/surface-distance).
# The project results (Dice coefficients and Hausdorff distance) should be
# reported   for   the   three   tasks:   body   mask   segmentation,   left   lung
# segmentation, right lung segmentation.
# 

# %% [markdown]
# ## <a id='toc1_1_'></a>[Imports](#toc0_)

# %%
import numpy as np
import nibabel as nib
import cv2
from skimage import morphology, measure
from skimage.segmentation import watershed
from skimage import filters
from skimage.filters import rank
from scipy import ndimage
import matplotlib.pyplot as plt

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction
from skimage.segmentation import flood, flood_fill
from sklearn import cluster
from skimage.segmentation import watershed

# Remove the following import
# from surface_distance import metrics

# Add custom Dice coefficient computation function
def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the Dice coefficient between two binary masks.
    
    :param mask1: First binary mask.
    :param mask2: Second binary mask.
    :return: Dice coefficient as a float.
    """
    intersection = np.sum(mask1 & mask2)
    volume_sum = np.sum(mask1) + np.sum(mask2)
    if volume_sum == 0:
        return 1.0
    return 2.0 * intersection / volume_sum

# %% [markdown]
# ## <a id='toc1_2_'></a>[Load / save .nii files and Visualization](#toc0_)

# %%
def load_nii_gz_file(file_path: str) -> tuple:
    nii_img = nib.load(file_path)
    nii_data = nii_img.get_fdata()
    return nii_data, nii_img.affine

def save_to_nii(segmented_data: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    segmented_nii = nib.Nifti1Image(segmented_data.astype(np.uint8), affine)
    nib.save(segmented_nii, output_path)
    
def view_nii_data(nii_data: np.ndarray) -> None:
    for i in range(nii_data.shape[2]):
        cv2.imshow('slice', nii_data[:, :, i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def visualize_photo(img: np.ndarray, photo_title: str, *slices: int) -> None:
    print(f"Visualizing {photo_title}")
    plt.figure(figsize=(5 * len(slices), 5)) 
    
    for i, slice_num in enumerate(slices):
        plt.subplot(1, len(slices), i + 1)
        plt.title(f"photo Slice {slice_num}")
        plt.imshow(img[:, :, slice_num], cmap="gray")
        
    plt.tight_layout() 
    plt.show()
    
def visualize_photo_with_centers(img: np.ndarray, photo_title: str, centers: np.ndarray, *slices: int) -> None:
    print(f"Visualizing {photo_title}")
    plt.figure(figsize=(5 * len(slices), 5)) 
    
    for i, slice_num in enumerate(slices):
        plt.subplot(1, len(slices), i + 1)
        plt.title(f"photo Slice {slice_num}")
        plt.imshow(img[:, :, slice_num], cmap="gray")

        for (_, x, y) in centers:
            plt.scatter(x, y, color="red", s=100, marker="o") 
        
    plt.tight_layout() 
    plt.show()    
    
def visualize_photos(original: np.ndarray, segmented: np.ndarray, reference: np.ndarray, *slices: int) -> None:
    num_slices = len(slices)
    plt.figure(figsize=(15, 5 * num_slices))  # Adjust figure size based on the number of slices

    for i, slice_num in enumerate(slices):
        # Original slice
        plt.subplot(num_slices, 3, 3 * i + 1)
        plt.title(f"Original Slice {slice_num}")
        plt.imshow(original[:, :, slice_num], cmap="gray")
        
        # Segmented slice
        plt.subplot(num_slices, 3, 3 * i + 2)
        plt.title(f"Segmented Slice {slice_num}")
        plt.imshow(segmented[:, :, slice_num], cmap="gray")
        
        # Reference slice
        plt.subplot(num_slices, 3, 3 * i + 3)
        plt.title(f"Reference Slice {slice_num}")
        plt.imshow(reference[:, :, slice_num], cmap="gray")

    plt.tight_layout() 
    plt.show()
    
def plot_histogram(img: np.ndarray) -> None:
    plt.hist(img.ravel(), bins=256, range=(img.min()+1, img.max()-1), fc='k', ec='k')
    plt.axvline(x=-320, color='red', linestyle='--', linewidth=1.5)
    plt.show()

# %% [markdown]
# ## <a id='toc1_3_'></a>[Binarize functions](#toc0_)

# %%
THR_HU = -320

def binarize(image: np.ndarray, threshold: float = THR_HU) -> np.ndarray:
    res = image.copy()
    res[image >= threshold] = 1
    res[image < threshold] = 0
    return res

# %% [markdown]
# ## <a id='toc1_4_'></a>[Body Mask](#toc0_)

# %%
def resize_3d_image(img: np.ndarray, output_shape: tuple) -> np.ndarray:
    sizes = img.shape
    resized_img = np.zeros((output_shape[0], output_shape[1], sizes[2]), dtype=img.dtype)
    
    for i in range(img.shape[2]):
        resized_img[:,:,i] = cv2.resize(img[:,:,i], (output_shape[0], output_shape[1]), interpolation=cv2.INTER_LINEAR) 
    
    return resized_img


def medianBlur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    sizes = img.shape
    blurred = np.zeros_like(img)
    
    for i in range(sizes[2]):
        blurred[:,:,i] = cv2.medianBlur(img[:,:,i], kernel_size)
    
    return blurred

def prepare_before_body_mask(img: np.ndarray) -> np.ndarray:
    binary_image = binarize(img, THR_HU)
    binary_image = ndimage.binary_fill_holes(binary_image)
    binary_image = morphology.binary_opening(binary_image, morphology.ball(2))
    binary_image = morphology.binary_closing(binary_image, morphology.ball(2))
    return binary_image


def reconstruct_3d(img: np.ndarray) -> np.ndarray:
    labels = label(img, connectivity=1)
    regions = regionprops(labels)
    largest_region = max(regions, key=lambda x: x.area)
    largest_region_mask = labels == largest_region.label
    
    mask = largest_region_mask
    neg_mask = np.logical_not(mask)
    
    seed = np.zeros_like(mask)
    list_of_corners = [(0, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0), (-1, -1, -1), (-1, -1, 0), (-1, 0, -1), (0, -1, -1)]
    for corner in list_of_corners:
        seed[corner] = 1
    
    reconstructed = reconstruction(seed, neg_mask)
    insert_part = 1 - reconstructed
    
    return insert_part

def create_body_mask(img: np.ndarray) -> np.ndarray:
    binary_image = prepare_before_body_mask(img)
    body_mask = reconstruct_3d(binary_image)
    return body_mask

# %% [markdown]
# ## <a id='toc1_5_'></a>[Lungs segmentation](#toc0_)

# %%
def lungs_initial_transform(img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    lungs = cut_out_body(img, body_mask)
    lungs = binarize(lungs, THR_HU)
    lungs = body_mask - lungs
    lungs = abs(lungs)
    lungs = morphology.binary_closing(lungs, morphology.ball(2))
    lungs = morphology.binary_opening(lungs, morphology.ball(2))
    return lungs

def kmeans_clusterization(img: np.ndarray) -> np.ndarray:
    markers_x, markers_y, markers_z = np.nonzero(img)
    
    initial_clusters = cluster.KMeans(n_clusters=2)
    _ = initial_clusters.fit_predict(np.stack((markers_x, markers_y, markers_z), axis=-1))
    initial_centers = initial_clusters.cluster_centers_
    
    centers = []
    for center in initial_centers:
        center_x, center_y, center_z = center
        distances = ((markers_x - center_x) ** 2 + (markers_y - center_y) ** 2 + (markers_z - center_z) ** 2) ** 0.5
        closest_index = np.argmin(distances)
        centers.append((markers_x[closest_index], markers_y[closest_index], markers_z[closest_index]))
    
    visualize_photo_with_centers(img, "After kmeans with centers", centers, 50, 70, 100, 120)
    centers = np.array(centers)
    centers = centers[centers[:, 1].argsort()] # sorting centers by x coordinate
    return np.array(centers)

def watershed_segmentation(img: np.ndarray, centers: tuple) -> np.ndarray:
    markers = np.zeros(img.shape, dtype=np.int32)

    for i, center in enumerate(centers):
        center_3d_point = tuple(np.round(center).astype(int))
        print(f"Center point {center_3d_point}")

        if 0 <= center_3d_point[0] < img.shape[0] and 0 <= center_3d_point[1] < img.shape[1]:
            markers[center_3d_point] = i + 1  
            
    gradient = rank.gradient(img, morphology.ball(2))
    return watershed(gradient, markers, mask=img)

def cut_out_body(img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    res = img.copy()
    res = img * body_mask + img.min() * (1 - body_mask)
    return res

# %% [markdown]
# ## <a id='toc1_6_'></a>[Compare with reference](#toc0_)

# %%
class Scores:
    def __init__(self, img: np.ndarray, ref_img: np.ndarray, centers: np.ndarray, body_masks: np.ndarray, body_masks_ref: np.ndarray) -> None:
        self.img: np.ndarray = img
        self.ref_img: np.ndarray = ref_img
        self.body_masks = body_masks
        self.body_masks_ref = body_masks_ref
        plt.imshow(self.img[..., 70], cmap='gray')
        plt.show()
        self.centers = centers
        self.my_left_lung, self.my_right_lung = self.__get_lungs_from_labels(image=self.img, left_label=1, right_label=2)
        self.ref_left_lung, self.ref_right_lung = self.__get_lungs_from_labels(image=self.ref_img, left_label=2, right_label=3)
        
    def __get_lungs_from_labels(self, image: np.ndarray, left_label: int, right_label: int) -> tuple[np.ndarray, np.ndarray]:
        left_lung = np.where(image == left_label, 1, 0)
        right_lung = np.where(image == right_label, 1, 0)
        return left_lung, right_lung
        
    def calculate_dice_coef(self) -> tuple[float, float, float]:
        dice_coef_left = compute_dice_coefficient(self.my_left_lung.astype(bool), self.ref_left_lung.astype(bool))
        dice_coef_right = compute_dice_coefficient(self.my_right_lung.astype(bool), self.ref_right_lung.astype(bool))
        dice_coef_body_masks = compute_dice_coefficient(self.body_masks.astype(bool), self.body_masks_ref.astype(bool))
        return dice_coef_left, dice_coef_right, dice_coef_body_masks
    
    # Remove the following method to avoid dependency on 'surface_distance'
    # def caculate_hausdorff_distance(self) -> tuple[float, float, float]:
    #     left_surf_distances = metrics.compute_surface_distances(self.my_left_lung.astype(bool), self.ref_left_lung.astype(bool), (1, 1, 1))
    #     right_surf_distances = metrics.compute_surface_distances(self.my_right_lung.astype(bool), self.ref_right_lung.astype(bool), (1, 1, 1))
    #     body_mask_distances = metrics.compute_surface_distances(self.body_masks.astype(bool), self.body_masks_ref.astype(bool), (1, 1, 1))
    #     
    #     hausdorff_left = metrics.compute_robust_hausdorff(left_surf_distances, 95)
    #     hausdorff_right = metrics.compute_robust_hausdorff(right_surf_distances, 95)
    #     hausdorff_body_mask = metrics.compute_robust_hausdorff(body_mask_distances, 95)
    #     
    #     return hausdorff_left, hausdorff_right, hausdorff_body_mask
    
    def plot_lungs(self) -> None:
        for i in (50, 70, 90, 100):
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 6, 1)
            plt.title('My left lung')
            plt.imshow(self.my_left_lung[..., i], cmap='gray')
            plt.subplot(1, 6, 2)
            plt.title('Reference left lung')
            plt.imshow(self.ref_left_lung[..., i], cmap='gray')
            plt.subplot(1, 6, 3)
            plt.title('My right lung')
            plt.imshow(self.my_right_lung[..., i], cmap='gray')
            plt.subplot(1, 6, 4)
            plt.title('Reference right lung')
            plt.imshow(self.ref_right_lung[..., i], cmap='gray')
            plt.subplot(1, 6, 5)
            plt.title('Diff right')
            plt.imshow(abs(self.my_right_lung[:, :, i].astype('int') - self.ref_right_lung[..., i].astype('int')), cmap='gray')
            plt.subplot(1, 6, 6)
            plt.title('Diff left')
            plt.imshow(abs(self.my_left_lung[:, :, i].astype('int') - self.ref_left_lung[..., i].astype('int')), cmap='gray')
            plt.show()

# %% [markdown]
# ## <a id='toc1_7_'></a>[Main](#toc0_)

# %%
def lunginator_3000(img: np.ndarray) -> np.ndarray:
    # Body mask creation
    body_mask = create_body_mask(img)
    
    # Lung segmentation
    lungs = lungs_initial_transform(img, body_mask)    
    centers = kmeans_clusterization(lungs)
    result = watershed_segmentation(lungs, centers)
         
    return result




