#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2 as cv2
from functions.filter_bank import (
    oriented_derivative_of_gaussian_bank,
    lm_bank,
    half_disk_bank,
    gabor_bank,
    read_images_and_filter,
)

from functions.maps import (
    texton_images,
    brightness_images,
    color_images,
    gradient_images,
)
from functions.basic_operations import plot_filtered_images, plot_filtered_final
import os


def main():
    # Number of image to compute
    number_images = 5

    # Cluster for texture or texton
    K_texture = 64

    # Cluster for brightness
    K_brightness = 16

    # Cluster for color
    K_color = 16

    # Vector with the clusters
    K = [K_texture, K_brightness, K_color]

    ## Path to save the filters bank
    results_path = os.path.join(os.getcwd(), "Filter_bank")
    results_texton_path = os.path.join(os.getcwd(), "Texture_map")
    results_texton_gradient_path = os.path.join(os.getcwd(), "Texture_map_gradient")
    results_brightness_path = os.path.join(os.getcwd(), "Brightness_map")
    results_brightness_gradient_path = os.path.join(
        os.getcwd(), "Brightness_map_gradient"
    )
    results_color_path = os.path.join(os.getcwd(), "Color_map")
    results_color_gradient_path = os.path.join(os.getcwd(), "Color_map_gradient")

    results_maps_path = os.path.join(os.getcwd(), "Maps")
    results_maps_gradient_path = os.path.join(os.getcwd(), "Maps_gradient")

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    images_canny_path = os.path.join(parent_directory, "BSDS500", "CannyBaseline")
    images_sobel_path = os.path.join(parent_directory, "BSDS500", "SobelBaseline")

    results_pb_path = os.path.join(os.getcwd(), "Pb")

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """

    # create a directory in case it does not exist
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(results_texton_path, exist_ok=True)
    os.makedirs(results_brightness_path, exist_ok=True)
    os.makedirs(results_color_path, exist_ok=True)
    os.makedirs(results_maps_path, exist_ok=True)

    os.makedirs(results_texton_gradient_path, exist_ok=True)
    os.makedirs(results_brightness_gradient_path, exist_ok=True)
    os.makedirs(results_color_gradient_path, exist_ok=True)
    os.makedirs(results_maps_gradient_path, exist_ok=True)
    os.makedirs(results_pb_path, exist_ok=True)

    ## Oriented DOG filters
    name_gauss = "DoG"
    scales_gauss = np.array([1.0, 2.0])
    angles_gauss = np.linspace(0.0, 360, num=16, endpoint=False, retstep=False)
    gauss_derivative_bank = oriented_derivative_of_gaussian_bank(
        scales_gauss, angles_gauss, name_gauss, results_path
    )

    """
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
    # LMS filter
    name_file_lms = "LM_Small"
    scales_small = np.array([1.0, np.sqrt(2), 2, 2 * np.sqrt(2)])
    angles = np.linspace(0.0, 360, num=6, endpoint=False, retstep=False)
    lms_bank = lm_bank(scales_small, angles, name_file_lms, results_path)

    ## LMS filter
    name_file_lml = "LM_Large"
    scales_large_ = np.array([np.sqrt(2), 2, 2 * np.sqrt(2), 4])
    lml_bank = lm_bank(scales_large_, angles, name_file_lml, results_path)

    """
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
    ## Gabor Filter
    scales_gabor = np.array([5, 6, 7, 8, 9])
    angles_gabor = np.linspace(0.0, 360, num=16, endpoint=False, retstep=False)
    name_file_gabor = "Gabor"
    bank_gabor = gabor_bank(scales_gabor, angles_gabor, name_file_gabor, results_path)
    """
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
    # Half Filter
    sizes_half = np.array([11, 21, 31], dtype=np.int8)
    angles_half = np.linspace(0.0, 360, num=16, endpoint=False, retstep=False)
    name_file_half_1 = "HDMask_1"
    name_file_half_2 = "HDMask_2"
    name_file_half = "HDMasks"
    half_1_bank, half_2_bank = half_disk_bank(
        sizes_half,
        angles_half,
        name_file_half_1,
        name_file_half_2,
        name_file_half,
        results_path,
    )

    # Stack filters
    gauss_derivative_bank = gauss_derivative_bank.reshape(
        (gauss_derivative_bank.shape[0] * gauss_derivative_bank.shape[1],)
    )
    lms_bank = lms_bank.reshape((lms_bank.shape[0] * lms_bank.shape[1],))
    lml_bank = lml_bank.reshape((lml_bank.shape[0] * lml_bank.shape[1],))
    bank_gabor = bank_gabor.reshape((bank_gabor.shape[0] * bank_gabor.shape[1],))

    ## Final filter bank of the system
    bank = np.hstack((gauss_derivative_bank, lms_bank, lml_bank, bank_gabor))

    # bank = bank_gauss
    bank = bank.reshape((bank.shape[0], 1))

    # Stack filters
    half_1_bank = half_1_bank.reshape((half_1_bank.shape[0] * half_1_bank.shape[1],))
    half_2_bank = half_2_bank.reshape((half_2_bank.shape[0] * half_2_bank.shape[1],))

    # Save filters
    name_filter_bank = "filter_bank"
    filter_bank_path = os.path.join(results_path, f"{name_filter_bank}.npy")
    np.save(filter_bank_path, bank)

    # Save half disk filters
    filter_half_disk_1_path = os.path.join(results_path, f"{name_file_half_1}.npy")
    filter_half_disk_2_path = os.path.join(results_path, f"{name_file_half_2}.npy")

    np.save(filter_half_disk_1_path, half_1_bank)
    np.save(filter_half_disk_2_path, half_2_bank)

    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """
    images_filters, images_gray, images_rgb, images_canny, images_sobel = (
        read_images_and_filter(number_images, bank, results_path)
    )

    """
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """
    filtered_images_texton = texton_images(
        number_images, K_texture, results_path, results_texton_path
    )

    """
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """

    filtered_images_texton_gradient = gradient_images(
        number_images,
        K_texture,
        results_texton_path,
        results_texton_gradient_path,
        "texture",
    )

    """
    Generate Brightness Map
    Perform brightness binning
    """
    filtered_brightness_images = brightness_images(
        number_images,
        K_brightness,
        results_path,
        results_brightness_path,
    )

    """
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
     """
    filtered_images_brightness_gradient = gradient_images(
        number_images,
        K_brightness,
        results_brightness_path,
        results_brightness_gradient_path,
        "brightness",
    )

    """
    Generate Color Map
    Perform color binning or clustering
    """
    filtered_color_images = color_images(
        number_images, K_color, results_path, results_color_path
    )
    #
    """
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    filtered_images_color_gradient = gradient_images(
        number_images,
        K_color,
        results_color_path,
        results_color_gradient_path,
        "color",
    )

    # Save all the results in order to get the images for the report
    full_filter = np.hstack(
        (filtered_images_texton, filtered_brightness_images, filtered_color_images)
    )

    full_gradient_filter = np.hstack(
        (
            filtered_images_texton_gradient,
            filtered_images_brightness_gradient,
            filtered_images_color_gradient,
        )
    )

    name = "images_maps"
    name_gradient = "images__gradient_maps"

    plot_filtered_images(full_filter, name, results_maps_path)
    plot_filtered_images(
        full_gradient_filter, name_gradient, results_maps_gradient_path
    )
    #
    filter_bank_path = os.path.join(results_maps_path, f"{name}.npy")
    filter_bank_gradient_path = os.path.join(
        results_maps_gradient_path, f"{name_gradient}.npy"
    )

    np.save(filter_bank_path, full_filter)
    np.save(filter_bank_gradient_path, full_gradient_filter)

    load_texton_images = np.load(filter_bank_path, allow_pickle=True)
    load_texton_gradient_images = np.load(filter_bank_gradient_path, allow_pickle=True)

    """
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
    images_names = [str(i) for i in range(1, number_images + 1)]
    empty_matrix_sobel = np.empty((len(images_names), 1), dtype=object)
    for image_number in range(0, len(images_names)):
        image_sobel = cv2.imread(
            images_sobel_path + "/" + images_names[image_number] + ".png"
        )
        # Transform image from cv2 to numpy
        sobel_float = image_sobel.astype(np.double)
        image_sobel_data = np.array(sobel_float)
        empty_matrix_sobel[image_number, 0] = image_sobel_data[:, :, 0]

    """
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
    images_names = [str(i) for i in range(1, number_images + 1)]
    empty_matrix_canny = np.empty((len(images_names), 1), dtype=object)
    for image_number in range(0, len(images_names)):
        image_canny = cv2.imread(
            images_canny_path + "/" + images_names[image_number] + ".png"
        )
        # Transform image from cv2 to numpy
        canny_float = image_canny.astype(np.double)
        image_canny_data = np.array(canny_float)
        empty_matrix_canny[image_number, 0] = image_canny_data[:, :, 0]

    """
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    empty_matrix_pb = np.empty((len(images_names), 1), dtype=object)
    for image_number in range(0, len(images_names)):

        load_canny_filter = empty_matrix_canny[image_number, 0]
        load_sobel_filter = empty_matrix_sobel[image_number, 0]

        load_canny_filter = load_canny_filter
        load_sobel_filter = load_sobel_filter

        # Texture
        texture_gradient = 255 * (load_texton_gradient_images[image_number, 0] / 1)
        brightness_gradient = 255 * (load_texton_gradient_images[image_number, 1] / 1)
        color_gradient = 255 * (load_texton_gradient_images[image_number, 2] / 1)

        pb_edges = ((texture_gradient + brightness_gradient + color_gradient) / 3) * (
            load_canny_filter * 0.5 + load_sobel_filter * 0.5
        )

        empty_matrix_pb[image_number, 0] = pb_edges

        abs_grad_x = cv2.convertScaleAbs(pb_edges)

        window_name = "Sobel Demo - Simple Edge Detector"
        cv2.imshow(window_name, abs_grad_x)
        cv2.waitKey(0)

    name_final = "final_results"
    results = np.hstack((empty_matrix_sobel, empty_matrix_canny, empty_matrix_pb))
    print(results.shape)
    plot_filtered_final(results, name_final, results_pb_path)


if __name__ == "__main__":
    main()
