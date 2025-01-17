import numpy as np
import cv2 as cv2
from .basic_operations import (
    gaussian_2d,
    gabor_2d,
    first_derivative_gaussian_2d_x,
    second_derivative_gaussian_2d_x,
    laplacian_gaussian_2d,
    filter_bank,
    filter_bank_half_disk,
    plot_banks,
    convolve2d_manual_filter_bank,
    half__disk_2d_negative,
    half__disk_2d_positive,
)
import os


def oriented_derivative_of_gaussian_bank(scales, angles, name_file, location):
    # Name of the file
    elongation = False

    # size of the kernel
    size = 21

    # Computing aproximation using gauss and sobel
    # bank_1 = filter_bank_gauss_x_aprox(scales, angles, size, elongation)

    # Computing the first derivative of a gaussian kernel
    bank_1 = filter_bank(
        scales, angles, size, elongation, first_derivative_gaussian_2d_x
    )

    # New variable
    bank = bank_1

    # Plot Results
    plot_banks(bank, name_file, location)

    return bank


def half_disk_bank(scales, angles, name_file_1, name_file_2, name, location):
    # Name of the file
    elongation = False

    # Computing the first derivative of a gaussian kernel
    bank_side_1 = filter_bank_half_disk(
        scales, angles, elongation, half__disk_2d_positive
    )
    bank_side_2 = filter_bank_half_disk(
        scales, angles, elongation, half__disk_2d_negative
    )

    plot_banks(bank_side_1, name_file_1, location)
    plot_banks(bank_side_2, name_file_2, location)

    bank_results = np.empty(
        (bank_side_1.shape[0], bank_side_1.shape[1] + bank_side_2.shape[1]),
        dtype=object,
    )

    for j in range(0, bank_results.shape[0]):
        ii = 0
        for i in range(0, bank_side_1.shape[1]):
            bank_results[j, ii] = bank_side_1[j, i]
            bank_results[j, ii + 1] = bank_side_2[j, i]
            ii = ii + 2

    plot_banks(bank_results, name, location)
    return bank_side_1, bank_side_2


def lm_bank(scales, angles, name_file_lms, location):
    elongation_derivatives = True
    elongation_gauss = False

    # Size of the kernel
    size = 41

    # Computing the first derivative of a gaussian kernel
    # bank_1 = filter_bank_gauss_x_aprox(scales[0:3], angles, size, elongation)
    bank_1 = filter_bank(
        scales[0:3],
        angles,
        size,
        elongation_derivatives,
        first_derivative_gaussian_2d_x,
    )
    bank_2 = filter_bank(
        scales[0:3],
        angles,
        size,
        elongation_derivatives,
        second_derivative_gaussian_2d_x,
    )
    bank_3 = filter_bank(scales, np.array([0.0]), size, elongation_gauss, gaussian_2d)
    bank_4 = filter_bank(
        np.hstack([scales[:], 3 * scales[:]]),
        np.array([0.0]),
        size,
        elongation_gauss,
        laplacian_gaussian_2d,
    )
    # Reshape filter bank
    bank_3 = bank_3.T
    bank_4 = bank_4.T

    # Stack filters
    bank_aux_1 = np.hstack((bank_1, bank_2))
    bank_aux_2 = np.hstack((bank_4, bank_3))
    bank = np.vstack((bank_aux_1, bank_aux_2))

    # Plot Results
    plot_banks(bank, name_file_lms, location)

    return bank


def gabor_bank(scales, angles, name_file_lms, location):
    elongation_gauss = False

    # Size of the kernel
    size = 51

    bank_3 = filter_bank(scales, angles, size, elongation_gauss, gabor_2d)

    # Stack filters
    bank = bank_3

    # Plot Results
    plot_banks(bank, name_file_lms, location)

    return bank


def read_images_and_filter(number_images, bank, results_path):
    ## Get the current working directory
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    images_path = os.path.join(parent_directory, "BSDS500", "Images")

    images_canny_path = os.path.join(parent_directory, "BSDS500", "CannyBaseline")
    images_sobel_path = os.path.join(parent_directory, "BSDS500", "SobelBaseline")

    # Read the image
    # Be Careful with the images, try to call them by the location folder
    images_names = [str(i) for i in range(1, number_images + 1)]
    for image_number in range(0, len(images_names)):

        print(
            "-----------------------------------Reading Images-----------------------------"
        )
        # read Image
        image = cv2.imread(images_path + "/" + images_names[image_number] + ".jpg")

        image_canny = cv2.imread(
            images_canny_path + "/" + images_names[image_number] + ".png"
        )

        image_sobel = cv2.imread(
            images_sobel_path + "/" + images_names[image_number] + ".png"
        )
        # Move the image to the gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Transform image from cv2 to numpy
        gray_float = gray.astype(np.double)
        image_gray = np.array(gray_float)
        # Transform image from cv2 to numpy
        rgb_float = rgb.astype(np.double)
        image_rgb = np.array(rgb_float)

        # Transform image from cv2 to numpy
        canny_float = image_canny.astype(np.double)
        image_canny = np.array(canny_float)

        # Transform image from cv2 to numpy
        sobel_float = image_sobel.astype(np.double)
        image_sobel = np.array(sobel_float)

        # Filtering gray image
        image_filters = convolve2d_manual_filter_bank(image_gray, bank)

        # Save the image with its filters
        # Construct the file path
        filtered_image_texture_name = os.path.join(
            results_path, f"{images_names[image_number]}_texture.npy"
        )
        filtered_image_bridhtness_name = os.path.join(
            results_path, f"{images_names[image_number]}_brightness.npy"
        )

        filtered_image_color_name = os.path.join(
            results_path, f"{images_names[image_number]}_color.npy"
        )

        filtered_image_canny_name = os.path.join(
            results_path, f"{images_names[image_number]}_canny.npy"
        )

        filtered_image_sobel_name = os.path.join(
            results_path, f"{images_names[image_number]}_sobel.npy"
        )

        # Save the image with the different channels
        np.save(filtered_image_texture_name, image_filters)
        np.save(filtered_image_bridhtness_name, image_gray)
        np.save(filtered_image_color_name, image_rgb)
        np.save(filtered_image_canny_name, image_canny)
        np.save(filtered_image_sobel_name, image_sobel)

    return image_filters, image_gray, image_rgb, image_canny, image_sobel


def filter_banks(number_images):
    ## Path to save the filters
    results_path = os.path.join(os.getcwd(), "Filter_bank")

    # create a directory in case it does not exist
    os.makedirs(results_path, exist_ok=True)

    ## Get the current working directory
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    images_path = os.path.join(parent_directory, "BSDS500", "Images")

    images_canny_path = os.path.join(parent_directory, "BSDS500", "CannyBaseline")
    images_sobel_path = os.path.join(parent_directory, "BSDS500", "SobelBaseline")

    ## Oriented DOG filters
    name_gauss = "DoG"

    scales_gauss = np.array([1.0, 2.0])
    angles_gauss = np.linspace(0.0, 360, num=16, endpoint=False, retstep=False)

    bank_gauss = oriented_derivative_of_gaussian_bank(
        scales_gauss, angles_gauss, name_gauss, results_path
    )

    # LMS filter
    name_file_lms = "LM_Small"
    scales_small = np.array([1.0, np.sqrt(2), 2, 2 * np.sqrt(2)])
    angles = np.linspace(0.0, 360, num=6, endpoint=False, retstep=False)
    bank_lms = lm_bank(scales_small, angles, name_file_lms, results_path)

    ## LMS filter
    name_file_lml = "LM_Large"
    scales_large_ = np.array([np.sqrt(2), 2, 2 * np.sqrt(2), 4])
    bank_lml = lm_bank(scales_large_, angles, name_file_lml, results_path)

    ## Gabor Filter
    scales_gabor = np.array([5, 6, 7, 8, 9])
    angles_gabor = np.linspace(0.0, 360, num=16, endpoint=False, retstep=False)
    name_file_gabor = "Gabor"
    bank_gabor = gabor_bank(scales_gabor, angles_gabor, name_file_gabor, results_path)

    # Half Filter
    sizes_half = np.array([11, 21, 31], dtype=np.int8)
    angles_half = np.linspace(0.0, 360, num=16, endpoint=False, retstep=False)
    name_file_half_1 = "Half_disk_1"
    name_file_half_2 = "Half_disk_2"
    name_file_half = "HDMasks"
    bank_half_1, bank_half_2 = half_disk_bank(
        sizes_half,
        angles_half,
        name_file_half_1,
        name_file_half_2,
        name_file_half,
        results_path,
    )

    # Reshape data
    bank_half_1 = bank_half_1.reshape((bank_half_1.shape[0] * bank_half_1.shape[1],))
    bank_half_2 = bank_half_2.reshape((bank_half_2.shape[0] * bank_half_2.shape[1],))

    # Stack filter
    bank_gauss = bank_gauss.reshape((bank_gauss.shape[0] * bank_gauss.shape[1],))
    bank_lms = bank_lms.reshape((bank_lms.shape[0] * bank_lms.shape[1],))
    bank_lml = bank_lml.reshape((bank_lml.shape[0] * bank_lml.shape[1],))
    bank_gabor = bank_gabor.reshape((bank_gabor.shape[0] * bank_gabor.shape[1],))

    ## Final filter bank of the system
    bank = np.hstack((bank_gauss, bank_lms, bank_lml, bank_gabor))
    # bank = bank_gauss
    bank = bank.reshape((bank.shape[0], 1))

    # Read the image
    # Be Careful with the images, try to call them by the location folder
    images_names = [str(i) for i in range(1, number_images + 1)]
    for image_number in range(0, len(images_names)):
        # read Image
        image = cv2.imread(images_path + "/" + images_names[image_number] + ".jpg")

        image_canny = cv2.imread(
            images_canny_path + "/" + images_names[image_number] + ".png"
        )

        image_sobel = cv2.imread(
            images_sobel_path + "/" + images_names[image_number] + ".png"
        )
        # Move the image to the gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image from cv2 to numpy
        gray_float = gray.astype(np.double)
        image_gray = np.array(gray_float)
        # Transform image from cv2 to numpy
        rgb_float = rgb.astype(np.double)
        image_rgb = np.array(rgb_float)

        # Transform image from cv2 to numpy
        canny_float = image_canny.astype(np.double)
        image_canny = np.array(canny_float)

        # Transform image from cv2 to numpy
        sobel_float = image_sobel.astype(np.double)
        image_sobel = np.array(sobel_float)

        # Filtering gray image
        image_filters = convolve2d_manual_filter_bank(image_gray, bank)

        # Save the image with its filters
        # Construct the file path
        filtered_image_texture_name = os.path.join(
            results_path, f"{images_names[image_number]}_texture.npy"
        )
        filtered_image_bridhtness_name = os.path.join(
            results_path, f"{images_names[image_number]}_brightness.npy"
        )

        filtered_image_color_name = os.path.join(
            results_path, f"{images_names[image_number]}_color.npy"
        )

        filtered_image_canny_name = os.path.join(
            results_path, f"{images_names[image_number]}_canny.npy"
        )

        filtered_image_sobel_name = os.path.join(
            results_path, f"{images_names[image_number]}_sobel.npy"
        )

        # Save the image with the different channels
        np.save(filtered_image_texture_name, image_filters)
        np.save(filtered_image_bridhtness_name, image_gray)
        np.save(filtered_image_color_name, image_rgb)
        np.save(filtered_image_canny_name, image_canny)
        np.save(filtered_image_sobel_name, image_sobel)

    # Save filter bank
    name_filter_bank = "filter_bank"
    filter_bank_path = os.path.join(results_path, f"{name_filter_bank}.npy")
    np.save(filter_bank_path, bank)

    # Save half disk filters
    filter_half_disk_1_path = os.path.join(results_path, f"{name_file_half_1}.npy")
    filter_half_disk_2_path = os.path.join(results_path, f"{name_file_half_2}.npy")
    np.save(filter_half_disk_1_path, bank_half_1)
    np.save(filter_half_disk_2_path, bank_half_2)

    return None
