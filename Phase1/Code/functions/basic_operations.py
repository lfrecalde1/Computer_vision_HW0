import numpy as np
from scipy.ndimage import rotate, convolve
from matplotlib import pyplot as plt
import os


def convolve2d_manual(image, kernel):
    # Flip the kernel horizontally and vertically to match convolution rules
    kernel_flip_x = np.fliplr(kernel)
    kernel_flip = np.flipud(kernel_flip_x)

    # Get the dimensions of the flipped kernel
    # ------> x
    # |
    # |
    # |
    # |
    # y

    y, x = kernel_flip.shape
    edge_x = x // 2
    edge_y = y // 2

    # Pad the image to handle boundaries
    image_boundaries = np.pad(image, ((edge_y, edge_y), (edge_x, edge_x)), mode="edge")

    # Create an empty output image to store the filtered results
    filter = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)

    # Perform the convolution by sliding the kernel over the image
    for y_i in range(image.shape[0]):  # Iterate over each row of the image
        for x_i in range(image.shape[1]):  # Iterate over each column of the image
            # Extract the region of interest from the padded image
            region = image_boundaries[y_i : y_i + y, x_i : x_i + x]
            # Perform element-wise multiplication of the kernel and region, then sum the result
            filter[y_i, x_i] = np.sum(region * kernel_flip)

    # Return the filtered (convolved) image
    return filter


def gaussian_2d(size, sigma_x, sigma_y, elongation):
    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # z_x = np.ceil(3 * sigma_x)
    # z_y = np.ceil(3 * sigma_y)

    # k_x = np.arange(-z_x, z_x + 1)
    # k_y = np.arange(-z_y, z_y + 1)

    # filter = np.zeros((k_y.shape[0], k_x.shape[0]), dtype=np.float64)

    # Compute the Gaussian filter using for loops
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the filter
            y = i - size // 2
            x = j - size // 2
            # Apply the Gaussian function
            filter[i, j] = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
                -((x) ** 2 / (2 * sigma_x**2)) - ((y) ** 2 / (2 * sigma_y**2))
            )

    filter /= np.sum(filter)
    return filter


def half__disk_2d_positive(size, sigma_x, sigma_y, elongation):
    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # Compute the Gaussian filter using for loops
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the filter
            y = i - size // 2
            x = j - size // 2

            # Euclidean distance square center
            distance = np.sqrt(x**2 + y**2)
            if distance < size // 2:
                if x < 0:
                    filter[i, j] = 1.0
            else:
                filter[i, j] = 0.0

    filter /= np.sum(filter)
    return filter


def half__disk_2d_negative(size, sigma_x, sigma_y, elongation):
    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # Compute the Gaussian filter using for loops
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the filter
            y = i - size // 2
            x = j - size // 2

            # Euclidean distance square center
            distance = np.sqrt(x**2 + y**2)
            if distance < size // 2:
                if x > 0:
                    filter[i, j] = 1.0
            else:
                filter[i, j] = 0.0

    filter /= np.sum(filter)
    return filter


def gabor_2d(size, sigma_x, sigma_y, elongation):
    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # Compute the Gaussian filter using for loops
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the filter
            y = i - size // 2
            x = j - size // 2
            # Apply the Gaussian function
            filter[i, j] = (
                (1 / (2 * np.pi * sigma_x * sigma_y))
                * np.exp(-((x) ** 2 / (2 * sigma_x**2)) - ((y) ** 2 / (2 * sigma_y**2)))
                * np.cos(2 * np.pi * 0.15 * x)
            )

    norm_l2 = np.sqrt(np.sum(filter**2))
    if norm_l2 != 0:
        filter /= norm_l2

    return filter


def first_derivative_gaussian_2d_x(size, sigma_x, sigma_y, elongation):

    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # Compute the Gaussian filter using (for) loop
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the kernel
            y = i - size // 2
            x = j - size // 2

            # Apply the Gaussian function
            z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
                -((x) ** 2 / (2 * sigma_x**2)) - ((y) ** 2 / (2 * sigma_y**2))
            )
            filter[i, j] = -(x / sigma_x**2) * z

    norm_factor = np.sqrt(np.sum(filter**2))
    if norm_factor != 0:
        filter /= norm_factor
    return filter


def second_derivative_gaussian_2d_x(size, sigma_x, sigma_y, elongation):
    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # Compute the Gaussian filter using (for) loop
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the kernel
            y = i - size // 2
            x = j - size // 2

            # Apply the Gaussian function
            z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
                -((x) ** 2 / (2 * sigma_x**2)) - ((y) ** 2 / (2 * sigma_y**2))
            )

            # Second time derivative of the gaussian function
            filter[i, j] = ((x**2 / sigma_x**4) - (1 / sigma_x**2)) * z

    norm_factor = np.sqrt(np.sum(filter**2))
    if norm_factor != 0:
        filter /= norm_factor

    return filter


def laplacian_gaussian_2d(size, sigma_x, sigma_y, elongation):

    # Check for elongation
    if elongation:
        sigma_x = sigma_x
        sigma_y = 3 * sigma_x
    else:
        sigma_x = sigma_x
        sigma_y = sigma_y

    # Initialize the Gaussian filter
    filter = np.zeros((size, size), dtype=np.float64)

    # Compute the Gaussian filter using for loops
    for i in range(0, size):
        for j in range(0, size):
            # Compute the x and y coordinates relative to the center of the filter
            y = i - size // 2
            x = j - size // 2
            # Apply the Gaussian function
            z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
                -((x) ** 2 / (2 * sigma_x**2)) - ((y) ** 2 / (2 * sigma_y**2))
            )
            filter[i, j] = (
                ((x**2) / sigma_x**4) + ((y**2) / sigma_x**4) - (2 / sigma_x**2)
            ) * z
    norm_factor = np.sqrt(np.sum(filter**2))
    if norm_factor != 0:
        filter /= norm_factor
    return filter


def filter_bank(scales, angles, size, elongation, operator):

    bank = np.empty((scales.shape[0], angles.shape[0]), dtype=object)

    # Bank Filters
    for scales_index in range(0, scales.shape[0]):
        for angles_index in range(0, angles.shape[0]):
            sigma_x = scales[scales_index]
            sigma_y = scales[scales_index]
            gaussian_filter = operator(size, sigma_x, sigma_y, elongation)
            # Manual Filter
            filter_manual_gauss = gaussian_filter
            # filter_manual_image = convolve2d_manual(x1, gaussian_filter)

            # Rotate output of the gaussian
            angle_of_rotation = angles[angles_index]
            rotated_img = rotate(
                filter_manual_gauss, angle_of_rotation, reshape=False, order=5
            )
            bank[scales_index, angles_index] = rotated_img

    return bank


def filter_bank_half_disk(scales, angles, elongation, operator):

    bank = np.empty((scales.shape[0], angles.shape[0]), dtype=object)

    # Bank Filters
    for scales_index in range(0, scales.shape[0]):
        for angles_index in range(0, angles.shape[0]):
            sigma_x = scales[scales_index]
            sigma_y = scales[scales_index]
            gaussian_filter = operator(
                scales[scales_index], sigma_x, sigma_y, elongation
            )
            # Manual Filter
            filter_manual_gauss = gaussian_filter
            # filter_manual_image = convolve2d_manual(x1, gaussian_filter)

            # Rotate output of the gaussian
            angle_of_rotation = angles[angles_index]
            rotated_img = rotate(
                filter_manual_gauss, angle_of_rotation, reshape=False, order=0
            )
            bank[scales_index, angles_index] = rotated_img

    return bank


def plot_banks(bank, name, location):
    n, m = bank.shape

    # Create a figure
    fig, axs = plt.subplots(n, m, figsize=(1.3 * m, 1.3 * n))

    axs = np.ravel(axs)

    # Loop over each filter in the bank
    for i in range(n):
        for j in range(m):
            # Calculate the index in the flattened array
            subplot_idx = i * m + j

            # Current axes
            ax = axs[subplot_idx]

            # Plot the filter in grayscale
            im = ax.imshow(bank[i, j], cmap="gray", interpolation="none")

            # Remove x- and y-axis ticks for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)

    # Path
    file_path = os.path.join(location, f"{name}.pdf")

    # Save it
    plt.savefig(file_path, format="pdf", bbox_inches="tight")

    # Clear
    plt.clf()
    plt.close()

    print(f"PDF with colorbar saved to: {file_path}")

    return None


def plot_filtered_images(bank, name, location):

    n, m = bank.shape

    column_titles = ["Texture", "Brightness", "Color"]  # exactly three columns

    plt.figure(figsize=(2 * m, 1.3 * n))

    for i in range(n):
        for j in range(m):
            subplot_idx = i * m + j + 1
            plt.subplot(n, m, subplot_idx)

            # Plot each filter using the plasma colormap
            plt.imshow(bank[i, j], cmap="viridis", interpolation="none")

            # Only set a title for the top row (i == 0)
            if i == 0 and j < len(column_titles):
                plt.title(column_titles[j])

            # Remove tick marks for clarity
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

    # Adjust layout so titles and images don’t overlap
    plt.tight_layout()

    # Construct the file path and save as PDF
    file_path = os.path.join(location, f"{name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")

    # Clear the figure from memory
    plt.clf()
    plt.close()
    print(f"PDF with colorbar saved to: {file_path}")

    return None


def plot_filtered_final(bank, name, location):

    n, m = bank.shape

    column_titles = ["Sobel", "Canny", "Pb"]  # exactly three columns

    plt.figure(figsize=(2 * m, 1.3 * n))

    for i in range(n):
        for j in range(m):
            subplot_idx = i * m + j + 1
            plt.subplot(n, m, subplot_idx)

            # Plot each filter using the plasma colormap
            plt.imshow(bank[i, j], cmap="gray", interpolation="none")

            # Only set a title for the top row (i == 0)
            if i == 0 and j < len(column_titles):
                plt.title(column_titles[j])

            # Remove tick marks for clarity
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

    # Adjust layout so titles and images don’t overlap
    plt.tight_layout()

    # Construct the file path and save as PDF
    file_path = os.path.join(location, f"{name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")

    # Clear the figure from memory
    plt.clf()
    plt.close()
    print(f"PDF with colorbar saved to: {file_path}")

    return None


def convolve2d_manual_filter_bank(image, filterbank):
    image_filter = np.zeros((image.shape[0], image.shape[1], filterbank.shape[0]))
    for index in range(0, filterbank.shape[0]):
        image_filter[:, :, index] = convolve2d_manual(image, filterbank[index, 0])

    return image_filter
