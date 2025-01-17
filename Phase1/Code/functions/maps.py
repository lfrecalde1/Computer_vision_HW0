import os
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from .basic_operations import plot_filtered_images
from scipy.ndimage import convolve


def color(number_images, K):
    print(
        "-------------------------Computing Color------------------------------------"
    )
    # type
    type_name = "color"
    # Image Name
    # Path to load filtered image
    results_path = os.path.join(os.getcwd(), "Filter_bank")

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]

    # Results texton
    results_texton_path = os.path.join(os.getcwd(), "Color_map")

    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        # Save the image with the different channels
        loaded_image_filter = np.load(filtered_image_name)
        # plt.imshow(loaded_image_filter / 255, interpolation="none")
        # plt.show()

        # Resize the image
        Image_flat = loaded_image_filter.reshape((-1, 3))

        # Move the image to float32
        Image_flat = np.float32(Image_flat)

        # Criteria for Kmeans
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

        # Computing clusters
        difference_cluster, texton, cluster_centers = cv2.kmeans(
            Image_flat, K, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Check the accuracy of the classification
        print(difference_cluster)

        # Reshape
        texton = texton.reshape(
            loaded_image_filter.shape[0], loaded_image_filter.shape[1]
        )

        empty_matrix[image_number, 0] = texton

        # Convert image to opencv
        texton_cv = cv2.convertScaleAbs(texton)

        # Show image
        plt.imshow(texton, cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.pdf"
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

    return empty_matrix


def color_images(number_images, K, results_path, results_texton_path):
    print(
        "-------------------------Computing Color------------------------------------"
    )
    # type
    type_name = "color"
    # Image Name
    # Path to load filtered image
    results_path = os.path.join(os.getcwd(), "Filter_bank")

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]
    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        # Save the image with the different channels
        loaded_image_filter = np.load(filtered_image_name)
        # plt.imshow(loaded_image_filter / 255, interpolation="none")
        # plt.show()

        # Resize the image
        Image_flat = loaded_image_filter.reshape((-1, 3))

        # Move the image to float32
        Image_flat = np.float32(Image_flat)

        # Criteria for Kmeans
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

        # Computing clusters
        difference_cluster, texton, cluster_centers = cv2.kmeans(
            Image_flat, K, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Check the accuracy of the classification
        print(difference_cluster)

        # Reshape
        texton = texton.reshape(
            loaded_image_filter.shape[0], loaded_image_filter.shape[1]
        )

        empty_matrix[image_number, 0] = texton

        # Convert image to opencv
        texton_cv = cv2.convertScaleAbs(texton)

        # Show image
        plt.imshow(texton, cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.pdf"
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

        file_path_data = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.npy"
        )
        np.save(file_path_data, texton)

    return empty_matrix


def brightness(number_images, K):
    print(
        "-------------------------Computing Brightness------------------------------------"
    )
    # type
    type_name = "brightness"
    # Image Name
    # Path to load filtered image
    results_path = os.path.join(os.getcwd(), "Filter_bank")

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]

    # Results texton
    results_texton_path = os.path.join(os.getcwd(), "Brightness_map")

    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        # Save the image with the different channels
        loaded_image_filter = np.load(filtered_image_name)

        # Resize the image
        Image_flat = loaded_image_filter.reshape((-1, 1))

        # Move the image to float32
        Image_flat = np.float32(Image_flat)

        # Criteria for Kmeans
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

        # Computing clusters
        difference_cluster, texton, cluster_centers = cv2.kmeans(
            Image_flat, K, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Check the accuracy of the classification
        print(difference_cluster)

        # Reshape
        texton = texton.reshape(
            loaded_image_filter.shape[0], loaded_image_filter.shape[1]
        )

        empty_matrix[image_number, 0] = texton

        # Convert image to opencv
        texton_cv = cv2.convertScaleAbs(texton)

        # Show image
        plt.imshow(texton, cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.pdf"
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

    return empty_matrix


def brightness_images(number_images, K, results_path, results_texton_path):
    print(
        "-------------------------Computing Brightness------------------------------------"
    )
    # type
    type_name = "brightness"
    # Image Name

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]
    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        # Save the image with the different channels
        loaded_image_filter = np.load(filtered_image_name)

        # Resize the image
        Image_flat = loaded_image_filter.reshape((-1, 1))

        # Move the image to float32
        Image_flat = np.float32(Image_flat)

        # Criteria for Kmeans
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

        # Computing clusters
        difference_cluster, texton, cluster_centers = cv2.kmeans(
            Image_flat, K, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Check the accuracy of the classification
        print(difference_cluster)

        # Reshape
        texton = texton.reshape(
            loaded_image_filter.shape[0], loaded_image_filter.shape[1]
        )

        empty_matrix[image_number, 0] = texton

        # Convert image to opencv
        texton_cv = cv2.convertScaleAbs(texton)

        # Show image
        plt.imshow(texton, cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.pdf"
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

        file_path_data = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.npy"
        )
        np.save(file_path_data, texton)

    return empty_matrix


def texton_images(number_images, K, results_path, results_texton_path):
    print(
        "-----------------------------------Computing Texton Texture-----------------------------"
    )
    # type
    type_name = "texture"
    # Image Name

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]

    # Load Filter
    name_filter_bank = "filter_bank"
    filter_bank_path = os.path.join(results_path, f"{name_filter_bank}.npy")
    loaded_filter_bank = np.load(filter_bank_path, allow_pickle=True)

    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_texture_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        # Save the image with the different channels
        loaded_image_filter = np.load(filtered_image_texture_name)
        # Resize the image
        Image_flat = loaded_image_filter.reshape((-1, loaded_filter_bank.shape[0]))

        # Move the image to float32
        Image_flat = np.float32(Image_flat)

        # Criteria for Kmeans
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

        # Computing clusters
        difference_cluster, texton, cluster_centers = cv2.kmeans(
            Image_flat, K, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Check the accuracy of the classification
        print(difference_cluster)

        # Reshape
        texton = texton.reshape(
            loaded_image_filter.shape[0], loaded_image_filter.shape[1]
        )

        empty_matrix[image_number, 0] = texton

        # Convert image to opencv
        # texton_cv = cv2.convertScaleAbs(texton)

        # Show image
        plt.imshow(texton, cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.pdf"
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

        file_path_data = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.npy"
        )
        np.save(file_path_data, texton)

    return empty_matrix


def gradient_images(number_images, K, results_path, results_texton_path, type_name):
    aux_print = "Computing"
    type_name_gradient = type_name + "_" + "gradient"

    print(
        "--------------------------"
        + aux_print
        + " "
        + type_name_gradient
        + "------------------------------------"
    )

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]

    ## Path to load the halfdisk filters
    filter_path = os.path.join(os.getcwd(), "Filter_bank")
    name_file_half_1 = "HDMask_1"
    name_file_half_2 = "HDMask_2"

    filter_half_disk_1_path = os.path.join(filter_path, f"{name_file_half_1}.npy")
    filter_half_disk_2_path = os.path.join(filter_path, f"{name_file_half_2}.npy")

    half_disk_1 = np.load(filter_half_disk_1_path, allow_pickle=True)
    half_disk_2 = np.load(filter_half_disk_2_path, allow_pickle=True)

    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_texture_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        loaded_image_filter = np.load(filtered_image_texture_name)
        gradient = chi_square_distance(loaded_image_filter, half_disk_1, half_disk_2, K)

        empty_matrix[image_number, 0] = gradient

        # Show image
        plt.imshow(empty_matrix[image_number, 0], cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name_gradient})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path,
            f"{images_names[image_number]}_{type_name_gradient}.pdf",
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

        file_path_data = os.path.join(
            results_texton_path,
            f"{images_names[image_number]}_{type_name_gradient}.npy",
        )
        np.save(file_path_data, empty_matrix[image_number, 0])

    return empty_matrix


def texton(number_images, K):
    print(
        "-----------------------------------Computing Texton Texture-----------------------------"
    )
    # type
    type_name = "texture"
    # Image Name
    # Path to load filtered image
    results_path = os.path.join(os.getcwd(), "Filter_bank")

    # Index images
    images_names = [str(i) for i in range(1, number_images + 1)]

    # Load Filter
    name_filter_bank = "filter_bank"
    filter_bank_path = os.path.join(results_path, f"{name_filter_bank}.npy")
    loaded_filter_bank = np.load(filter_bank_path, allow_pickle=True)

    # Results texton
    results_texton_path = os.path.join(os.getcwd(), "Texture_map")

    empty_matrix = np.empty((len(images_names), 1), dtype=object)

    # For loop all images
    for image_number in range(0, len(images_names)):
        filtered_image_texture_name = os.path.join(
            results_path, f"{images_names[image_number]}_{type_name}.npy"
        )

        # Save the image with the different channels
        loaded_image_filter = np.load(filtered_image_texture_name)
        # Resize the image
        Image_flat = loaded_image_filter.reshape((-1, loaded_filter_bank.shape[0]))

        # Move the image to float32
        Image_flat = np.float32(Image_flat)

        # Criteria for Kmeans
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

        # Computing clusters
        difference_cluster, texton, cluster_centers = cv2.kmeans(
            Image_flat, K, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Check the accuracy of the classification
        print(difference_cluster)

        # Reshape
        texton = texton.reshape(
            loaded_image_filter.shape[0], loaded_image_filter.shape[1]
        )

        empty_matrix[image_number, 0] = texton

        # Convert image to opencv
        # texton_cv = cv2.convertScaleAbs(texton)

        # Show image
        plt.imshow(texton, cmap="viridis", interpolation="none")
        plt.title(f"Filter ({type_name})")  # Title indicating position
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Adjust layout
        plt.tight_layout()

        # Construct the file path
        file_path = os.path.join(
            results_texton_path, f"{images_names[image_number]}_{type_name}.pdf"
        )

        # Save the figure
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

        # Clear the figure to prevent overlaps
        plt.clf()
        plt.close()

    return empty_matrix


def maps_images(number_images, K_texture, K_brightness, K_color):
    results_maps_path = os.path.join(os.getcwd(), "Maps")
    name = "images_maps"

    # K means Texton
    texton_images = texton(number_images, K_texture)

    # K means Brightness
    bridghtness_images = brightness(number_images, K_brightness)

    # K means Color
    color_images = color(number_images, K_color)

    full_filter = np.hstack((texton_images, bridghtness_images, color_images))

    plot_filtered_images(full_filter, name, results_maps_path)

    filter_bank_path = os.path.join(results_maps_path, f"{name}.npy")
    np.save(filter_bank_path, full_filter)


def chi_square_distance(image, half_disk_1, half_disk_2, K):
    # Filters
    left_mask = half_disk_1
    right_mask = half_disk_2

    # Aux image
    aux_image = np.zeros(image.shape)

    K_values = [i for i in range(0, K)]
    K_values = np.array(K_values)

    # Distances of all filters
    distances = np.zeros((image.shape[0], image.shape[1], half_disk_1.shape[0]))

    for index in range(0, half_disk_1.shape[0]):
        distance = np.zeros(image.shape)
        for k in range(0, K_values.shape[0]):
            aux_image[image == K_values[k]] = 1
            left = convolve(aux_image, left_mask[index])
            right = convolve(aux_image, right_mask[index])
            distance = distance + (left - right) ** 2 / (left + right + np.exp(-7))

        distance = distance / 2
        distances[:, :, index] = distance

    mean = np.mean(distances, axis=2)
    return mean


def maps_images_gradient(number_images, K):
    # Load maps of the images
    images_maps_path = os.path.join(os.getcwd(), "Maps")
    images_maps_gradient_path = os.path.join(os.getcwd(), "Maps_gradient")

    name = "images_maps"
    name_gradient = "images_gradient"

    filter_bank_path = os.path.join(images_maps_path, f"{name}.npy")
    map_images_data = np.load(filter_bank_path, allow_pickle=True)

    ## Path to load the halfdisk filters
    filter_path = os.path.join(os.getcwd(), "Filter_bank")
    name_file_half_1 = "Half_disk_1"
    name_file_half_2 = "Half_disk_2"

    filter_half_disk_1_path = os.path.join(filter_path, f"{name_file_half_1}.npy")
    filter_half_disk_2_path = os.path.join(filter_path, f"{name_file_half_2}.npy")

    half_disk_1 = np.load(filter_half_disk_1_path, allow_pickle=True)
    half_disk_2 = np.load(filter_half_disk_2_path, allow_pickle=True)

    # Computing chi square
    empty_matrix = np.empty(
        (map_images_data.shape[0], map_images_data.shape[1]), dtype=object
    )

    for j in range(0, map_images_data.shape[0]):
        print(
            "---------------------- Computing chi distance of all images -------------------------"
        )
        for i in range(0, map_images_data.shape[1]):
            empty_matrix[j, i] = chi_square_distance(
                map_images_data[j, i], half_disk_1, half_disk_2, K[i]
            )

    plot_filtered_images(empty_matrix, name_gradient, images_maps_gradient_path)
    filter_chi_path = os.path.join(images_maps_gradient_path, f"{name_gradient}.npy")
    np.save(filter_chi_path, empty_matrix)

    return None


def pb_algorithm(number_images):
    results_path = os.path.join(os.getcwd(), "Filter_bank")

    name_canny = "canny"
    name_sobel = "sobel"

    images_names = [str(i) for i in range(1, number_images + 1)]

    # Read gradients
    images_maps_gradient_path = os.path.join(os.getcwd(), "Maps_gradient")
    name_gradient = "images_gradient"
    filter_chi_path = os.path.join(images_maps_gradient_path, f"{name_gradient}.npy")

    gradients = np.load(filter_chi_path, allow_pickle=True)

    for image_number in range(0, len(images_names)):

        canny_path_name = os.path.join(
            results_path, f"{images_names[image_number]}_{name_canny}.npy"
        )

        sobel_path_name = os.path.join(
            results_path, f"{images_names[image_number]}_{name_sobel}.npy"
        )

        load_canny_filter = np.load(canny_path_name)
        load_sobel_filter = np.load(sobel_path_name)

        load_canny_filter = load_canny_filter[:, :, 0]
        load_sobel_filter = load_sobel_filter[:, :, 0]

        # Texture
        texture_gradient = 255 * (gradients[image_number, 0] / 1)
        brightness_gradient = 255 * (gradients[image_number, 1])
        color_gradient = 255 * (gradients[image_number, 2])

        pb_edges = ((texture_gradient + brightness_gradient + color_gradient) / 3) * (
            load_canny_filter * 0.5 + load_sobel_filter * 0.5
        )

        abs_grad_x = cv2.convertScaleAbs(pb_edges)

        window_name = "Sobel Demo - Simple Edge Detector"
        cv2.imshow(window_name, abs_grad_x)
        cv2.waitKey(0)

        plt.imshow(pb_edges, interpolation="none")
        plt.show()
    return None
