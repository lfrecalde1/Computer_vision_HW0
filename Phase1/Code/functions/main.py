from functions.filter_bank import filter_banks

# from maps import maps_images
# from maps import maps_images_gradient
# from maps import pb_algorithm


def main():
    # Number of image to compute
    number_images = 9

    # Cluster for texture or texton
    K_texture = 64

    # Cluster for brightness
    K_brightness = 16

    # Cluster for color
    K_color = 16

    # Vector with the clusters
    K = [K_texture, K_brightness, K_color]

    # Computing and saving filter_bank
    filter_banks(number_images)

    # Maps
    # maps_images(number_images, K_texture, K_brightness, K_color)

    # Gradient of the maps
    # maps_images_gradient(number_images, K)

    # Pb
    # pb_algorithm(number_images)

    return None


if __name__ == "__main__":
    main()
