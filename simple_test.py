import argparse
from shutil import copy
from PIL import Image
from pathlib import Path

from ColorCube.colorcube import ColorCube
from config.args import color_list, get_args, ACCEPTED_IMAGE_EXTENTIONS
from utils.color_functions import *
from crop.imagecrop import ImageCrop


def main(input_dir, output_dir, crop, min_side_length, image_resize):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_cropper = ImageCrop(min_side_length)

    # read images from each of the classes dirs
    # for subdir in MASKED_SUBDIRS:
    images = [
        image_path
        for ext in ACCEPTED_IMAGE_EXTENTIONS
        for image_path in Path(input_dir).glob(ext)
    ]

    make_color_dir_heirarchy(output_dir, color_list)

    # Create color cube, avoiding resulting colors that are too close to black.
    # note: this doesnt avoid these colors, just ignores them at the end!!
    color_cube = ColorCube(avoid_color=[0.0, 0.0, 0.0])
    image_cropper = ImageCrop(min_side_length)

    for image_path in images:
        # Load image and scale down to make the algorithm faster.
        # Scaling down also gives colors that are more dominant in perception.
        image_name = image_path.name
        image = cv2.imread(str(image_path))

        # get cropped image
        if crop:
            image = image_cropper.crop_image(image)
        if image is not None:
            # resize image to image_resizeximage_resize and change to PIL for colorcube
            image = cv2.resize(image, (image_resize, image_resize))
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Get colors for image
            colors = color_cube.get_colors(image)
            if colors:
                # get index and name of color for image
                color_name = get_nearest_color(colors, color_list)
                # try to save image
                try:
                    copy(input_dir / image_name, output_dir / color_name)
                except Exception as e:
                    print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Color Classify Images")
    parser.add_argument("input_dir", type=str, help="dir containing images")
    parser.add_argument(
        "output_dir",
        default="./colors",
        type=str,
        help="location of dir to contain color category sperated images",
    )
    parser.add_argument(
        "--orig_dir",
        default=None,
        help='dir to contain original (un-segregated) images, if left as "" then an original image dir is not created',
    )
    parser.add_argument(
        "--delete", action="store_true", help="delete files in input after analysis"
    )
    parser.add_argument(
        "--crop",
        default="twoside",
        choices=[None, "twoside", "fitted"],
        help="crop images using twoside crop before color classificaiton",
    )
    parser.add_argument(
        "--min_side_length",
        default=32,
        type=int,
        help="minimum side length to accept image for color classification",
    )
    parser.add_argument(
        "--image_resize",
        default=100,
        type=int,
        help="size of resized image to pass through color cube",
    )
    args = parser.parse_args()
    main(**vars(args))
