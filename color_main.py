"""
Perform color identification on images and save them to their respective color dir
    Arguments:
        input_dir - dir to read images from
        (NOTE: only takes in one directory, no recursive reading)
        output_dir - dir to write results to
        save_orig - if save_orig is true then we will also save the original files
        delete - option to delete masked images after you perform analysis on them
        (for pipeline use)
        save_color - save image attributes (color, path, name) to txt file for pair matching later
        crop - which method to crop with twoside or fullsize
        min_side_len - min length of each side of image to accept image cropped version
        image_resize - size to resize images for color cube classification
"""
from shutil import copy
from PIL import Image
from pathlib import Path

from colorcube.colorcube import ColorCube
from config.args import get_args
from config.constants import color_list, ACCEPTED_IMAGE_EXTENTIONS
from utils.color_functions import *
from crop.imagecrop import ImageCrop


def main(input_dir, output_dir, orig_dir, delete, crop, min_side_length, image_resize):

    # create the color subdirs
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    original_images_dir = Path(args.orig_dir)
    clothing_position = input_dir.name

    # create the output dir heirarchy if needed
    cropped_images_output = output_dir / "cropped" / clothing_position
    original_images_output = output_dir / "orig_images" / clothing_position
    make_color_dir_heirarchy(cropped_images_output, color_list)
    if str(original_images_dir):
        make_color_dir_heirarchy(original_images_output, color_list)

    # Create color cube, avoiding resulting colors that are too close to black.
    # note: this doesnt avoid these colors, just ignores them at the end!!
    color_cube = ColorCube(avoid_color=[0.0, 0.0, 0.0])

    # read images of accepted exts from input_dir
    images = [
        image_path
        for ext in ACCEPTED_IMAGE_EXTENTIONS
        for image_path in Path(input_dir).glob(ext)
    ]

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
                # get name of color for image from color_list
                color_name = get_nearest_color(colors, my_list)
                # save image to output dirs
                cropped_output_path = cropped_images_output / color_name
                original_output_path = original_images_output / color_name
                # try to save image
                # copy is faster than cv2.imwrite
                try:
                    copy(image_path, cropped_output_path)
                    if original_images_dir is not None:
                        copy(original_images_dir / image_name, original_output_path)
                except Exception as e:
                    print(e)
        # delete source if arg set
        if args.delete:
            image_path.unlink()


if __name__ == "__main__":
    main(**vars(get_args()))
