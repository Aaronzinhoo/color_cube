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
"""
import csv
import glob
import os
import argparse
from shutil import copy
from PIL import Image
from pathlib import Path

from ColorCube.colorcube import ColorCube
from utils.color_functions import *


parser = argparse.ArgumentParser(description='Color Classify Images')
parser.add_argument('input_dir', type=str, help='dir containing images')
parser.add_argument('output_dir', default='./colors', type=str,
                    help='location of dir to contain color category sperated images')
parser.add_argument('--orig_dir', default=None,
                    help='dir to contain original images, if left as "" then an original image dir is not created')
parser.add_argument('--delete', action='store_true',
                    help='delete files in input after analysis')
parser.add_argument('--crop', default='twoside', choices=[None, 'twoside', 'fitted'],
                    help='crop images using twoside crop before color classificaiton')
parser.add_argument('--min_side_length', default=32, type=int,
                    help='minimum side length to accept image for color classification')
args = parser.parse_args()

BASE = Path(__file__).resolve().parent
RESIZE = 100

if __name__ == "__main__":

    # open color name/characteristics file
    with open(str(BASE / 'color_groupings' / "munsell_rgb_non_color.csv"), 'r') as my_file:
        reader = csv.reader(my_file)
        my_list = list(reader)

    # create the color subdirs
    input_dir = args.input_dir
    output_dir = args.output_dir
    orig_image_dir = args.orig_dir

    # create the output dir heirarchy if needed
    for i in range(0, len(my_list)):
        save_cropped_dirs = os.path.join(output_dir, 'cropped',
                                         os.path.basename(input_dir), str(my_list[i][0]))
        if not os.path.exists(save_cropped_dirs):
            os.makedirs(save_cropped_dirs)
        # create original image output
        if orig_image_dir is not None:
            save_orig_dirs = os.path.join(output_dir, 'orig_images',
                                          os.path.basename(input_dir), str(my_list[i][0]))
            if not os.path.exists(save_orig_dirs):
                os.makedirs(save_orig_dirs)

    # Create color cube, avoiding resulting colors that are too close to black.
    # note: this doesnt avoid these colors, just ignores them at the end!!
    cc = ColorCube(avoid_color=[0., 0., 0.])

    # read images from each of the classes dirs
    # for subdir in MASKED_SUBDIRS:
    images = []
    image_color_pair = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG'):
        images.extend(glob.glob(os.path.join(input_dir, ext)))

    for img_path in images:
        # Load image and scale down to make the algorithm faster.
        # Scaling down also gives colors that are more dominant in perception.
        img_name = img_path.split('/')[-1]
        image = cv2.imread(img_path)

        # get cropped image
        if args.crop:
            try:
                image = get_largest_bbox(image)
            except Exception as e:
                print("Error: ", e, img_path)
                if args.delete:
                    os.remove(os.path.join(input_dir, img_name))
                continue
            if image.shape[0] < args.min_side_length and image.shape[1] < args.min_side_length and args.delete:
                os.remove(os.path.join(input_dir, img_name))
                continue
            elif args.crop == 'twoside':
                image = iterative_twoside_crop(image, min_crop=50, boundary_thresh=0.0)
        # resize image to RESIZExRESIZE and change to PIL for colorcube
        image = cv2.resize(image, (RESIZE, RESIZE))
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masked_src = os.path.join(input_dir, img_name)
        # Get colors for image
        colors = cc.get_colors(image)
        if not len(colors) and args.delete:
            os.remove(masked_src)
            continue
        elif not len(colors):
            continue

        # get index and name of color for image
        min_index = get_color_index(colors, my_list)
        color = str(my_list[min_index][0])

        # save image to output dirs

        clothing_label = os.path.basename(input_dir)
        masked_output_dir = os.path.join(output_dir, 'cropped',
                                         clothing_label, color, img_name)
        orig_output_dir = os.path.join(output_dir, 'orig_images',
                                       clothing_label, color, img_name)
        # try to save image
        try:
            copy(masked_src, masked_output_dir)
            image_color_pair.append((masked_output_dir, img_name, color))
            if orig_image_dir is not None:
                copy(orig_image_dir + '/{}'.format(img_name), orig_output_dir)
        except Exception as e:
            print(e)
        # delete source if arg set
        if args.delete:
            os.remove(masked_src)
