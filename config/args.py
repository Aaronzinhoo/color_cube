import csv
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def get_args():
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
    return parser.parse_args()


ACCEPTED_IMAGE_EXTENTIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG")

# open color name/characteristics file
with open(str(root / "color_groupings" / "munsell_rgb_non_color.csv"), "r") as my_file:
    color_list = list(csv.reader(my_file))
