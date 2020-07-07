import csv
from pathlib import Path

ACCEPTED_IMAGE_EXTENTIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG")

# open color name/characteristics file
root = Path(__file__).resolve().parent.parent
with open(str(root / "color_groupings" / "munsell_rgb_non_color.csv"), "r") as my_file:
    color_list = list(csv.reader(my_file))
