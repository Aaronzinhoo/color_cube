import cv2
import math
import numpy as np
from enum import IntEnum
from pathlib import Path


class AchromaticColorIndex(IntEnum):
    Black = 144
    White = 145
    LightGray = 146
    DarkGrey = 147


def make_color_dir_heirarchy(output_dir, color_list):
    """ create the directory of color dirs for images to be placed in"""
    root_dir = Path(output_dir)
    for color in color_list:
        (root_dir / color[0]).mkdir(parents=True, exist_ok=True)


def is_color_gray(color, min_distance=10):
    return (
        abs(color[0] - color[1]) < min_distance
        and abs(color[0] - color[2]) < min_distance
        and abs(color[1] - color[2]) < min_distance
    )


def get_nearest_color(colors, color_list, top_k_colors=1):
    """
    Given A set of colors, return the index of closest munsell color
        If colors is empty return None
        Else ret index of closest color
        If color close to black, white, or RGB colors < 10 away from each other
            Assign color and return
        Closest found by comparing distances in RGB color space
    """
    color_name = ""
    min_index = None
    # check only the top color since rest are irrelevant (may change later if improved)
    # Print first four colors (there might be much more)
    # check if the color is either white, black, light grey or dark grey
    # if it is, then skip checking munsell colors and add this image to the dir
    for color in colors[:top_k_colors]:
        # check black
        if color[0] <= 35 and color[1] <= 35 and color[2] <= 35:
            min_index = AchromaticColorIndex.Black
        # check white
        elif color[0] > 240 and color[1] > 240 and color[2] > 240:
            min_index = AchromaticColorIndex.White
        # check light grey
        elif (
            color[0] < 230
            and color[1] < 230
            and color[2] < 230
            and color[0] > 168
            and color[1] > 168
            and color[2] > 168
        ):
            if is_color_gray(color):
                min_index = AchromaticColorIndex.LightGray

        # check dark grey
        elif (
            color[0] < 132
            and color[1] < 132
            and color[2] < 132
            and color[0] > 60
            and color[1] > 60
            and color[2] > 60
        ):
            if is_color_gray(color):
                min_index = AchromaticColorIndex.DarkGrey

        if not min_index:
            # if not generic black, white, light/dark grey
            # find closest color group
            distances = []
            color = np.array((color[0], color[1], color[2]))
            # i do not exactly remember why -2.. regardless grays are not included
            # this means black and white can still be attributed if they are not assigned
            # above
            for pccs_color in color_list[:-2]:
                # index 0 is the name of the color
                pccs_color = np.array(
                    (int(pccs_color[1]), int(pccs_color[2]), int(pccs_color[3]))
                )
                distances.append(np.linalg.norm(color - pccs_color))
            min_index = np.argmin(distances)
    if min_index is not None:
        color_name = color_list[min_index][0]
    return color_name
