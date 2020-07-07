import cv2
import math
import numpy as np


def get_largest_bbox(image):
    """
    returns cropped image that fits largest contour found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_mask = cv2.inRange(img_gray, 1, 255)
    contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    x, y, w, h = cv2.boundingRect(biggest_contour)
    return image[y:y+h, x:x+w, :]

def iterative_twoside_crop(image, iterator=1, min_crop=(80, 80), boundary_thresh=0.1):
    """
    Crop image by iteratively cropping 2 sides at a time (left or right side and top or bottom side)
        Until the image size == min_crop or threshold of black pixels is met 
            for each pair of sides, choose:
            - side with largest amount of black pixels
            - crop this side out
            check if we made a crop:
            - if crop made, rerun the loop
            - else break and return image cropped at the new dimensions
    Args:
        image: image in form of nd.array
        iterator: size of crop made on given side
        min_crop: minimum size image can be cropped to
        boundary_thresh: percentage of black pixels allowed on boundary of new cropped image
    """
    assert isinstance(min_crop, (int, list, tuple)),"min_crop must be a int value or list"
    if isinstance(min_crop, int):
        temp = min_crop
        min_crop = [temp, temp]    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    y_lim, x_lim = image.shape[0], image.shape[1]
    x_top, y_top = 0, 0
    y_bot, x_bot = y_lim-1, x_lim-1
    min_rectangle = False
    cropped=False
    while not min_rectangle:
        # and top_row > bp_threshold*x_top-x_bot applies black pixel threshold
        top_row = np.sum(img_gray[y_top, x_top:x_bot] == 0)
        bot_row = np.sum(img_gray[y_bot, x_top:x_bot] == 0)
        left_col = np.sum(img_gray[y_top:y_bot, x_top] == 0)
        right_col = np.sum(img_gray[y_top:y_bot, x_bot] == 0)
        row_index = np.argmax([top_row, bot_row])
        col_index = np.argmax([left_col, right_col])
        
        if not (y_bot-y_top-1 <= min_crop[0] or boundary_thresh*(x_bot-x_top) >= top_row) and row_index == 0:
            # print("increasing top_row")
            y_top += iterator
            cropped = True
        if not (y_bot-y_top-1 <= min_crop[0] or boundary_thresh*(x_bot-x_top) >= bot_row) and row_index == 1:
            # print("decreasing bot row")
            y_bot -= iterator
            cropped = True
        if not (x_bot-x_top-1 <= min_crop[1] or boundary_thresh*(y_bot-y_top) >= left_col) and col_index == 0:
            # print("increasing left col")
            x_top += iterator
            cropped = True
        if not (x_bot-x_top-1 <= min_crop[1] or boundary_thresh*(y_bot-y_top) >= right_col) and col_index == 1:
            # print("decreasing right col")
            x_bot -= iterator
            cropped = True
        if cropped:
            cropped = False
        else:
            min_rectangle = True
    return image[y_top:y_bot, x_top:x_bot, :]


def get_color_index(colors, color_list):
    """
    Given A set of colors, return the index of closest munsell color
        If colors is empty return None
        Else ret index of closest color
        If color close to black, white, or RGB colors < 10 away from each other
            Assign color and return
        Closest found by comparing distances in RGB color space
    """
    min_distance = 100000000000
    min_index = None

    # Print first four colors (there might be much more)
    # check if the color is either white, black, light grey or dark grey
    # if it is, then skip checking munsell colors and add this image to the dir
    for c in colors[:1]:
        # do something special for white or black
        if c[0] > 240 and c[1] > 240 and c[2] > 240:
            min_index = 145
            break
        # munsell colors dont account for colors that have all rgb values < 35
        elif c[0] <= 35 and c[1] <= 35 and c[2] <= 35:
            min_index = 144
            break
        # dark grey
        elif c[0] < 230 and c[1] < 230 and c[2] < 230 and c[0] > 168 and c[1] > 168 and c[2] > 168:
            if abs(c[0]-c[1]) < 10 and abs(c[0]-c[2]) < 10 and abs(c[1]-c[2]) < 10:
                min_index = 146
                break

        # light grey
        elif c[0] < 132 and c[1] < 132 and c[2] < 132 and c[0] > 60 and c[1] > 60 and c[2] > 60:
            if abs(c[0]-c[1]) < 10 and abs(c[0]-c[2]) < 10 and abs(c[1]-c[2]) < 10:
                min_index = 147
                break

        # if not generic black, white, light/dark grey
        # find closest color group
        for i in range(0, len(color_list)-2):
            a = np.array((c[0], c[1], c[2]))
            b = np.array((int(color_list[i][1]), int(
                color_list[i][2]), int(color_list[i][3])))

            r_delta = a[0] - b[0]
            g_delta = a[1] - b[1]
            b_delta = a[2] - b[2]
            # Compute delta in color space distance
            distance = math.sqrt(r_delta**2 + g_delta**2 + b_delta**2)

            if min_distance > distance:
                min_distance = distance
                min_index = i
    return min_index
