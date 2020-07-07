import cv2
import numpy as np


class ImageCrop(object):
    def __init__(
        self, min_side_length, iterator_size=1, min_crop=50, boundary_threshold=0.0
    ):
        self.iterator_size = iterator_size
        self.min_side_length = min_side_length
        self.min_crop = min_crop
        self.boundary_threshold = boundary_threshold

    def crop_image(self, image):
        try:
            image = self.get_largest_bbox(image)
        except Exception as e:
            print("Error: ", e, img_path)
            return None
        if (
            image.shape[0] < self.min_side_length
            and image.shape[1] < self.min_side_length
        ):
            image = None
        else:
            image = self.iterative_twoside_crop(
                image, self.iterator_size, self.min_crop, self.boundary_threshold
            )
        return image

    @staticmethod
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
        return image[y : y + h, x : x + w, :]

    @staticmethod
    def iterative_twoside_crop(
        image, iterator=1, min_crop=(80, 80), boundary_thresh=0.1
    ):
        """
        Crop image by iteratively cropping 2 sides at a time (left or right side and top or bottom side)
            Until the image size == min_crop or threshold of black pixels is met
                for each pair of sides, choose:
                - side with largest amount of black pixels
                - crop this side out
                - set cropped -> True
                check if crop made:
                - if crop made, rerun the loop
                - else break and return image cropped at the new dimensions
        Args:
            image: image in form of nd.array
            iterator: size of crop made on given side
            min_crop: minimum size image can be cropped to
            boundary_thresh: percentage of black pixels allowed on boundary of new cropped image
        """
        assert isinstance(
            min_crop, (int, list, tuple)
        ), "min_crop must be a int value or list"
        if isinstance(min_crop, int):
            temp = min_crop
            min_crop = [temp, temp]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        y_lim, x_lim = image.shape[0], image.shape[1]
        x_top, y_top = 0, 0
        y_bot, x_bot = y_lim - 1, x_lim - 1
        min_rectangle = False
        cropped = False
        while not min_rectangle:
            # and top_row > bp_threshold*x_top-x_bot applies black pixel threshold
            top_row = np.sum(img_gray[y_top, x_top:x_bot] == 0)
            bot_row = np.sum(img_gray[y_bot, x_top:x_bot] == 0)
            left_col = np.sum(img_gray[y_top:y_bot, x_top] == 0)
            right_col = np.sum(img_gray[y_top:y_bot, x_bot] == 0)
            row_index = np.argmax([top_row, bot_row])
            col_index = np.argmax([left_col, right_col])

            if (
                not (
                    y_bot - y_top - 1 <= min_crop[0]
                    or boundary_thresh * (x_bot - x_top) >= top_row
                )
                and row_index == 0
            ):
                # print("increasing top_row")
                y_top += iterator
                cropped = True
            if (
                not (
                    y_bot - y_top - 1 <= min_crop[0]
                    or boundary_thresh * (x_bot - x_top) >= bot_row
                )
                and row_index == 1
            ):
                # print("decreasing bot row")
                y_bot -= iterator
                cropped = True
            if (
                not (
                    x_bot - x_top - 1 <= min_crop[1]
                    or boundary_thresh * (y_bot - y_top) >= left_col
                )
                and col_index == 0
            ):
                # print("increasing left col")
                x_top += iterator
                cropped = True
            if (
                not (
                    x_bot - x_top - 1 <= min_crop[1]
                    or boundary_thresh * (y_bot - y_top) >= right_col
                )
                and col_index == 1
            ):
                # print("decreasing right col")
                x_bot -= iterator
                cropped = True
            if cropped:
                cropped = False
            else:
                min_rectangle = True
        return image[y_top:y_bot, x_top:x_bot, :]
