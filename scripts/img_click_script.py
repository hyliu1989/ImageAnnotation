# -*- coding: utf-8 -*-
"""An example script to show how to use pointclick."""
import cv2 as cv
import matplotlib
import numpy as np

import annotateimg.pointclick as pointclick  # This might need to be moved to after matplotlib.use()

matplotlib.use("qt5agg")


left = cv.imread("left.png")
right = cv.imread("right.png")

left = cv.cvtColor(left, cv.COLOR_BGR2RGB)
right = cv.cvtColor(right, cv.COLOR_BGR2RGB)

pts1, pts2 = pointclick.get_control_points(left, right)

name_base = input("name base for saving:")

np.save(name_base + "_left_pts_xy.npy", pts1)
np.save(name_base + "_right_pts_xy.npy", pts2)
