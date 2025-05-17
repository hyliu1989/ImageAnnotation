# -*- coding: utf-8 -*-
"""

Starter code for point selecting under IPython notebook environment

IPython version 4.0.0
Python version 3.4.3

"""
import cv2 as cv
import IPython
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.widgets import Button

UseNotebook = False

__all__ = [
    "get_control_points", "get_control_points_1img", "drag_control_points", "save_points", "load_points", "show_point"
    "ImageStreamClickAnnotator"
]


def get_control_points(im1, im2, style='g.', **kw):
    """
    Please hit a pair of points on the two displayed images, and then the next pair.
    The hit points will be saved and output as tuple of point lists
    (points1, points2) where points1=[(x1_1,y1_1), (x2_1,y2_1), (x3_1,y3_1), ...]
                             points2=[(x1_2,y1_2), (x2_2,y2_2), (x3_2,y3_2), ...]
    """
    if UseNotebook:
        ip = IPython.get_ipython()
        ip.magic('pylab')
    
    if 'figsize' in kw.keys():
        fh = plt.figure('Hit enter to terminate', figsize=kw['figsize'])
    else:
        fh = plt.figure('Hit enter to terminate')
        
    ah1 = fh.add_subplot(121)
    ah2 = fh.add_subplot(122)
    ah1.imshow(im1, cmap='gray')
    ah2.imshow(im2, cmap='gray')
    temp = ah1.axis()
    ah1.set_xlim(temp[0:2])
    ah1.set_ylim(temp[2:4])
    temp = ah2.axis()
    ah2.set_xlim(temp[0:2])
    ah2.set_ylim(temp[2:4])
    fh.tight_layout()
    fh.canvas.draw()
    
    pts1 = []
    pts2 = []
    try:
        while True:
            p1p2 = fh.ginput(2, timeout=-1)
            print(p1p2)
            if len(p1p2) != 2:
                break
            pts1.append(p1p2[0])
            pts2.append(p1p2[1])
            
            #         [     x     ], [     y     ], styles...
            ah1.plot([pts1[-1][0]], [pts1[-1][1]], style, markersize=6)
            ah2.plot([pts2[-1][0]], [pts2[-1][1]], style, markersize=6)
            # force drawing
            fh.canvas.draw()
    except KeyboardInterrupt:
        pass

    if UseNotebook:
        ip.magic('pylab inline')
    
    return np.array(pts1), np.array(pts2)


def get_control_points_1img(im1, style='g.', **kw):
    """
    Hit the point and it will be displayed. All the hit points will be displayed.
    """
    if UseNotebook:
        ip = IPython.get_ipython()
        ip.magic('pylab')
    
    if 'figsize' in kw.keys():
        fh = plt.figure('Close window or hit enter to terminate', figsize=kw['figsize'])
    else:
        fh = plt.figure('Close window or hit enter to terminate')
        
    ah1 = fh.add_subplot(111)
    ah1.imshow(im1, cmap='gray')
    temp = ah1.axis()
    ah1.set_xlim(temp[0:2])
    ah1.set_ylim(temp[2:4])
    
    pts1 = []
    while True:
        p1p2 = fh.ginput(1, timeout=-1)
        if len(p1p2) != 1: break
        pts1.append(p1p2[0])
        
        #         [     x     ], [     y     ], styles...
        ah1.plot( [pts1[-1][0]], [pts1[-1][1]], style, markersize=6 )
        # force drawing
        fh.canvas.draw()

    if UseNotebook:
        ip.magic('pylab inline')
    
    return np.array(pts1)


def drag_control_points(img, cpts, style='g.'):
    """
    Give an initial set of control points; then you can move the points around to their desired positions.
    cpts should be in the shape of [# of points, 2] where 2 stands for x and y coordinates. x is 0, y is 1.
    """
    if UseNotebook:
        ip = IPython.get_ipython()
        ip.magic('pylab')

    cpts = cpts.copy()
    scale = (img.shape[0]**2 + img.shape[1]**2)**0.5/20
    fh = plt.figure('Close window to terminate')
    ah = fh.add_subplot(111)
    ah.imshow(img, cmap='gray')
    temp = ah.axis()
    ah.set_xlim(temp[0:2])
    ah.set_ylim(temp[2:4])
    lh = [None]
    lh[0] = ah.plot(cpts[:,0], cpts[:,1], style)[0]

    idx = [None]
    figure_exist = [True]

    def on_press(event):
        diff = np.abs(np.array([[event.xdata, event.ydata]]) - cpts).sum(axis=(1,))
        idx[0] = np.argmin(diff)
        if diff[idx[0]] > scale:
            idx[0] = None
        else:
            temp_cpts = np.delete(cpts, idx[0], axis=0)
            lh[0].remove()
            lh[0] = ah.plot(temp_cpts[:,0], temp_cpts[:,1], style)[0]
            fh.canvas.draw()

    def on_release(event):
        if idx[0] != None:
            cpts[idx[0], 0] = event.xdata
            cpts[idx[0], 1] = event.ydata
            lh[0].remove()
            lh[0] = ah.plot(cpts[:,0], cpts[:,1], style)[0]
            fh.canvas.draw()

    def handle_close(event):
        print("exit the handle close aha")
        figure_exist[0] = False

    fh.canvas.mpl_connect('close_event', handle_close)
    fh.canvas.mpl_connect('button_press_event', on_press)
    fh.canvas.mpl_connect('button_release_event', on_release)

    fh.show()

    while figure_exist[0]:
        plt.waitforbuttonpress()

    if UseNotebook:
        ip.magic('pylab inline')

    return cpts


def save_points(filename, pointarray):
    if len(pointarray.shape) == 2 and pointarray.shape[1] == 2:
        pass
    else:
        raise TypeError('Point array should be n-by-2 with the first column for coordinate x and second for y.')
    
    np.save(filename, pointarray)
    return None


def load_points(filename):
    return np.load(filename)


def show_point(ah, im, cpts, lineshape='g*'):
    # fh = plt.figure(figsize=(14,14))
    # ah = fh.add_subplot(111)
    ah.imshow(im)
    temp = ah.axis()
    ah.set_xlim(temp[0:2])
    ah.set_ylim(temp[2:4])
    ah.plot(cpts[:,0],cpts[:,1], lineshape, markersize=8)


class ImageStreamClickAnnotator:
    """Allows the user to click through a list of images for corresponding points in an image sequence."""
    def __init__(self, image_paths, max_points=10):
        self.image_paths = image_paths
        self.max_points = max_points
        self.annotations = {}
        self._image_index = 0
        self._selected_point_index = 0
        self._points = [None] * max_points

        self._fig, self._ax = plt.subplots()
        self._img_plot = None
        self._graphics = {"scatter": None, "texts": []}
        self._cids = {}

        # Used in keeping the zoom from previous image.
        self._prev_xlim = None
        self._prev_ylim = None

    def _draw_annotations(self):
        # Clear previous annotations
        for txt in self._graphics["texts"]:
            txt.remove()
        self._graphics["texts"].clear()

        # Filter valid points and update plot
        valid_points = [(pt[0], pt[1]) for pt in self._points if pt is not None]
        indices = [i for i, pt in enumerate(self._points) if pt is not None]

        if self._graphics["scatter"]:
            self._graphics["scatter"].remove()

        if valid_points:
            x, y = zip(*valid_points)
            self._graphics["scatter"] = self._ax.scatter(x, y, c='lime', s=40)
            for i, (pt_x, pt_y) in zip(indices, valid_points):
                self._graphics["texts"].append(self._ax.text(pt_x + 3, pt_y - 3, str(i), color='lime', fontsize=9))

    def _collect(self):
        path = self.image_paths[self._image_index]
        self.annotations[path] = self._points.copy()

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return
        x, y = event.xdata, event.ydata
        self._points[self._selected_point_index] = (x, y)
        self._draw_annotations()
        self._fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in map(str, range(10)):
            # Change the current point focus
            self._selected_point_index = int(event.key)
        elif event.key == 'enter':
            # Clean up the current image tasks
            self._collect()
            # Move to the next image
            self._image_index += 1
            if self._image_index < len(self.image_paths):
                self._load_image()
            else:
                print("Annotation complete.")
                plt.close()
        elif event.key == 'escape':
            print("Aborted.")
            plt.close()

    def _load_image(self):
        # Store current view limits before clearing
        if self._img_plot is not None:
            self._prev_xlim = self._ax.get_xlim()
            self._prev_ylim = self._ax.get_ylim()

        self._points = [None] * self.max_points
        img = mpimg.imread(self.image_paths[self._image_index])

        # Clear the previous image and registered graphics.
        self._ax.clear()
        self._graphics["scatter"] = None
        self._graphics["texts"].clear()

        self._img_plot = self._ax.imshow(img)
        self._ax.set_title(
            f"Image {self._image_index + 1}/{len(self.image_paths)}: {self.image_paths[self._image_index]}")
        self._draw_annotations()

        # Restore previous zoom/pan if available
        if self._prev_xlim and self._prev_ylim:
            self._ax.set_xlim(self._prev_xlim)
            self._ax.set_ylim(self._prev_ylim)

        self._fig.canvas.draw_idle()

    def run(self):
        self._load_image()
        self._cids["click"] = self._fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._cids["key"] = self._fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._cids["close"] = self._fig.canvas.mpl_connect('close_event', lambda event: self._collect())
        plt.show()

        # Output after completion
        print("Final Annotations:")
        for path, pts in self.annotations.items():
            print(f"{path}: {pts}")


if __name__ == '__main__':
    """
    This part is meant to demonstrate how to use the point selecting function under IPython notebook environment.
    This part can be put into IPython notebook after uncommenting the first line and 
    changing get_control_points to imagePointsInput.get_control_points.
    """
    im1 = np.zeros((900,900))
    im2 = np.zeros((950,950))
    im2[-1,:] = 1
    im1[-1,:] = 1
    x1,x2 = get_control_points(im1, im2)
    print(x1)


    """
    This part demonstrates how to use drag_control_points
    """
    img = np.zeros((100, 200, 3), dtype=np.float64)
    cpts = np.array([[10, 10],
                     [20, 20],
                     [10, 20],
                     [50, 50],
                     [20, 40],], dtype=np.float64)
    drag_control_points(img, cpts)
