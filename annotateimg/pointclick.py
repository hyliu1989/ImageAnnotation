# -*- coding: utf-8 -*-
"""Image annotation module for image pairs or image stream.

It is originally the starter code for point selecting under IPython notebook environment.

IPython version 4.0.0
Python version 3.4.3

"""
import logging

import IPython
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

UseNotebook = False
logger = logging.getLogger(__name__)

__all__ = [
    "get_control_points",
    "get_control_points_1img",
    "drag_control_points",
    "save_points",
    "load_points",
    "show_point",
    "ImageStreamClickAnnotator",
    "ImagePairClickAnnotator",
]


def get_control_points(im1, im2):
    """Please hit a pair of points on the two displayed images, and then the next pair.

    The hit points will be saved and output as tuple of point lists
    (points1, points2) where points1=[(x1_1,y1_1), (x2_1,y2_1), (x3_1,y3_1), ...]
                             points2=[(x1_2,y1_2), (x2_2,y2_2), (x3_2,y3_2), ...]
    """
    return ImagePairClickAnnotator([im1, im2]).run()


def get_control_points_1img(im1):
    """Hit the point and it will be displayed. All the hit points will be displayed."""
    return ImageStreamClickAnnotator([im1]).run()


def drag_control_points(img, cpts, style="g."):
    """Allow the users to drag the annotated points.

    Give an initial set of control points; then you can move the points around to their desired
    positions. cpts should be in the shape of [# of points, 2] where 2 stands for x and y
    coordinates. x is 0, y is 1.
    """
    if UseNotebook:
        ip = IPython.get_ipython()
        ip.magic("pylab")

    cpts = cpts.copy()
    scale = (img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5 / 20
    fh = plt.figure("Close window to terminate")
    ah = fh.add_subplot(111)
    ah.imshow(img, cmap="gray")
    temp = ah.axis()
    ah.set_xlim(temp[0:2])
    ah.set_ylim(temp[2:4])
    lh = [None]
    lh[0] = ah.plot(cpts[:, 0], cpts[:, 1], style)[0]

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
            lh[0] = ah.plot(temp_cpts[:, 0], temp_cpts[:, 1], style)[0]
            fh.canvas.draw()

    def on_release(event):
        if idx[0] is not None:
            cpts[idx[0], 0] = event.xdata
            cpts[idx[0], 1] = event.ydata
            lh[0].remove()
            lh[0] = ah.plot(cpts[:, 0], cpts[:, 1], style)[0]
            fh.canvas.draw()

    def handle_close(event):
        print("exit the handle close aha")
        figure_exist[0] = False

    fh.canvas.mpl_connect("close_event", handle_close)
    fh.canvas.mpl_connect("button_press_event", on_press)
    fh.canvas.mpl_connect("button_release_event", on_release)

    fh.show()

    while figure_exist[0]:
        plt.waitforbuttonpress()

    if UseNotebook:
        ip.magic("pylab inline")

    return cpts


def save_points(filename, pointarray):
    """Saves the points to a file."""
    if len(pointarray.shape) == 2 and pointarray.shape[1] == 2:
        pass
    else:
        raise TypeError(
            "Point array should be n-by-2 with the first column for coordinate x and second for y."
        )

    np.save(filename, pointarray)
    return None


def load_points(filename):
    """Loads points from a file."""
    return np.load(filename)


def show_point(ah, im, cpts, lineshape="g*"):
    """Shows points in a given image."""
    # fh = plt.figure(figsize=(14,14))
    # ah = fh.add_subplot(111)
    ah.imshow(im)
    temp = ah.axis()
    ah.set_xlim(temp[0:2])
    ah.set_ylim(temp[2:4])
    ah.plot(cpts[:, 0], cpts[:, 1], lineshape, markersize=8)


class ImagePairClickAnnotator:
    """Allows the user to click corresponding points in two images.

    Args:
        images: List of two image paths or two images.
        max_points: Maximum number of points to annotate for each commitment.
    """

    def __init__(self, images: list[str] | list[npt.NDArray], max_points: int = 10):
        if 0 < max_points <= 10:
            pass
        else:
            raise ValueError("max_points must be between 1 to 10.")
        self.max_points = max_points

        if isinstance(images[0], str):
            image_ids = images
            image_arrays = [mpimg.imread(path) for path in images]
        else:
            image_ids = [0, 1]
            image_arrays = images

        fig, axes = plt.subplots(1, 2)
        self._image_content = [
            {
                "id": image_ids[i],
                "array": image_arrays[i],
                "ax": axes[i],
                "img_plot": None,
                "graphics": {"scatters": [None, None], "texts": []},
            }
            for i in range(2)
        ]
        self._fig = fig
        self._cids = {}
        self._commited_points: list[list[tuple[float, float]]] = []
        self._selected_point_index = 0
        self._points: list[list[None | tuple[float, float]]] = [
            [None, None] for _ in range(max_points)
        ]

    @property
    def annotations(self):
        return {
            self._image_content[i]["id"]: [
                matched_point_pair[i] for matched_point_pair in self._commited_points
            ]
            for i in range(2)
        }

    def _get_fig_title(self):
        return (
            "Click on the images to select point pairs.  "
            f"[point index: {self._selected_point_index}]"
            "\n  Press 0-9 to select the point index."
            "\n  Press Enter to commit point pair."
            "\n  Press Esc or close window to finish."
        )

    def _clear_annotations(self):
        """Clears the current annotations in the axes."""
        for idx_image in range(2):
            graphics = self._image_content[idx_image]["graphics"]
            # Clear previous annotations
            text_annotation: list[plt.Text] = graphics["texts"]
            for txt in text_annotation:
                txt.remove()
            text_annotation.clear()
            # Clear previous scatter points
            scatter_plots: list[plt.PathCollection | None] = graphics["scatters"]
            for i, scatter in enumerate(scatter_plots):
                if scatter is not None:
                    scatter.remove()
                scatter_plots[i] = None

    def _collect(self):
        for idx_point, point_pair in enumerate(self._points):
            if (point_pair[0] is None) ^ (point_pair[1] is None):
                logger.warning(f"Point pair {idx_point} [{point_pair}] contains an unpaired point!")
                continue
            if point_pair[0] is None and point_pair[1] is None:
                # Ignore empty point pairs
                continue
            assert point_pair[0] is not None and point_pair[1] is not None
            # Commit the point pair
            self._commited_points.append([point_pair[0], point_pair[1]])
        self._points = [[None, None] for _ in range(self.max_points)]

    def _draw_annotations(self):
        """Creates the annotations based on current selections and draws them on the axes."""
        self._clear_annotations()
        ax_img0 = self._image_content[0]["ax"]
        ax_img1 = self._image_content[1]["ax"]
        graphics_img0 = self._image_content[0]["graphics"]
        graphics_img1 = self._image_content[1]["graphics"]

        # Plot commited points
        if self._commited_points:
            pts_img0, pts_img1 = zip(*self._commited_points)
            xs_img0, ys_img0 = zip(*pts_img0)
            xs_img1, ys_img1 = zip(*pts_img1)
            scatter_committed_img0 = ax_img0.scatter(xs_img0, ys_img0, c="limegreen", s=40)
            scatter_committed_img1 = ax_img1.scatter(xs_img1, ys_img1, c="limegreen", s=40)
            graphics_img0["scatters"][0] = scatter_committed_img0  # [0] for commited points
            graphics_img1["scatters"][0] = scatter_committed_img1  # [0] for commited points

        # Filter valid points and update plot
        points_img0: list[None | tuple[int, int]] = [pair[0] for pair in self._points]
        valid_points_and_indices_img0 = [
            (pt[0], pt[1], i) for i, pt in enumerate(points_img0) if pt is not None
        ]
        if valid_points_and_indices_img0:
            xs_img0, ys_img0, _ = zip(*valid_points_and_indices_img0)
            # graphics_img0["scatters"][1] for current selection
            graphics_img0["scatters"][1] = ax_img0.scatter(xs_img0, ys_img0, c="lime", s=40)
            for pt_x, pt_y, i in valid_points_and_indices_img0:
                graphics_img0["texts"].append(
                    ax_img0.text(pt_x + 3, pt_y - 3, str(i), color="lime", fontsize=9)
                )

        points_img1: list[None | tuple[int, int]] = [pair[1] for pair in self._points]
        valid_points_and_indices_img1 = [
            (pt[0], pt[1], i) for i, pt in enumerate(points_img1) if pt is not None
        ]
        if valid_points_and_indices_img1:
            xs_img1, ys_img1, _ = zip(*valid_points_and_indices_img1)
            # graphics_img1["scatters"][1] for current selection
            graphics_img1["scatters"][1] = ax_img1.scatter(xs_img1, ys_img1, c="lime", s=40)
            for pt_x, pt_y, i in valid_points_and_indices_img1:
                graphics_img1["texts"].append(
                    ax_img1.text(pt_x + 3, pt_y - 3, str(i), color="lime", fontsize=9)
                )

    def _on_click(self, event):
        for idx_image in range(2):
            if event.inaxes != self._image_content[idx_image]["ax"]:
                continue
            x, y = event.xdata, event.ydata
            self._points[self._selected_point_index][idx_image] = (x, y)
            self._draw_annotations()
            self._fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in map(str, range(10)):
            # Change the current point focus
            self._selected_point_index = int(event.key)
            self._fig.suptitle(self._get_fig_title())
        elif event.key == "enter":
            # Commit the current points
            self._collect()
            self._draw_annotations()
        elif event.key == "escape":
            logger.info("Aborted.")
            plt.close()

    def _load_image(self):
        self._points = [[None, None] for _ in range(self.max_points)]
        self._clear_annotations()
        for idx_image in range(2):
            img = self._image_content[idx_image]["array"]
            axis = self._image_content[idx_image]["ax"]
            self._image_content[idx_image]["img_plot"] = axis.imshow(img)
            self._draw_annotations()

        self._fig.canvas.draw_idle()

    def run(self):
        self._load_image()
        self._cids["click"] = self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._cids["key"] = self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._cids["close"] = self._fig.canvas.mpl_connect(
            "close_event", lambda event: self._collect()
        )
        self._fig.suptitle(self._get_fig_title())
        plt.show()

        # Output after completion
        logger.info("Final Annotations:")
        for image_id, pts in self.annotations.items():
            logger.info(f"{image_id}: {pts}")
        return self.annotations


class ImageStreamClickAnnotator:
    """Allows the user to click through a list of images for corresponding points in each image.

    The result is stored in property `annotations` of this class. The format is a dictionary
    mapping from image path to a list of points or None. The order of the points in the list
    corresponds to the key (0 to 9) used when the point was clicked.

    E.g.
    annotations = {
        "path/to/image_a.jpg": [(a_x1, a_y1), None, (a_x3, a_y4)],
        "path/to/image_b.jpg": [(b_x1, b_y1), None, (b_x3, b_y4)],
    }
    # for max_points == 3.

    Args:
        images: List of image paths or images to annotate.
        max_points: Maximum number of points to annotate per image. Default is 10.
    """

    def __init__(self, images: list[str] | list[npt.NDArray], max_points: int = 10):
        if 0 < max_points <= 10:
            pass
        else:
            raise ValueError("max_points must be between 1 to 10.")

        if isinstance(images[0], str):
            self._image_paths = images
            self._images = None
            self._image_ids = self._image_paths
        else:
            self._image_paths = None
            self._images = images
            self._image_ids = np.arange(len(images))
        self._max_num_points = max_points
        self.annotations = {}
        self._image_index = 0
        self._selected_point_index: int = 0
        self._points: list[None | tuple[float, float]] = [None for _ in range(max_points)]

        self._fig, self._ax = plt.subplots()
        self._img_plot = None
        self._graphics = {"scatter": None, "texts": []}
        self._cids = {}

        # Used in keeping the zoom from previous image.
        self._prev_xlim = None
        self._prev_ylim = None

    def _get_fig_title(self):
        return (
            "Click on the image to select points.  "
            f"[point index: {self._selected_point_index}]"
            "\n  Press 0-9 to select the point index."
            "\n  Press Enter to advance to next image."
            "\n  Press Esc or close window to finish."
        )

    def _clear_annotations(self):
        """Clears the current annotations in the axes."""
        text_annotation: list[plt.Text] = self._graphics["texts"]
        for txt in text_annotation:
            txt.remove()
        text_annotation.clear()
        # Clear previous scatter points
        scatter_plot: plt.PathCollection | None = self._graphics["scatter"]
        if scatter_plot:
            scatter_plot.remove()
        self._graphics["scatter"] = None

    def _draw_annotations(self):
        """Creates the annotations based on current selections and draws them on the axes."""
        # Clear previous annotations and points
        self._clear_annotations()

        # Filter valid points and update plot
        valid_points_and_indices = [
            (pt[0], pt[1], i) for i, pt in enumerate(self._points) if pt is not None
        ]

        if valid_points_and_indices:
            xs, ys, _ = zip(*valid_points_and_indices)
            self._graphics["scatter"] = self._ax.scatter(xs, ys, c="lime", s=40)
            for pt_x, pt_y, i in valid_points_and_indices:
                self._graphics["texts"].append(
                    self._ax.text(pt_x + 3, pt_y - 3, str(i), color="lime", fontsize=9)
                )

    def _collect(self):
        if self._image_index >= len(self._image_ids):
            # Already collected at the end of the sequence. Do not re-collect.
            return
        image_id = self._image_ids[self._image_index]
        self.annotations[image_id] = self._points.copy()

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return
        x, y = event.xdata, event.ydata
        self._points[self._selected_point_index] = (x, y)
        self._draw_annotations()
        self._fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in map(str, range(self._max_num_points)):
            # Change the current point focus
            self._selected_point_index = int(event.key)
            self._fig.suptitle(self._get_fig_title())
        elif event.key == "enter":
            # Clean up the current image tasks
            self._collect()
            # Move to the next image
            self._image_index += 1
            if self._image_index < len(self._image_ids):
                self._load_image()
            else:
                logger.info("Annotation complete.")
                plt.close()
        elif event.key == "escape":
            logger.info("Aborted.")
            plt.close()

    def _load_image(self):
        # Store current view limits before clearing
        if self._img_plot is not None:
            self._prev_xlim = self._ax.get_xlim()
            self._prev_ylim = self._ax.get_ylim()

        self._points = [None for _ in range(self._max_num_points)]
        if self._image_paths is not None:
            img = mpimg.imread(self._image_paths[self._image_index])
        else:
            assert self._images is not None
            img = self._images[self._image_index]

        # Clear the previous image and registered graphics.
        self._clear_annotations()

        self._img_plot = self._ax.imshow(img)
        title = f"Image {self._image_index + 1}/{len(self._image_ids)}"
        if self._image_paths is not None:
            title += f": {self._image_paths[self._image_index]}"
        self._ax.set_title(title)
        self._draw_annotations()

        # Restore previous zoom/pan if available
        if self._prev_xlim and self._prev_ylim:
            self._ax.set_xlim(self._prev_xlim)
            self._ax.set_ylim(self._prev_ylim)

        self._fig.canvas.draw_idle()

    def run(self):
        self._load_image()
        self._cids["click"] = self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._cids["key"] = self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._cids["close"] = self._fig.canvas.mpl_connect(
            "close_event", lambda event: self._collect()
        )
        self._fig.suptitle(self._get_fig_title())
        plt.show()

        # Output after completion
        logger.info("Final Annotations:")
        for path, pts in self.annotations.items():
            logger.info(f"{path}: {pts}")
        return self.annotations


def main():
    """Main function for the illustration.

    This part is meant to demonstrate how to use the point selecting function under IPython notebook
    environment. This part can be put into IPython notebook after uncommenting the first line and
    changing get_control_points to imagePointsInput.get_control_points.
    """
    im1 = np.zeros((900, 900))
    im2 = np.zeros((950, 950))
    im2[-1, :] = 1
    im1[-1, :] = 1
    x1, x2 = get_control_points(im1, im2)
    print(x1)

    """
    This part demonstrates how to use drag_control_points
    """
    img = np.zeros((100, 200, 3), dtype=np.float64)
    cpts = np.array(
        [
            [10, 10],
            [20, 20],
            [10, 20],
            [50, 50],
            [20, 40],
        ],
        dtype=np.float64,
    )
    drag_control_points(img, cpts)


if __name__ == "__main__":
    main()
