import itertools
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import Axes, Text, Line2D
from matplotlib.lines import Path
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from typing import Optional, List, Tuple, Union

DEFAULT_RESOLUTION = 100
DEFAULT_ATTRACTION = 0.7
DEFAULT_REPULSION = 0.3
DEFAULT_SIGMA = 0.4
DEFAULT_NOISE = 0.0
DEFAULT_NOISE_SEED = 7
DEFAULT_TEXTBOX_MARGIN = 1.5


def _get_text_bb(text: Text, textbox_margin: float = 1.0) -> Bbox:
    transf = text.axes.transData.inverted()
    bb = text.get_window_extent(renderer=text.axes.figure.canvas.get_renderer())
    bb_t = bb.transformed(transf)
    return bb_t.expanded(textbox_margin, textbox_margin)


def _get_midpoint(box: Bbox) -> Tuple[float, float]:
    cx = (box.x0+box.x1)/2
    cy = (box.y0+box.y1)/2
    return cx, cy


def real_legend(axis: Axes = None, lines: Optional[List[Union[Line2D, List[Line2D]]]] = None,
                labels: Optional[List[str]] = None,
                text_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
                arrow_threshold: Optional[float] = None,
                textbox_margin: float = 1.0,
                resolution: int = DEFAULT_RESOLUTION,
                attraction: float = DEFAULT_ATTRACTION, repulsion: float = DEFAULT_REPULSION,
                sigma: float = DEFAULT_SIGMA,
                noise: float = DEFAULT_NOISE, noise_seed: int = DEFAULT_NOISE_SEED,
                debug: bool = False, **kwargs) -> List[Text]:
    """Applies the real legend to an axis object which removes the legend box and adds labels
    annotating important lines in the figure.

    It uses a method of greedy local optimization algorithms. In particular, we model the whole space as
    a square grid of pixels that correspond to placement potential. We black out all forbidden
    places (such as edges and other objects in the figure) and then define for each label a different
    optimization space. For each label, we model the target line to be "attractive" and all other objects
    to be "repulsive". Then we blur everything out to make the space smooth and simply pick the "best spot"
    given by the highest value of the placement potential.

    Parameters
    ----------
    axis: Axes, optional
        The `Axes` object to be targeted. If omitted, the current `Axes` object will be tarteted.

    lines: list of Line2D or list of list of Line2D, optional
        List of lines which we want to label. Each item is either a single `Line2D` instance or a list of `Line2D`
        instances. In the latter case, multiple lines are treated as one object and will be assigned one label.
        If omitted, all lines from the given `Axes` instance will be targeted.

    labels: list of str, optional
        List of labels to assign to the lines. If omitted, the `label` properties from the `Line2D`
        objects will be used.

    text_positions: list of pair of float, optional
        Gives the ability to force one or more labels to be placed in specific positions. Given as a list of
        two-element `float` tuples that represent coordinates in the coordinate system of the figure data.

    arrow_threshold: float, optional
        If specified, it is the minimum distance that a label object will have from the line in order for
        an arrow to be drawn from the label to the line. By default no arrows are shown. The distance is given
        in the scale of the figure data.

    textbox_margin: float
        Allows us to specify a margin around label objects to prevent them from colliding or being to close
        to other objects. The margin is given relatively to the current size of the label. For example,
        the margin `1.0` means there will be no extra margin around the text (default behavior).
        As another example, the margin 2.0 means that the total bounding box around the label text will be twice
        as large as the original text.

    resolution: int
        Controls the resolution of the label placement space given in pixels. A higher number will mean more
        precision in placement but also more time to compute the positions. Lower values will be faster
        but with more rigid placement.

    attraction: float
        Controls the relative strength of how much the target line will attract the label.
        Tweak this parameter to fine tune placement.

    repulsion: float
        Controls the relative streength of how much all other non-target lines will repel the label.
        Tweak this parameter to fine tune placement.

    sigma: float
        Controlls how much we will smooth out the label positioning horizon.
        Tweak this parameter to fine tune placement.

    noise: float
        Controls the noise power to inject into the label positioning process. More noise will increase the probability
        that the object will be placed further from some optimal position based on the model of this method.
        On the other hand it can allow placing nodes in more convenient places.

        Tweak this parameter to fine tune placement. Default is 0.0 but for best results use values
        between 0.0 and 0.5.

    noise_seed: int
        If noise is added to the placement process, this is the random seed. Change the seed to change the placement
        outcome.

    debug: bool
        If set to `True` then a debug figure will be shown with optimization heatmaps for every line object.
        Dark areas will show areas we were trying to avoid. Light areas will show areas where the label
        was likely to be placed. The red dot is the final placement.

    """

    # If no axis is specified, we simply take the current one being drawn.
    if axis is None:
        axis = plt.gca()

    # If no lines are specified as targets, we simply target all lines.
    if lines is None:
        lines = axis.lines

    # Make sure if labels and/or text positions are specified, that they are the same length as lines.
    if labels is not None:
        assert(len(labels) == len(lines))
    if text_positions is not None:
        assert(len(text_positions) == len(lines))

    num_lines = len(lines)
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # Draw text to get the bounding boxes. We place it in the center of the plot to avoid any impact on the viewport.
    labels_specified = True
    if labels is not None:
        labels_specified = False
        labels = []
    colors = []
    texts = []
    texts_bb = []
    xc, yc = xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2
    for l in range(num_lines):
        line = lines[l] if not isinstance(lines[l], list) else lines[l][0]

        if not labels_specified:
            labels.append(line.get_label())
        label = labels[l]

        color = line.get_color()
        colors.append(color)

        # The text position is either going to be in the center or, in some given position
        # if it is specified in the input arguments.
        x, y = xc, yc
        if text_positions is not None and text_positions[l] is not None:
            assert(isinstance(text_positions[l], tuple))
            assert(len(text_positions[l]) == 2)
            x, y = text_positions[l]

        text = axis.text(xc, yc, label,
                         color=color,
                         horizontalalignment='center',
                         verticalalignment='center')
        texts.append(text)

        text_bb = _get_text_bb(text, textbox_margin)
        texts_bb.append(text_bb.translated(-text_bb.width/2-text_bb.x0, -text_bb.height/2-text_bb.y0))

    # Build the "points of presence" matrix with all that belong to certain lines.
    pop = np.zeros((num_lines, resolution, resolution), dtype=np.float)

    for l in range(num_lines):
        line = [lines[l]] if not isinstance(lines[l], tuple) else lines[l]

        for x_i, y_i in itertools.product(range(resolution), range(resolution)):
            x_f, y_f = (np.array([x_i, y_i]) / resolution) * ([xmax-xmin, ymax-ymin]) + [xmin, ymin]
            text_bb_xy = texts_bb[l].translated(x_f, y_f)

            if text_bb_xy.x0 < xmin or text_bb_xy.x1 > xmax or text_bb_xy.y0 < ymin or text_bb_xy.y1 > ymax:
                pop[l, x_i, y_i] = 1.0

            elif any(line_part.get_path().intersects_bbox(text_bb_xy, filled=False) for line_part in line):
                pop[l, x_i, y_i] = 1.0

            # If a text position is already specified, we will immediately add it to the pop.
            if text_positions is not None and text_positions[l] is not None:
                if texts_bb[l].overlaps(text_bb_xy):
                    pop[l, x_i, y_i] = 1.0

    if debug:
        debug_f, debug_ax = plt.subplots(nrows=1, ncols=num_lines)

    for l in range(num_lines):

        # If the position of this label has been provided in the input arguments, we can just skip it.
        if text_positions is not None and text_positions[l] is not None:
            continue

        # Find empty space, which is a nice place for labels.
        empty_space = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0

        # blur the pop's
        pop_blurred = pop.copy()
        for ll in range(num_lines):
            pop_blurred[ll] = ndimage.gaussian_filter(pop[ll], sigma=sigma * resolution/5)

        # Positive weights for current line, negative weight for others....
        w = - repulsion * np.ones(num_lines, dtype=np.float)
        w[l] = attraction

        # calculate a field
        p = empty_space + np.sum(w[:, np.newaxis, np.newaxis] * pop_blurred, axis=0)

        # Add noise to the field if specified.
        if noise > 0.0:
            np.random.seed(noise_seed)
            p += np.random.normal(0.0, noise, p.shape)

        pos = np.argmax(p)  # note, argmax flattens the array first
        best_x, best_y = (pos / resolution, pos % resolution)
        x = xmin + (xmax-xmin) * best_x / resolution
        y = ymin + (ymax-ymin) * best_y / resolution

        if debug:
            im1 = debug_ax[l].imshow(p.T, interpolation='nearest', origin="lower")
            debug_ax[l].set_title("Heatmap for: " + texts[l].get_text())
            debug_ax[l].plot(best_x, best_y, 'ro')
            divider = make_axes_locatable(debug_ax[l])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            debug_f.colorbar(im1, cax=cax, orientation='vertical')

        texts[l].set_position((x, y))

        # Prevent collision by blocking out the bounding box of this text box.
        text_bb_new = _get_text_bb(texts[l], textbox_margin)
        x_i_min, y_i_min = tuple(((text_bb_new.min - [xmin, ymin]) / ([xmax-xmin, ymax-ymin]) * resolution).astype(int))
        x_i_max, y_i_max = tuple(((text_bb_new.max - [xmin, ymin]) / ([xmax-xmin, ymax-ymin]) * resolution).astype(int))

        # Augmend the barrier to prevent collision between labels.
        w_barrier = int(round((x_i_max-x_i_min) / 2))
        h_barrier = int(round((y_i_max-y_i_min) / 2))
        x_i_min = int(max(0, x_i_min-w_barrier))
        y_i_min = int(max(0, y_i_min-h_barrier))
        x_i_max = int(min(resolution-1, x_i_max+w_barrier))
        y_i_max = int(min(resolution-1, y_i_max+h_barrier))

        pop[l, x_i_min:x_i_max+1, y_i_min:y_i_max+1] = 1.0

    # If the arrow threshold has been specified, draw arrows where needed.
    if arrow_threshold is not None:
        for l in range(num_lines):

            # Get all points on the path (including some interpolated ones).
            line = [lines[l]] if not isinstance(lines[l], tuple) else lines[l]
            points = np.vstack([l.get_path().interpolated(10).vertices for l in line])

            # Get the midpoint of the text box.
            text_c = np.array(_get_midpoint(_get_text_bb(texts[l], textbox_margin)))

            # Get all distances.
            distances = [np.linalg.norm(text_c - p) for p in points]
            d_min_idx = np.argmin(distances)

            # If the distance is larger than the threshold, draw the line.
            if distances[d_min_idx] > arrow_threshold:

                # Find first point that doesn't intersect with any other text box.
                d_sorted_idx = np.argsort(distances)
                xytext = texts[l].get_position()
                xy = points[d_min_idx, :]
                for idx in d_sorted_idx:
                    tmp_line = Path([xytext, points[idx, :]])
                    intersects_with_any_textbox = all(
                        not tmp_line.intersects_bbox(_get_text_bb(texts[i], textbox_margin))
                        for i in range(len(texts))
                        if i != l)
                    if intersects_with_any_textbox:
                        xy = points[idx, :]
                        break

                # Draw the new text with the arrow.
                a = axis.annotate(labels[l], xy=xy, xytext=xytext, ha="center", va="center", color=colors[l],
                                  arrowprops=dict(arrowstyle="->", color=colors[l]))

                # Hide original text.
                texts[l].set_visible(False)
                texts[l] = a

    # Remove the ugly legend.
    ugly_legend = axis.get_legend()
    if ugly_legend is not None:
        ugly_legend.remove()

    if debug:
        debug_f.show()

    # We return all the placed labels.
    return texts
