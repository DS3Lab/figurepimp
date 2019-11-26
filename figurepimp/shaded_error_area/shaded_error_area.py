
import numpy as np

from matplotlib.ticker import EngFormatter


def shaded_error_area(axis, timeline, results_mean, results_std, result_labels=None, result_label_locations=None,
                      result_colors=None, xaxis_label=None, yaxis_label=None, title=None,
                      human_readable_labels=False):
    """
    Plot a shaded error area plot plot on a given axis instance.

    This plot contains one or more lines that represent mean values with each having an optional
    background shading of the same color that represents the standard deviation.

    """

    # If colors are not specified use matplotlib defailt palette.
    if result_colors is None:
        result_colors = ["C%d" % i for i in range(len(results_mean))]

    # Draw the shaded areas.
    for i in range(len(results_std)):
        axis.fill_between(timeline, results_mean[i]-results_std[i], results_mean[i]+results_std[i],
                          color=result_colors[i], alpha=0.05)

    # Draw the mean lines.
    result_lines = []
    for i in range(len(results_std)):
        label = None if result_labels is None else result_labels[i]
        line = axis.plot(timeline, results_mean[i], lw=2, color=result_colors[i], label=label)
        result_lines.append(line)

    # Draw the labels if they were specified.
    if result_labels is not None and result_label_locations is not None:
        for i in range(len(result_labels)):
            axis.text(result_label_locations[i][0], result_label_locations[i][1],
                      result_labels[i], color=result_colors[i])

    # Put labels on the x and y axes if specified.
    if xaxis_label is not None:
        axis.set_xlabel(xaxis_label)
    if yaxis_label is not None:
        axis.set_ylabel(yaxis_label)

    # Configure human readable formatting if specified.
    if human_readable_labels:
        formatter0 = EngFormatter(places=0)
        axis.yaxis.set_major_formatter(formatter0)

    # Remove the top and right axis lines to make the graph more clean.
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # Set title if specified.
    if title is not None:
        axis.set_title(title)

    # Return the result mean lines.
    return result_lines


def difference_span(axis, timeline, line1, line2, y_value, bar_width=0.02, text_downshift=0.04):
    """
    Draws a horizontal line at height y_value between two points on line1 and line2.
    """

    # Make sure we are dealing with numpy arrays.
    line1, line2 = np.array(line1), np.array(line2)

    # Get the indices of the leftmost intersection points between the horizontal cutoff value and each of the lines.
    idx1, idx2 = np.argmax(line1 < y_value), np.argmax(line2 < y_value)

    # Get the corresponding x-coordinate values given the timeline.
    x1, x2 = timeline[idx1], timeline[idx2]

    # Draw a horizontal arrow from line1 to line2 at height defined by y_value.
    axis.plot([x1, x1], [y_value-bar_width, y_value+bar_width], color='black')
    axis.plot([x2, x2], [y_value-bar_width, y_value+bar_width], color='black')
    axis.annotate(s='', xy=(x1, y_value), xytext=(x2, y_value), arrowprops=dict(arrowstyle='<->'))

    # Draw the caption with relative horizontal difference. Its position is vertically shifted by text_downshift.
    axis.text((x1+x2)/2, y_value-text_downshift, '%.1fx' % (x2/x1), horizontalalignment='center')
