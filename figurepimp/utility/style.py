import matplotlib.pyplot as plt

from matplotlib.pyplot import Axes, Figure


def apply_style(target: Axes = None, fontsize: int = None) -> None:
    """Applies the basic style to an axis or figure.

    This style includes removing the top and right axes and setting the font size to ticks.

    Parameters
    ----------
    target: Axes or Figure, optional
        The axis or figure to which we want to apply the style. If a figure is specified, all axes are targeted.
        If `None` is specified, we target the current axis being drawn on.
    fontsize: int, optional
        If specified, the given fontsize will be applied to all axis labels.
    """

    # If no axis is specified, we simply take the current one being drawn.
    if target is None:
        target = plt.gca()

    # If the target is a figure, we treat all its axes.
    if isinstance(target, Figure):
        target = target.axes

    if not isinstance(target, list):
        target = [target]

    for axis in target:
        # Remove the top and right axis lines to make the graph more clean.
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        if fontsize is not None:
            axis.tick_params(labelsize=fontsize)
