# An example usage script that shows a figure window and saves it in a PNG file.

import pandas as pd
import matplotlib.pyplot as plt
from shaded_error_area import shaded_error_area, difference_span

# Enables us to include LaTeX symbols in all plot captions.
plt.rc('text', usetex=True)

if __name__ == "__main__":

    data = pd.read_csv('sample_data.csv', index_col=0)

    labellocs = [
        [(0.28, 80000), (0.07, 130000), (0.25, -3000)],
        [(0.28, 0.1), (0.02, 0.18), (0.31, -0.04)]
    ]
    labels = ['Naive Approach 1', 'Naive Approach 2', 'Our Approach']

    f, ax = plt.subplots(1, 2, figsize=(10, 4))

    shaded_error_area(ax[0],
                      timeline=data.index,
                      results_mean=[data['mean_cumm_2'], data['mean_cumm_1'], data['mean_cumm_0']],
                      results_std=[data['std_cumm_2'], data['std_cumm_1'], data['std_cumm_0']],
                      result_labels=labels,
                      xaxis_label='Timeline',
                      yaxis_label='Function Value',
                      result_label_locations=labellocs[0],
                      title='(a) \\textsc{Example of 3 growing trends}',
                      human_readable_labels=True)

    shaded_error_area(ax[1],
                      timeline=data.index,
                      results_mean=[data['mean_inst_2'], data['mean_inst_1'], data['mean_inst_0']],
                      results_std=[data['std_inst_2'], data['std_inst_1'], data['std_inst_0']],
                      result_labels=labels,
                      xaxis_label='Timeline',
                      yaxis_label='Function Value',
                      result_label_locations=labellocs[1],
                      title='(b) \\textsc{Example of a gap span}')

    difference_span(ax[1], data.index, data['mean_inst_0'], data['mean_inst_2'], 0.07)

    f.savefig('sample.png')
    plt.show()
