import os

import numpy as np
import matplotlib as mp
from matplotlib import pyplot as pl

def average(spa, start=0, end=None):
    """
    plots average profile
    :param spa: SinglePulseAnalysis class
    :param start: first pulse
    :param end: last pulse
    :return:
    """

    bins = len(spa.stokes_[0][0])
    size = len(spa.stokes_[0])
    if end is None:
        end = size

    average_ = np.zeros(bins)
    for i in xrange(start, end, 1):
        average_ += spa.stokes_[0][i]

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 1.9465637440940307))  # 8cm x 4.9443cm (golden ratio)
    pl.minorticks_on()
    pl.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.99)
    pl.plot(average_)
    pl.savefig(os.path.join(spa.output_dir, 'average_profile_%d_%d.svg' % (start, end)))
    pl.savefig(os.path.join(spa.output_dir, 'average_profile_%d_%d.pdf' % (start, end)))
    pl.show()
