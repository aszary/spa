import os

import numpy as np
import matplotlib as mp
from matplotlib import pyplot as pl

import fun


def average(spa, start=0, length=None, show=True):
    """
    plots average profile
    :param spa: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param show: show plot on screen?
    :return:
    """

    bins = len(spa.stokes_[0][0])
    size = len(spa.stokes_[0])
    if length is None:
        length = size - start

    average_ = np.zeros(bins)
    for i in xrange(start, start+length, 1):
        average_ += spa.stokes_[0][i]

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 1.9465637440940307))  # 8cm x 4.9443cm (golden ratio)
    pl.minorticks_on()
    pl.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.99)
    pl.plot(average_)
    pl.savefig(os.path.join(spa.output_dir, 'average_profile_st%d_le%d.svg' % (start, length)))
    pl.savefig(os.path.join(spa.output_dir, 'average_profile_st%d_le%d.pdf' % (start, length)))
    if show is True:
       pl.show()
    pl.close()


def single_old(spa, start=0, length=100, norm=0.1, show=True):
    """
    plots single pulses (old style)
    :param spa: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param norm: normalization factor (by hand - lazy!)
    :param show: show plot on screen?
    :return:
    """
    bins = len(spa.stokes_[0][0])
    size = len(spa.stokes_[0])
    if length is None:
        length = size - start

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)
    pl.minorticks_on()
    # plots data
    for i in xrange(start, start + length, 1):
        da = np.array(spa.stokes_[0][i]) * norm + i
        pl.plot(da, c="grey")
    pl.xlim(0, bins)
    pl.ylim(start, start+length)
    pl.savefig(os.path.join(spa.output_dir, 'single_pulses_old_st%d_le%d.svg' % (start, length)))
    pl.savefig(os.path.join(spa.output_dir, 'single_pulses_old_st%d_le%d.pdf' % (start, length)))
    if show is True:
       pl.show()
    pl.close()


def single(spa, start=0, length=100, ph_st=None, ph_end=None, cmap="inferno", show=True):
    """
    plots single pulses (new style)

    :param spa: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param show: show plot on screen?
    :return:
    """

    bins = len(spa.stokes_[0][0])
    size = len(spa.stokes_[0])
    if length is None:
        length = size - start

    single_ = spa.stokes_[0][start:start+length][:]
    if ph_st is not None:
        old_len = float(len(single_[0]))
        ns_ = np.zeros([len(single_), ph_end-ph_st])
        for i in xrange(len(single_)):
            ns_[i] = single_[i][ph_st:ph_end]
        single_ = ns_
        phase_ = np.linspace(ph_st/old_len*360., ph_end/old_len*360., len(single_[0]))
        phase_ = fun.zeroed(phase_)
    else:
        phase_ = np.linspace(0., 360., len(single_[0]))

    average_ = fun.average_profile(single_)
    counts_, pulses_ = fun.counts(single_)
    pulses_ += start

    grey = '#737373'

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.15, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)

    ax = pl.subplot2grid((5, 3), (0, 0), rowspan=4)
    pl.minorticks_on()
    pl.plot(counts_, pulses_, c=grey)
    pl.ylim(np.min(pulses_), np.max(pulses_))
    pl.xlim(1.1, -0.1)
    pl.xticks([0.1, 0.5, 0.9])
    pl.xlabel(r'counts')
    pl.ylabel('Pulse number')

    ax = pl.subplot2grid((5, 3), (0, 1), rowspan=4, colspan=2)
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto')
    pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', vmax=np.max(single_))  #, clim=(0., 1.0))
    pl.xticks([], [])
    ymin, ymax = pl.ylim()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)

    ax = pl.subplot2grid((5, 3), (4, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey)
    x0, x1 = pl.xlim(np.min(phase_), np.max(phase_))
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.xlabel(r'longitude [$^{\circ}$]')
    pl.tick_params(labeltop=False, labelbottom=True)
    pl.savefig(os.path.join(spa.output_dir, 'single_pulses_st%d_le%d.svg' % (start, length)))
    pl.savefig(os.path.join(spa.output_dir, 'single_pulses_st%d_le%d.pdf' % (start, length)))
    if show is True:
        pl.show()
    pl.close()
