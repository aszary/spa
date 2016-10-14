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


def lrfs(spa, start=0, length=512, ph_st=None, ph_end=None, cmap="inferno", show=True):
    """
    the Longitude Resolved Fluctuation Spectra
    :param spa: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param show: show plot on screen?
    :return:
    """

    if length == None:
        length = len(spa.stokes_[0])

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
    lrfs_, freq_ = fun.lrfs(single_, None)

    # TODO play with that
    # average above zero
    #print np.min(average_), np.max(average_)
    #average_ += np.fabs(np.min(average_)) + 1.
    #print np.min(average_), np.max(average_)
    #exit()
    #for i in xrange(len(lrfs_)):
    #    lrfs_[i] /= average_

    counts_, pulses_ = fun.counts(np.abs(lrfs_))
    p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
    #p3, max_ind = fun.get_p3_rahuls(counts_, freq=freq_, thres=5)
    ffph_ = fun.lrfs_phase(max_ind, lrfs_)

    print "P_3 =", p3, "+-", p3_err

    grey = '#737373'

    #single_, y_min, y_max = fun.single_pulses(pc.profile_, start=50, end=100)
    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.13, bottom=0.08, right=0.99, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((5, 3), (0, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, ffph_, c=grey)
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'phase [$^{\circ}$]')
    pl.yticks([-150, 0, 150])
    ax.xaxis.set_label_position("top")
    pl.tick_params(labeltop=True, labelbottom=False)

    ax = pl.subplot2grid((5, 3), (1, 0), rowspan=3)
    pl.minorticks_on()
    pl.plot(counts_, freq_, c=grey)
    pl.ylim(freq_[0], freq_[-1])
    pl.xlim(1.1, -0.1)
    pl.xticks([0.1, 0.5, 0.9])
    pl.ylabel('frequency [$1/P$]')

    ax = pl.subplot2grid((5, 3), (1, 1), rowspan=3, colspan=2)
    pl.imshow(np.abs(lrfs_), origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto')  # , vmax=700.5)
    pl.xticks([], [])
    ymin, ymax = pl.ylim()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)

    ax = pl.subplot2grid((5, 3), (4, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey)
    x0, x1 = pl.xlim(phase_[0], phase_[-1])
    y0, y1 = pl.ylim()
    pl.ylim(y0-0.1*y1, 1.1*y1)
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.xlabel(r'longitude [$^{\circ}$]')
    pl.savefig(os.path.join(spa.output_dir, 'lrfs_st%d_le%d.svg' % (start, length)))
    pl.savefig(os.path.join(spa.output_dir, 'lrfs_st%d_le%d.pdf' % (start, length)))
    if show is True:
        pl.show()
    pl.close()


def folded(spa, p3=8., period=1., start=0, length=None, ph_st=None, ph_end=None, cmap="inferno", times=1, rngs=None, pthres=0.7, sthres=0.1, show=True):
    """
    folded profile
    :param spa: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param times: how many p3 periods to plot
    :param rngs: ranges fitting procedure
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param show: show plot on screen?
    :return:
    """


    if length is None:
        length = len(spa.stokes_[0])
    single_ = spa.stokes_[0][start:start+length][:]
    if ph_st is not None:
        old_len = float(len(single_[0]))
        ns_ = np.zeros([len(single_), ph_end-ph_st])
        for i in xrange(len(single_)):
            ns_[i] = single_[i][ph_st:ph_end]
        single_ = ns_
        ph_start = ph_st / old_len * 360.
        phase_ = np.linspace(ph_start, ph_end / old_len * 360., len(single_[0]))
        phase_ = fun.zeroed(phase_)
    else:
        phase_ = np.linspace(0., 360., len(single_[0]))

    ybins = 17
    single_ = fun.fold_single(single_, p3=p3, ybins=ybins)
    average_ = fun.average_profile(single_)

    single_ = np.array(list(single_) + (times-1) * list(single_))

    max_x_, max_y_ = fun.get_maxima(single_, 4, pthres=pthres, sthres=sthres, smooth=True)
    my_, mx_, vs, es, xs, xes = fun.fit_lines(max_y_, max_x_, rngs=rngs)
    dph = phase_[-1] - phase_[0]
    dind = len(single_[0])
    vs2 = []
    es2 = []
    xs2 = []
    xes2 = []
    for i, v in enumerate(vs):
        vs2.append(v / dind * dph * ybins / (p3 * period))  # drift rate deg/s
        es2.append(es[i] / dind * dph * ybins / (p3 * period))  # drift rate error
        if ph_st is None:
            xs2.append(xs[i] / dind * dph)  # phase
        else:
            xs2.append(xs[i] / dind * dph + phase_[0])  # phase
        xes2.append(xes[i] / dind * dph)  # phase error

    print "Drift rates:", vs2
    print "Drift rates errors:", es2

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    #pl.grid()
    #pl.scatter(range(len(vs2)), vs2, color=red, marker="+", s=200, lw=1.5)
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    pl.ylim([-1.3, 1.3])
    #pl.xlim(x0, x1)
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate [$^\circ / {\rm s}$]')
    pl.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude [$^{\circ}$]')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.contourf(single_, origin="lower", cmap=cmap, extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.grid(color="white")
    for c in xrange(4):
        pl.scatter(max_x_[c], max_y_[c], c="white", marker='x', s=10, lw=0.3)
        pl.plot(mx_[c], my_[c], c="white", lw=0.3)
    #pl.xticks([100, 200, 300, 400, 500, 600, 700, 800], [])
    pl.xticks([], [])
    pl.yticks([ybins/2., 3./2.*ybins], [r'$\frac{P_3}{2}$', r'$\frac{3P_3}{2}$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.1, 0.5, r"time", size=8., rotation=90., ha="center", va="center")
    pl.figtext(0.05, 0.5, r"$\longrightarrow$", size=20, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((4, 1), (3, 0))
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey, linewidth=1.)
    y0, y1 = pl.ylim()
    pl.ylim(y0-0.1*y1, 1.1*y1)
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude [$^{\circ}$]')
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.savefig(os.path.join(spa.output_dir, 'folded_st%d_le%d.svg' % (start, length)))
    pl.savefig(os.path.join(spa.output_dir, 'folded_st%d_le%d.pdf' % (start, length)))
    if show is True:
        pl.show()
    pl.close()

