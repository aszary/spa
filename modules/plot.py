import os

import numpy as np
import matplotlib as mp
mp.use("Agg")
from matplotlib import pyplot as pl
from matplotlib.colors import LogNorm, PowerNorm

import fun


def average(cls, start=0, length=None, name_mod=0, show=True):
    """
    plots average profile
    :param cls: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param name_mod: output filename prefix
    :param show: show plot on screen?
    :return:
    """

    bins = len(cls.data_[0])
    size = len(cls.data_)
    if length is None:
        length = size - start

    average_ = np.zeros(bins)
    for i in xrange(start, start+length, 1):
        average_ += cls.data_[i]

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 1.9465637440940307))  # 8cm x 4.9443cm (golden ratio)
    pl.minorticks_on()
    pl.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.99)
    pl.plot(average_)
    filename = '%s_average_profile_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".svg", ".pdf")))
    print "filename:", filename
    if show is True:
       pl.show()
    pl.close()


def single_old(cls, start=0, length=100, norm=0.1, name_mod=0, show=True):
    """
    plots single pulses (old style)
    :param cls: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param norm: normalization factor (by hand - lazy!)
    :param name_mod: output filename prefix
    :param show: show plot on screen?
    :return:
    """
    bins = len(cls.data_[0])
    size = len(cls.data_)
    if length is None:
        length = size - start

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)
    pl.minorticks_on()
    # plots data
    for i in xrange(start, start + length, 1):
        da = np.array(cls.data_[i]) * norm + i
        pl.plot(da, c="grey")
    pl.xlim(0, bins)
    pl.ylim(start, start+length)
    pl.savefig(os.path.join(cls.output_dir, '%s_single_pulses_old_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_single_pulses_old_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
       pl.show()
    pl.close()


def single(cls, start=0, length=100, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, brightness=0.5,  show=True):
    """
    plots single pulses (new style)
    :param cls: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param name_mod: output filename prefix
    :param show: show plot on screen?
    :return:
    """

    bins = len(cls.data_[0])
    size = len(cls.data_)
    if length is None:
        length = size - start

    single_ = cls.data_[start:start+length][:]
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

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)

    ax = pl.subplot2grid((5, 3), (0, 0), rowspan=4)
    pl.minorticks_on()
    pl.plot(counts_, pulses_, c=grey)
    pl.ylim(np.min(pulses_), np.max(pulses_))
    pl.xlim(1.1, -0.1)
    pl.xticks([0.1, 0.5, 0.9])
    pl.xlabel(r'intensity')
    pl.ylabel('Pulse number')

    ax = pl.subplot2grid((5, 3), (0, 1), rowspan=4, colspan=2)
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto')
    im = pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto', vmax=brightness*np.max(single_))  #, clim=(0., 1.0))
    pl.xticks([], [])
    ymin, ymax = pl.ylim()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)

    ax = pl.subplot2grid((5, 3), (4, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey)
    x0, x1 = pl.xlim(np.min(phase_), np.max(phase_))
    y0 = np.min(average_)
    y1 = np.max(average_)
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))  # why?
    pl.xlabel(r'longitude [$^{\circ}$]')
    pl.tick_params(labeltop=False, labelbottom=True)
    filename = '%s_single_pulses_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()


def lrfs(cls, start=0, length=512, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, brightness=0.5, show=True):
    """
    the Longitude Resolved Fluctuation Spectra
    :param cls: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param name_mod: output filename prefix
    :param show: show plot on screen?
    :return:
    """

    if length == None:
        length = len(cls.data_)

    single_ = cls.data_[start:start+length][:]
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
    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.13, bottom=0.08, right=0.99, top=0.94, wspace=0., hspace=0.)

    ax = pl.subplot2grid((5, 3), (0, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, ffph_, c=grey)
    pl.xlim(phase_[0], phase_[-1])
    #pl.xlabel(r'phase [$^{\circ}$]')
    pl.yticks([-150, 0, 150])
    ax.xaxis.set_label_position("top")
    pl.tick_params(labeltop=True, labelbottom=False, which="both", bottom=False, top=True)

    ax = pl.subplot2grid((5, 3), (1, 0), rowspan=3)
    pl.minorticks_on()
    pl.plot(counts_/np.max(counts_), freq_, c=grey)
    pl.ylim(freq_[0], freq_[-1])
    pl.xlim(1.1, -0.1)
    pl.xticks([0.1, 0.5, 0.9])
    pl.ylabel('frequency ($1/P$)')

    ax = pl.subplot2grid((5, 3), (1, 1), rowspan=3, colspan=2)
    pl.imshow(np.abs(lrfs_), origin="lower", cmap=cmap, interpolation='none', aspect='auto', vmax=brightness*np.max(np.abs(lrfs_)))  # , vmax=700.5)
    pl.xticks([], [])
    ymin, ymax = pl.ylim()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)

    ax = pl.subplot2grid((5, 3), (4, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey)
    x0, x1 = pl.xlim(phase_[0], phase_[-1])
    y0 = np.min(average_)
    y1 = np.max(average_)
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))  # why?
    pl.xlabel(r'longitude ($^{\circ}$)')
    filename = '%s_lrfs_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()


def prefolded(cls, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=2, name_mod=0, show=True):
    """
    folded profile
    :param cls: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param name_mod: output filename prefix
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    ybins = len(single_)
    average_ = fun.average_profile(single_)
    single_ = np.array(list(single_) + (times-1) * list(single_))

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0), rowspan=3)
    pl.minorticks_on()
    pl.tick_params(labeltop=True, labelbottom=False, top="on", direction="out")
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude [$^{\circ}$]')
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    #pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.contourf(single_, origin="lower", cmap=cmap, extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.grid(color="white")

    pl.xticks([], [])
    pl.yticks([ybins/2., 3./2.*ybins], [r'$\frac{P_3}{2}$', r'$\frac{3P_3}{2}$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.1, 0.6, r"time", size=8., rotation=90., ha="center", va="center")
    pl.figtext(0.05, 0.6, r"$\longrightarrow$", size=20, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((4, 1), (3, 0))
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey, linewidth=1.)
    y0 = np.min(average_)
    y1 = np.max(average_)
    #y0, y1 = pl.ylim()
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude [$^{\circ}$]')
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))  # why it is needed?
    filename = '%s_prefolded_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()

def prefolded_fit_big(cls, p3=12.3, period=1.84, comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=2, st_inds=[[32, 33], [20, 24]], lens=[[21, 20], [16, 12]], pthres=0.7, name_mod=0, move=0, inds=[18, 24, 30], show=True):
#def prefolded_fit(cls, p3=12.3, period=1.84, comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=2, st_inds=[[38, 39], [26, 30]], lens=[[21, 20], [16, 12]], pthres=0.7, name_mod=0, skip=0, show=True):
    """
    folded profile with drift characteristics fit and components search
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param st_inds: starting indexes for fitting procedure
    :param lens: fitting paths lengths
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param move: number of bins to move (to center the driftbands)
    :param inds: indexes of cuts for components search
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    ybins = len(single_)
    average_ = fun.average_profile(single_)
    single_ = np.array(list(single_) + (times-1) * list(single_))
    single_ = np.array(list(single_[move:]) + list(single_[:move]))

    max_x_, max_y_ = fun.get_maxima2(single_, comp_num=comp_num, pthres=pthres, smooth=False)
    maxx_x_ = []
    maxx_y_ = []
    # are you insane? yes
    for i in xrange(comp_num):
        for j in xrange(len(st_inds[i])):
            maxx_x_.append([])
            maxx_y_.append([])
            ind = st_inds[i][j]
            x, y = max_x_[i][ind], max_y_[i][ind]
            maxx_x_[-1].append(x)
            maxx_y_[-1].append(y)
            for k in xrange(lens[i][j]):
                rng = range(len(max_x_[i]))
                rng.pop(ind)
                # get the closest maximum
                min_ = 1e50
                for ii in rng:
                    xn = max_x_[i][ii]
                    yn = max_y_[i][ii]
                    dist = fun.distance2D([x, y], [xn, yn])
                    if dist < min_ and yn > y:
                        min_ = dist
                        xm, ym = xn, yn
                x, y = xm, ym
                maxx_x_[-1].append(x)
                maxx_y_[-1].append(y)

    dph = phase_[-1] - phase_[0]
    dind = len(single_[0])
    mxs_ = []
    mys_ = []
    rngs = []
    for i in xrange(len(maxx_x_)):
        rngs.append([])
        size = len(maxx_x_[i])
        rngs[-1].append((0, size))

    vs2 = []
    es2 = []
    xs2 = []
    xes2 = []
    ys2 = []
    ph_ = []
    ph_err_ = []

    #"""
    for i in xrange(len(maxx_x_)):
        my_, mx_, vs, es, xs, xes = fun.fit_lineseq(maxx_y_[i], maxx_x_[i], rngs=None)
        mxs_.append(mx_)
        mys_.append(my_)
        for j, v in enumerate(vs):
            vs2.append(v / dind * dph * ybins / (p3 * period))  # drift rate deg/s
            es2.append(es[j] / dind * dph * ybins / (p3 * period))  # drift rate error
            if ph_st is None:
                xs2.append(xs[j] / dind * dph)  # phase
            else:
                xs2.append(xs[j] / dind * dph + phase_[0])  # phase
            xes2.append(xes[j] / dind * dph)  # phase error
            ys2.append(rngs[i][j][0] + (rngs[i][j][1]-rngs[i][j][0]) / 2.)
            ph_.append(xs[j] / dind)
            ph_err_.append(xes[j] / dind)

    #"""

    print "Drift rates:", vs2
    print "Drift rates errors:", es2
    print "\n\tPhases [deg]:", xs2
    print "\n\tPhases errors [deg]:", xes2
    print "\n\tPhases:", ph_
    print "\n\tPhases errors:", ph_err_

    # get sample component fits
    cx, csignal, cmsignal, cgx, cga, cvs =  fun.get_maxima4(single_, comp_num=comp_num, pthres=pthres, smooth=False, inds=inds)

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961 * 2., 4.33071))  # 8*2cm x 11cm
    pl.subplots_adjust(left=0.09, bottom=0.08, right=0.93, top=0.92, wspace=0.1, hspace=0.)

    ax = pl.subplot2grid((12, 2), (0, 0), rowspan=3)#, colspan=2)
    pl.minorticks_on()
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    pl.axhline(y=0, ls=":", lw=0.5, c=grey)
    pl.ylim([-0.8, 0.8])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate ($^\circ / {\rm s}$)')
    pl.tick_params(labeltop=True, labelbottom=False, which="both", bottom=False, top=True)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude ($^{\circ}$)')

    ax = pl.subplot2grid((12, 2), (3, 0), rowspan=6)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    for i in inds:
        pl.axhline(y=i, c=blue, lw=0.5, alpha=0.5)

    #for c in xrange(comp_num):
    #    pl.scatter(max_x_[c], max_y_[c], c="grey", marker='x', s=4, lw=0.2, alpha=0.9)

    for i in xrange(len(maxx_x_)):
        pl.scatter(maxx_x_[i], maxx_y_[i], c="white", marker='x', s=5, lw=0.2)
        #pl.scatter(max_x_[c], max_y_[c], c="red", marker='x', s=10, lw=0.3)
        for j in xrange(len(mxs_[i])):
            pl.plot(mxs_[i][j], mys_[i][j], c=red, lw=0.3)
    pl.xticks([], [])
    pl.yticks([ybins/2, 3.*ybins/2], [r'$\frac{P_3}{2}$', r'$\frac{3P_3}{2}$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.07, 0.5, r"time", size=6., rotation=90., ha="center", va="center")
    pl.figtext(0.04, 0.5, r"$\longrightarrow$", size=13, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((12, 2), (9, 0), rowspan=3)
    pl.minorticks_on()
    pl.plot(phase_, average_ / np.max(average_), c=grey, linewidth=1.)
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude ($^{\circ}$)')
    #pl.yticks([0., 0.5])

    #for i in xrange(2, -1, -1):
    for i in xrange(3):
        ma = np.max(csignal[i])
        ax = pl.subplot2grid((12, 2), ((2-i)*4, 1), rowspan=4)#, colspan=2)
        pl.minorticks_on()
        pl.locator_params(nticks=4)
        pl.tick_params(labelleft=False, labelright=True, which="both", left=False, right=True)
        ax.yaxis.set_label_position("right")
        pl.ylabel("intensity")
        #cx, csignal, cmsignal, cgx, cga, cvs
        pl.plot(phase_, csignal[i]/ma, c=red, lw=1.5)
        pl.plot(phase_, cmsignal[i]/ma, c=blue, lw=1, ls="--")
        for j in xrange(len(cga[i])):
            for k in xrange(len(cga[i][j])):
                pl.plot(phase_[cgx[i][j][0]:cgx[i][j][-1]+1], cga[i][j][k]/ma, c="black", lw=0.5, ls=":")
            for k in xrange(len(cvs[i][j])):
                fr = cvs[i][j][k] / cx[i][-1]
                ph = phase_[0] + fr * (phase_[-1] - phase_[0])
                pl.axvline(x=ph, c="black", ls='-.')
        if i == 0 or i == 2:
            pl.xlabel(r'longitude ($^{\circ}$)')
        if i == 2:
            pl.tick_params(labeltop=True, labelbottom=False, which="both", bottom=False, top=True)
            ax.xaxis.set_label_position("top")
        if i == 1:
            pl.tick_params(labeltop=False, labelbottom=False, which="both", bottom=False, top=False)


    filename = '%s_prefolded_fit_big_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()


def prefolded_fit(cls, p3=12.3, period=1.84, comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=2, st_inds=[[32, 33], [20, 24]], lens=[[21, 20], [16, 12]], pthres=0.7, name_mod=0, move=0, show=True):
#def prefolded_fit(cls, p3=12.3, period=1.84, comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=2, st_inds=[[38, 39], [26, 30]], lens=[[21, 20], [16, 12]], pthres=0.7, name_mod=0, skip=0, show=True):
    """
    folded profile with drift characteristics fit
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param st_inds: starting indexes for fitting procedure
    :param lens: fitting paths lengths
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param move: number of bins to move (to center the driftbands)
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    ybins = len(single_)
    average_ = fun.average_profile(single_)
    single_ = np.array(list(single_) + (times-1) * list(single_))
    single_ = np.array(list(single_[move:]) + list(single_[:move]))

    max_x_, max_y_ = fun.get_maxima2(single_, comp_num=comp_num, pthres=pthres, smooth=False)
    maxx_x_ = []
    maxx_y_ = []
    # are you insane? yes
    for i in xrange(comp_num):
        for j in xrange(len(st_inds[i])):
            maxx_x_.append([])
            maxx_y_.append([])
            ind = st_inds[i][j]
            x, y = max_x_[i][ind], max_y_[i][ind]
            maxx_x_[-1].append(x)
            maxx_y_[-1].append(y)
            for k in xrange(lens[i][j]):
                rng = range(len(max_x_[i]))
                rng.pop(ind)
                # get the closest maximum
                min_ = 1e50
                for ii in rng:
                    xn = max_x_[i][ii]
                    yn = max_y_[i][ii]
                    dist = fun.distance2D([x, y], [xn, yn])
                    if dist < min_ and yn > y:
                        min_ = dist
                        xm, ym = xn, yn
                x, y = xm, ym
                maxx_x_[-1].append(x)
                maxx_y_[-1].append(y)

    dph = phase_[-1] - phase_[0]
    dind = len(single_[0])
    mxs_ = []
    mys_ = []
    rngs = []
    for i in xrange(len(maxx_x_)):
        rngs.append([])
        size = len(maxx_x_[i])
        rngs[-1].append((0, size))

    vs2 = []
    es2 = []
    xs2 = []
    xes2 = []
    ys2 = []
    ph_ = []
    ph_err_ = []

    #"""
    for i in xrange(len(maxx_x_)):
        my_, mx_, vs, es, xs, xes = fun.fit_lineseq(maxx_y_[i], maxx_x_[i], rngs=None)
        mxs_.append(mx_)
        mys_.append(my_)
        for j, v in enumerate(vs):
            vs2.append(v / dind * dph * ybins / (p3 * period))  # drift rate deg/s
            es2.append(es[j] / dind * dph * ybins / (p3 * period))  # drift rate error
            if ph_st is None:
                xs2.append(xs[j] / dind * dph)  # phase
            else:
                xs2.append(xs[j] / dind * dph + phase_[0])  # phase
            xes2.append(xes[j] / dind * dph)  # phase error
            ys2.append(rngs[i][j][0] + (rngs[i][j][1]-rngs[i][j][0]) / 2.)
            ph_.append(xs[j] / dind)
            ph_err_.append(xes[j] / dind)

    #"""
    print "Drift rates:", vs2
    print "Drift rates errors:", es2
    print "\n\tPhases [deg]:", xs2
    print "\n\tPhases errors [deg]:", xes2
    print "\n\tPhases:", ph_
    print "\n\tPhases errors:", ph_err_

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.15, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    pl.axhline(y=0, ls=":", lw=0.5, c=grey)
    pl.ylim([-0.8, 0.8])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate ($^\circ / {\rm s}$)')
    pl.tick_params(labeltop=True, labelbottom=False, which="both", bottom=False, top=True)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude ($^{\circ}$)')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))

    #for c in xrange(comp_num):
    #    pl.scatter(max_x_[c], max_y_[c], c="grey", marker='x', s=4, lw=0.2, alpha=0.9)

    for i in xrange(len(maxx_x_)):
        pl.scatter(maxx_x_[i], maxx_y_[i], c="white", marker='x', s=5, lw=0.2)
        #pl.scatter(max_x_[c], max_y_[c], c="red", marker='x', s=10, lw=0.3)
        for j in xrange(len(mxs_[i])):
            pl.plot(mxs_[i][j], mys_[i][j], c=red, lw=0.3)
    pl.xticks([], [])
    pl.yticks([ybins/2, 3.*ybins/2], [r'$\frac{P_3}{2}$', r'$\frac{3P_3}{2}$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.1, 0.5, r"time", size=6., rotation=90., ha="center", va="center")
    pl.figtext(0.05, 0.5, r"$\longrightarrow$", size=13, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((4, 1), (3, 0))
    pl.minorticks_on()
    pl.plot(phase_, average_ / np.max(average_), c=grey, linewidth=1.)
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude ($^{\circ}$)')
    #pl.yticks([0., 0.5])
    filename = '%s_prefolded_fit_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()



def prefolded_fit_model(cls, p3=12.3, period=1.84, comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="viridis", darkness=1., times=2, st_inds=[[32, 33], [20, 24]], lens=[[21, 20], [16, 12]], pthres=0.7, name_mod=0, move=0, show=True):
    """
    folded profile with drift characteristics fit
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param st_inds: starting indexes for fitting procedure
    :param lens: fitting paths lengths
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param move: number of bins to move (to center the driftbands)
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    ybins = len(single_)
    average_ = fun.average_profile(single_)
    single_ = np.array(list(single_) + (times-1) * list(single_))
    single_ = np.array(list(single_[move:]) + list(single_[:move]))

    max_x_, max_y_, pos_ = fun.get_maxima_peak(single_, comp_num=comp_num, pthres=pthres)
    """
    maxx_x_ = []
    maxx_y_ = []
    # are you insane? yes
    for i in xrange(comp_num):
        for j in xrange(len(st_inds[i])):
            maxx_x_.append([])
            maxx_y_.append([])
            ind = st_inds[i][j]
            x, y = max_x_[i][ind], max_y_[i][ind]
            maxx_x_[-1].append(x)
            maxx_y_[-1].append(y)
            for k in xrange(lens[i][j]):
                rng = range(len(max_x_[i]))
                rng.pop(ind)
                # get the closest maximum
                min_ = 1e50
                for ii in rng:
                    xn = max_x_[i][ii]
                    yn = max_y_[i][ii]
                    dist = fun.distance2D([x, y], [xn, yn])
                    if dist < min_ and yn > y:
                        min_ = dist
                        xm, ym = xn, yn
                x, y = xm, ym
                maxx_x_[-1].append(x)
                maxx_y_[-1].append(y)
    """

    dph = phase_[-1] - phase_[0]
    dind = len(single_[0])
    mxs_ = []
    mys_ = []
    rngs = []
    for i in xrange(len(max_x_)):
        rngs.append([])
        size = len(max_x_[i])
        rngs[-1].append((0, size))

    vs2 = []
    es2 = []
    xs2 = []
    xes2 = []
    ys2 = []
    ph_ = []
    ph_err_ = []

    """
    for i in xrange(len(maxx_x_)):
        my_, mx_, vs, es, xs, xes = fun.fit_lineseq(maxx_y_[i], maxx_x_[i], rngs=None)
        mxs_.append(mx_)
        mys_.append(my_)
        for j, v in enumerate(vs):
            vs2.append(v / dind * dph * ybins / (p3 * period))  # drift rate deg/s
            es2.append(es[j] / dind * dph * ybins / (p3 * period))  # drift rate error
            if ph_st is None:
                xs2.append(xs[j] / dind * dph)  # phase
            else:
                xs2.append(xs[j] / dind * dph + phase_[0])  # phase
            xes2.append(xes[j] / dind * dph)  # phase error
            ys2.append(rngs[i][j][0] + (rngs[i][j][1]-rngs[i][j][0]) / 2.)
            ph_.append(xs[j] / dind)
            ph_err_.append(xes[j] / dind)

    #"""
    print "Drift rates:", vs2
    print "Drift rates errors:", es2
    print "\n\tPhases [deg]:", xs2
    print "\n\tPhases errors [deg]:", xes2
    print "\n\tPhases:", ph_
    print "\n\tPhases errors:", ph_err_

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.15, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    pl.axhline(y=0, ls=":", lw=0.5, c=grey)
    pl.ylim([-0.8, 0.8])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate ($^\circ / {\rm s}$)')
    pl.tick_params(labeltop=True, labelbottom=False, which="both", bottom=False, top=True)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude ($^{\circ}$)')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))

    #for c in xrange(comp_num):
    #    pl.scatter(max_x_[c], max_y_[c], c="grey", marker='x', s=4, lw=0.2, alpha=0.9)

    for i in xrange(len(max_x_)):
        pl.scatter(max_x_[i], max_y_[i], c="white", marker='x', s=5, lw=0.2)
        #pl.scatter(max_x_[c], max_y_[c], c="red", marker='x', s=10, lw=0.3)
        #for j in xrange(len(mxs_[i])):
        #    pl.plot(mxs_[i][j], mys_[i][j], c=red, lw=0.3)
    pl.xticks([], [])
    pl.yticks([ybins/2, 3.*ybins/2], [r'$\frac{P_3}{2}$', r'$\frac{3P_3}{2}$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.1, 0.5, r"time", size=6., rotation=90., ha="center", va="center")
    pl.figtext(0.05, 0.5, r"$\longrightarrow$", size=13, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((4, 1), (3, 0))
    pl.minorticks_on()
    pl.plot(phase_, average_ / np.max(average_), c=grey, linewidth=1.)
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude ($^{\circ}$)')
    #pl.yticks([0., 0.5])
    filename = '%s_prefolded_fit_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()



def prefolded_fitseq(cls, p3=12.5, period=1.84, comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=2, st_inds=[[38, 39], [26, 30]], lens=[[21, 20], [16, 12]], seq=13, pthres=0.7, name_mod=0, show=True):
    """
    folded profile with drift characteristics fit
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param st_inds: starting indexes for fitting procedure
    :param lens: fitting paths lengths
    :param seq: sequance length (for fitting)
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    ybins = len(single_)
    average_ = fun.average_profile(single_)
    single_ = np.array(list(single_) + (times-1) * list(single_))

    # starting indexes for clean paths (used for fitting)
    max_x_, max_y_ = fun.get_maxima2(single_, comp_num=comp_num, pthres=pthres, smooth=False)
    maxx_x_ = []
    maxx_y_ = []
    # are you insane?
    for i in xrange(comp_num):
        for j in xrange(len(st_inds[i])):
            maxx_x_.append([])
            maxx_y_.append([])
            ind = st_inds[i][j]
            x, y = max_x_[i][ind], max_y_[i][ind]
            for k in xrange(lens[i][j]):
                maxx_x_[-1].append(x)
                maxx_y_[-1].append(y)
                rng = range(len(max_x_[i]))
                rng.pop(ind)
                # get the closest maximum
                min_ = 1e50
                for ii in rng:
                    xn = max_x_[i][ii]
                    yn = max_y_[i][ii]
                    dist = fun.distance2D([x, y], [xn, yn])
                    if dist < min_ and yn > y:
                        min_ = dist
                        xm, ym = xn, yn
                x, y = xm, ym
                maxx_x_[-1].append(x)
                maxx_y_[-1].append(y)

    #print len(max_x_[0]), len(max_x_[1])
    #exit()
    dph = phase_[-1] - phase_[0]
    dind = len(single_[0])
    mxs_ = []
    mys_ = []
    rngs = []
    for i in xrange(len(maxx_x_)):
        rngs.append([])
        size = len(maxx_x_[i])
        for j in xrange(size-seq+1):
            rngs[-1].append((j, j+seq))

    vs2 = []
    es2 = []
    xs2 = []
    xes2 = []
    ys2 = []

    #"""
    for i in xrange(len(maxx_x_)):
        my_, mx_, vs, es, xs, xes = fun.fit_lineseq(maxx_y_[i], maxx_x_[i], rngs=rngs[i])
        mxs_.append(mx_)
        mys_.append(my_)
        for j, v in enumerate(vs):
            vs2.append(v / dind * dph * ybins / (p3 * period))  # drift rate deg/s
            es2.append(es[j] / dind * dph * ybins / (p3 * period))  # drift rate error
            if ph_st is None:
                xs2.append(xs[j] / dind * dph)  # phase
            else:
                xs2.append(xs[j] / dind * dph + phase_[0])  # phase
            xes2.append(xes[j] / dind * dph)  # phase error
            ys2.append(rngs[i][j][0] + (rngs[i][j][1]-rngs[i][j][0]) / 2.)
    #"""
    #print "Drift rates:", vs2
    #print "Drift rates errors:", es2

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.15, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    pl.axhline(y=0, ls=":", lw=0.5, c=grey)
    #pl.ylim([-0.8, 0.8])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate [$^\circ / {\rm s}$]')
    pl.tick_params(labeltop=True, labelbottom=False, which="both", bottom=False, top=True)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude ($^{\circ}$)')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))

    for c in xrange(comp_num):
        pl.scatter(max_x_[c], max_y_[c], c="grey", marker='x', s=4, lw=0.4, alpha=0.9)

    for i in xrange(len(maxx_x_)):
        pl.scatter(maxx_x_[i], maxx_y_[i], c="white", marker='x', s=5, lw=0.2)
        #pl.scatter(max_x_[c], max_y_[c], c="red", marker='x', s=10, lw=0.3)
        for j in xrange(len(mxs_[i])):
            pl.plot(mxs_[i][j], mys_[i][j], c=red, lw=0.3)
    pl.xticks([], [])
    pl.yticks([ybins, 2.*ybins], [r'$P_3$', r'$2 P_3$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.1, 0.5, r"time", size=8., rotation=90., ha="center", va="center")
    pl.figtext(0.05, 0.5, r"$\longrightarrow$", size=15, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((4, 1), (3, 0))
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey, linewidth=1.)
    y0, y1 = pl.ylim()
    pl.ylim(y0-0.1*y1, 1.1*y1)
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude ($^{\circ}$)')
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.ylim(y0-0.1*y1, 1.1*y1)
    filename = '%s_prefolded_fitseq_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()



def folded(cls, p3=8., period=1., comp_num=1, start=0, length=None, ph_st=None, ph_end=None, cmap="inferno", darkness=1., times=2, rngs=None, pthres=0.7, sthres=0.1, name_mod=0, ybins=12, show=True):
    """
    folded profile
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param rngs: ranges fitting procedure
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    single_ = fun.fold_single(single_, p3=p3, ybins=ybins)
    average_ = fun.average_profile(single_)

    single_ = np.array(list(single_) + (times-1) * list(single_))

    # TODO no line fitting
    """
    #my_, mx_, vs, es, xs, xes = fun.fit_lines(max_y_, max_x_, rngs=rngs)
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
    """

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0), rowspan=3)
    pl.minorticks_on()
    pl.tick_params(labeltop=True, labelbottom=False, top="on", direction="out")
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude [$^{\circ}$]')
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.contourf(single_, origin="lower", cmap=cmap, extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.grid(color="white")

    pl.xticks([], [])
    pl.yticks([ybins/2., 3./2.*ybins], [r'$\frac{P_3}{2}$', r'$\frac{3P_3}{2}$'])
    pl.axis([0, len(single_[0])-1, 1, len(single_)-1])
    pl.figtext(0.1, 0.6, r"time", size=8., rotation=90., ha="center", va="center")
    pl.figtext(0.05, 0.6, r"$\longrightarrow$", size=20, rotation=90., ha="center", va="center")

    ax = pl.subplot2grid((4, 1), (3, 0))
    pl.minorticks_on()
    pl.plot(phase_, average_, c=grey, linewidth=1.)
    y0 = np.min(average_)
    y1 = np.max(average_)
    #y0, y1 = pl.ylim()
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))
    pl.xlim(phase_[0], phase_[-1])
    pl.xlabel(r'longitude [$^{\circ}$]')
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.ylim(y0 - 0.1 * (y1-y0), y1 + 0.1 * (y1-y0))  # why it is needed?
    filename = '%s_folded_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()

def folded_fit(cls, p3=8., period=1., comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=1, rngs=None, pthres=0.7, name_mod=0, ybins=12, show=True):
    """
    folded profile with drift characteristics fit
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param rngs: ranges fitting procedure
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    single_ = fun.fold_single(single_, p3=p3, ybins=ybins)
    average_ = fun.average_profile(single_)

    single_ = np.array(list(single_) + (times-1) * list(single_))

    max_x_, max_y_ = fun.get_maxima(single_, comp_num=comp_num, pthres=pthres, smooth=False)
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

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    #pl.ylim([-1.3, 1.3])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate [$^\circ / {\rm s}$]')
    pl.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude [$^{\circ}$]')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.contourf(single_, origin="lower", cmap=cmap, extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.grid(color="white")

    for c in xrange(comp_num):
        pl.scatter(max_x_[c], max_y_[c], c="white", marker='x', s=10, lw=0.3)
        pl.plot(mx_[c], my_[c], c="white", lw=0.4)
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
    pl.ylim(y0-0.1*y1, 1.1*y1)
    filename = '%s_folded_fit_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()



def folded_fitseq(cls, p3=8., period=1., comp_num=2, start=0, length=None, ph_st=None, ph_end=None, cmap="magma", darkness=1., times=1, rngs=None, pthres=0.7, name_mod=0, ybins=12, show=True):
    """
    folded profile with drift characteristics fit
    :param cls: SinglePulseAnalysis class
    :param p3: P_3 periodicity
    :param period: pulsar period
    :param comp_num: number of components in a profile
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param darkness: lower value for brighter plots
    :param times: how many p3 periods to plot
    :param rngs: ranges fitting procedure
    :param pthres: threshold for peak finding
    :param sthres: signal threshold
    :param name_mod: output filename prefix
    :param ybins: number of bins
    :param show: show plot on screen?
    :return:
    """

    if length is None:
        length = len(cls.data_)
    single_ = cls.data_[start:start+length][:]
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

    single_ = fun.fold_single(single_, p3=p3, ybins=ybins)
    average_ = fun.average_profile(single_)

    single_ = np.array(list(single_) + (times-1) * list(single_))

    max_x_, max_y_ = fun.get_maxima(single_, comp_num=comp_num, pthres=pthres, smooth=False)
    dph = phase_[-1] - phase_[0]
    dind = len(single_[0])
    mxs_ = []
    mys_ = []
    size = len(max_x_[0])
    if rngs is None:
        seq = 5
        rngs = []
        for i in xrange(size-seq+1):
            rngs.append((i, i+seq))

    vs2 = []
    es2 = []
    xs2 = []
    xes2 = []
    ys2 = []

    for i in xrange(comp_num):
        my_, mx_, vs, es, xs, xes = fun.fit_lineseq(max_y_[i], max_x_[i], rngs=rngs)
        mxs_.append(mx_)
        mys_.append(my_)
        for j, v in enumerate(vs):
            vs2.append(v / dind * dph * ybins / (p3 * period))  # drift rate deg/s
            es2.append(es[j] / dind * dph * ybins / (p3 * period))  # drift rate error
            if ph_st is None:
                xs2.append(xs[j] / dind * dph)  # phase
            else:
                xs2.append(xs[j] / dind * dph + phase_[0])  # phase
            xes2.append(xes[j] / dind * dph)  # phase error
            ys2.append(rngs[j][0] + (rngs[j][1]-rngs[j][0]) / 2.)
    #print "Drift rates:", vs2
    #print "Drift rates errors:", es2

    """
    #my_, mx_, vs, es, xs, xes = fun.fit_lines(max_y_, max_x_, rngs=rngs)
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
    """

    red = '#f15a60'
    green = '#7ac36a'
    blue = '#5a9bd4'
    orange = '#faa75b'
    purple = '#9e67ab'
    brown = '#ce7058'
    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    #pl.ylim([-1.3, 1.3])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate [$^\circ / {\rm s}$]')
    pl.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude [$^{\circ}$]')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='none', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.contourf(single_, origin="lower", cmap=cmap, extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.grid(color="white")

    for c in xrange(comp_num):
        pl.scatter(max_x_[c], max_y_[c], c="white", marker='x', s=10, lw=0.3)
        for j in xrange(len(mxs_[c])):
            pl.plot(mxs_[c][j], mys_[c][j], c=grey, lw=0.3)
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
    pl.ylim(y0-0.1*y1, 1.1*y1)
    filename = '%s_folded_fitseq_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()




def p3_evolution(cls, length=256, start=0, end=None, step=5, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, show=True):
    """
    P3 evolution with time
    :param cls: SinglePulseAnalysis class
    :param length: number of pulses to use in lrfs
    :param start: first pulse
    :param end: last pulse
    :param step: get new p3 every X pulses
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param name_mod: output filename prefix
    :param show: show plot on screen?
    :return:
    """

    if end is None:
        end = len(cls.data_)

    freqs_ = []
    p3_ = []
    p3_err_ = []
    p3_pulse_ = []
    p3_clean_ = []
    p3_err_clean_ = []
    p3_pulse_clean_ = []

    for i in xrange(start, end-length, step):
        single_ = cls.data_[i:i+length][:]
        if ph_st is not None:
            old_len = float(len(single_[0]))
            ns_ = np.zeros([len(single_), ph_end-ph_st])
            for j in xrange(len(single_)):
                ns_[j] = single_[j][ph_st:ph_end]
            single_ = ns_
        lrfs_, freq_ = fun.lrfs(single_, None)
        counts_, pulses_ = fun.counts(np.abs(lrfs_))
        # new approach
        p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
        if p3 is not None:
            p3_.append(p3)
            p3_err_.append(p3_err)
            p3_pulse_.append(i)
            if p3 is not None:# nope p3_err < 0.5 and p3 > 4.:  # Magic number here # HACK for bi-drifter!
                p3_clean_.append(p3)
                p3_err_clean_.append(p3_err)
                p3_pulse_clean_.append(i)
        freqs_.append(counts_)

    # continous p3
    #print p3_pulse_clean_[-1]
    p3_cont_ = np.zeros([p3_pulse_clean_[-1]+1])
    on_off_ = np.zeros([p3_pulse_clean_[-1]+1])
    for i in xrange(len(p3_pulse_clean_)):
        ind = p3_pulse_clean_[i]
        p3_cont_[ind] = p3_clean_[i]
        on_off_[ind] = 1.0
    #for i in xrange(p3_pulse_clean_[-1]):
    #    on_off_[i] = np.random.ranf()

    p3_len = len(p3_cont_)
    freq = np.fft.fftfreq(p3_len, d=1.)[1:p3_len/2]  # one side frequency range
    fft = np.fft.fft(p3_cont_)[1:p3_len/2]  # fft computing
    fft = np.abs(fft)
    fft /= np.max(fft)
    fft_on = np.fft.fft(on_off_)[1:p3_len/2]  # fft computing
    fft_on = np.abs(fft_on)
    fft_on /= np.max(fft_on)
    df = fft - fft_on
    df /= np.max(df)

    average_ = fun.average_profile(np.array(freqs_))

    grey = '#737373'

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.17, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 3), (0, 0), rowspan=3)
    pl.minorticks_on()
    pl.locator_params(axis='x', nbins=4)
    #pl.plot(p3_, p3_pulse_, c=grey)
    #pl.errorbar(p3_, p3_pulse_, xerr=p3_err_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    pl.errorbar(p3_clean_, p3_pulse_clean_, xerr=p3_err_clean_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    #pl.ylim(p3_pulse_[0], p3_pulse_[-1])
    pl.ylim(p3_pulse_clean_[0], p3_pulse_clean_[-1])
    #pl.locator_params(nbins=3)
    #pl.xlim(0.9*np.min(p3_), 1.1*np.max(p3_))
    pl.xlim([10, 20])   # comment this hack!
    #pl.xticks([15, 17, 19])
    pl.ylabel('start period no.')
    pl.xlabel('$P_3$')

    ax = pl.subplot2grid((4, 3), (0, 1), rowspan=3, colspan=2)
    pl.imshow(freqs_, origin="lower", cmap=cmap, interpolation='none', aspect='auto')  # , vmax=700.5)
    pl.xticks([], [])
    #pl.grid(color="white")
    #pl.axvline(x=14., lw=1., color="white")
    ymin, ymax = pl.ylim()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)

    ax = pl.subplot2grid((4, 3), (3, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(freq_, average_, c=grey)
    x0, x1 = pl.xlim(freq_[0], freq_[-1])
    y0, y1 = pl.ylim()
    pl.ylim(y0-0.1*y1, 1.1*y1)
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.xlabel('frequency [$1/P$]')
    filename = '%s_p3_evolution_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()

    pl.figure()
    pl.minorticks_on()
    ax = pl.subplot2grid((2, 1), (0, 0))
    pl.errorbar(p3_pulse_clean_,p3_clean_, yerr=p3_err_clean_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    pl.xlim(p3_pulse_clean_[0], p3_pulse_clean_[-1])
    #pl.scatter(range(0, p3_len), p3_cont_, c="red")
    pl.ylim([10, 20])  # comment it!
    pl.ylabel('$P_3$')

    ax = pl.subplot2grid((2, 1), (1, 0))
    pl.minorticks_on()
    #pl.axvline(x=0.005, color="red")
    pl.plot(freq, fft_on, lw=2, c="red", alpha=0.5, label="on/off")
    pl.plot(freq, fft, lw=1, c="blue", alpha=0.5, label="$P_3$")
    #pl.plot(freq, df, lw=2, c="black", alpha=0.7, label="$P_3$ - on/off")
    ylims = pl.ylim()
    pl.legend()

    pl.text(0.0025, ylims[0] + 0.85*(ylims[1]-ylims[0]), "0.0025")
    pl.text(0.005, ylims[0] + 0.8*(ylims[1]-ylims[0]), "0.005")
    pl.text(0.0007, ylims[0] + 0.95*(ylims[1]-ylims[0]), "0.0007")
    #pl.text(0.008, ylims[0] + 0.4*(ylims[1]-ylims[0]), "0.008")
    pl.xlim([0., 0.05])
    pl.xlabel('frequency [$1/P$]')
    filename = '%s_p3_evolution1D_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()


def p3_evolution_modes_b1839(cls, length=256, start=0, end=None, step=5, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, modes=[], show=True):
    """
    P3 evolution with time, version for B1839
    :param cls: SinglePulseAnalysis class
    :param length: number of pulses to use in lrfs
    :param start: first pulse
    :param end: last pulse
    :param step: get new p3 every X pulses
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param name_mod: output filename prefix
    :param modes: [[0...,1], [1,...,0]] tables with defined modes
    :param show: show plot on screen?
    :return:
    """

    if end is None:
        end = len(cls.data_)

    freqs_ = [[[] for i in xrange(end-start)] for m in modes]  # OK [2][pulse_num][]
    #print freqs_[0][0]
    #exit()
    p3_ = [[] for m in modes]
    p3_err_ = [[] for m in modes]
    p3_pulse_ = [[] for m in modes]
    #p3_clean_ = [[] for m in modes]
    #p3_err_clean_ = [[] for m in modes]
    #p3_pulse_clean_ = [[] for m in modes]


    for i,m in enumerate(modes):
        # get ranges
        rngs = []
        wh = np.where(m==1)[0]
        st = wh[0]
        for j in xrange(len(wh) - 1):
            if wh[j+1] - wh[j] > 1:
                rngs.append([st, wh[j]])
                st = wh[j+1]

        for j in xrange(len(rngs)):
            dr = rngs[j][1] - rngs[j][0]
            if dr > length:
                # single pulse data
                for k in xrange(rngs[j][0], rngs[j][1]-length, step):
                    single_ = cls.data_[k:k+length][:]
                    if ph_st is not None:
                        old_len = float(len(single_[0]))
                        ns_ = np.zeros([len(single_), ph_end-ph_st])
                        for j in xrange(len(single_)):
                            ns_[j] = single_[j][ph_st:ph_end]
                        single_ = ns_
                    lrfs_, freq_ = fun.lrfs(single_, None)
                    counts_, pulses_ = fun.counts(np.abs(lrfs_))
                    try:
                        p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
                    except:
                        p3 = None
                        print j

                    #p3, p3_err, max_ind = fun.get_p3_rahuls(counts_, freq_)
                    #p3, p3_err, max_ind = fun.get_p3_simple(counts_, x=freq_)
                    #if p3 is not None:
                    if p3 is not None and p3 < 16 and p3 > 10 and p3_err < 1.:# nope p3_err < 0.5 and p3 > 4.:  # Magic number here # HACK for bi-drifter!
                        p3_[i].append(p3)
                        p3_err_[i].append(p3_err)
                        p3_pulse_[i].append(k + length / 2)
                    fr_num = len(counts_)
                    for c in counts_:
                        freqs_[i][k].append(c)

    # fill in with zeros, not very smart!
    for i in xrange(len(freqs_)):
        for j in xrange(len(freqs_[0])):
            if len(freqs_[i][j]) == 0:
                for k in xrange(fr_num):
                    freqs_[i][j].append(0.)

    # continous p3
    p3_cont_ = np.zeros([end-start])
    on_off_ = np.zeros([end-start])
    for i in xrange(len(p3_pulse_[1])):
        ind = p3_pulse_[1][i]
        p3_cont_[ind] = p3_[1][i]
        on_off_[ind] = 1.0

    #signal fft
    p3_len = len(p3_cont_)
    freq = np.fft.fftfreq(p3_len, d=1.)[1:p3_len/2]  # one side frequency range
    fft = np.fft.fft(p3_cont_)[1:p3_len/2]  # fft computing
    fft = np.abs(fft)
    fft /= np.max(fft)

    # modes fft
    modes_num = len(modes)
    mfreqs = []
    mffts = []

    for i in xrange(modes_num):
        mlen = len(modes[i])
        mfreqs.append(np.fft.fftfreq(mlen, d=1.)[1:mlen/2])
        mfft = np.fft.fft(modes[i])[1:mlen/2]  # fft computing
        mffts.append(np.abs(mfft))
        mffts[-1] /= np.max(mffts[-1])

    average_ = fun.average_profile(np.array(freqs_[1]))

    grey = '#737373'
    green = '#7ac36a'
    blue = '#5a9bd4'
    cols = [blue, green]
    labs = ["B-mode", "Q-mode"]

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.17, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 3), (0, 0), rowspan=3)
    pl.minorticks_on()
    pl.locator_params(axis='x', nbins=4)
    #pl.plot(p3_, p3_pulse_, c=grey)
    #pl.errorbar(p3_, p3_pulse_, xerr=p3_err_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    for i in xrange(len(p3_)):
        pl.errorbar(p3_[i], p3_pulse_[i], xerr=p3_err_[i], color="none", lw=1., marker='_', mec=cols[i], ecolor=cols[i], capsize=0., mfc=grey, ms=1.)
    pl.ylim(start, end-start)
    pl.ylabel('start period no.')
    pl.xlabel('$P_3$')

    ax = pl.subplot2grid((4, 3), (0, 1), rowspan=3, colspan=2)
    pl.imshow(freqs_[1], origin="lower", cmap=cmap, interpolation='none', aspect='auto')  # , vmax=700.5)
    pl.xticks([], [])
    #pl.grid(color="white")
    #pl.axvline(x=14., lw=1., color="white")
    ymin, ymax = pl.ylim()
    xmin, xmax = pl.xlim()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)

    ax = pl.subplot2grid((4, 3), (3, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(freq_, average_, c=grey)
    x0, x1 = pl.xlim(freq_[0], freq_[-1])
    y0, y1 = pl.ylim()
    pl.ylim(y0-0.1*y1, 1.1*y1)
    yt = pl.yticks()
    pl.yticks(yt[0], [])
    pl.xlabel('frequency [$1/P$]')
    filename = '%s_p3_evolution_modes_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()

    pl.figure()
    pl.minorticks_on()
    ax = pl.subplot2grid((2, 1), (0, 0))
    for i in xrange(len(p3_)):
        pl.errorbar(p3_pulse_[i],p3_[i], yerr=p3_err_[i], color="none", lw=0.1, marker='_', mec=cols[i], ecolor=cols[i], capsize=0., mfc=grey, ms=0.1)
    mod = [3.6, 3.8, 4.]
    #pl.ylim([5.4, 7.1])  # comment it!
    for i in xrange(modes_num):
        pl.scatter(range(0, len(modes[i])), modes[i]*3+mod[i], c=cols[i], s=1.01, marker=",")
    pl.xlim(start, end-start)
    pl.ylabel('$P_3$')

    ax = pl.subplot2grid((2, 1), (1, 0))
    pl.minorticks_on()
    #pl.axvline(x=0.005, color="red")
    #pl.plot(freq, fft_on, lw=2, c="red", alpha=0.5, label="on/off")
    pl.plot(freq, fft, lw=1, c="grey", alpha=0.9, label="$P_3$")
    for i in xrange(modes_num):
        pl.plot(mfreqs[i], mffts[i], lw=2, c=cols[i], alpha=0.5, label=labs[i])

    #pl.plot(freq, df, lw=2, c="black", alpha=0.7, label="$P_3$ - on/off")
    ylims = pl.ylim()
    pl.legend()

    pl.text(0.0025, ylims[0] + 0.85*(ylims[1]-ylims[0]), "0.0025")
    pl.text(0.005, ylims[0] + 0.8*(ylims[1]-ylims[0]), "0.005")
    pl.text(0.0007, ylims[0] + 0.95*(ylims[1]-ylims[0]), "0.0007")
    #pl.text(0.008, ylims[0] + 0.4*(ylims[1]-ylims[0]), "0.008")
    pl.xlim([0., 0.05])
    pl.xlabel('frequency [$1/P$]')
    filename = '%s_p3_evolution1D_modes_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()


def p3_evolution_b1839(cls, length=256, start=0, end=None, step=5, ph_st=None, ph_end=None, cmap="inferno", brightness=0.5, name_mod=0, modes=[], exp_range=None, show=True):
    """
    P3 evolution with time, version for B1839
    :param cls: SinglePulseAnalysis class
    :param length: number of pulses to use in lrfs
    :param start: first pulse
    :param end: last pulse
    :param step: get new p3 every X pulses
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param name_mod: output filename prefix
    :param modes: [[0...,1], [1,...,0]] tables with defined modes
    :param exp_range: expected P3 range
    :param show: show plot on screen?
    :return:
    """

    if end is None:
        end = len(cls.data_)

    freqs_ = []
    p3_ = []
    p3_err_ = []
    p3_pulse_ = []
    p3_clean_ = []
    p3_err_clean_ = []
    p3_pulse_clean_ = []
    q_ = []
    b_ = []

    for i in xrange(start, end-length, step):
        single_ = cls.data_[i:i+length][:]
        if ph_st is not None:
            old_len = float(len(single_[0]))
            ns_ = np.zeros([len(single_), ph_end-ph_st])
            for j in xrange(len(single_)):
                ns_[j] = single_[j][ph_st:ph_end]
            single_ = ns_
        lrfs_, freq_ = fun.lrfs(single_, None)
        counts_, pulses_ = fun.counts(np.abs(lrfs_))
        try:
            p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
        except:
            print i
        #p3, p3_err, max_ind = fun.get_p3_rahuls(counts_, freq_)
        #p3, p3_err, max_ind = fun.get_p3_simple(counts_, x=freq_, on_fail=1)
        if p3 is not None:
            p3_.append(p3)
            p3_err_.append(p3_err)
            p3_pulse_.append(i)
            if p3 < 16 and p3 > 9 and p3_err < 2.:# nope p3_err < 0.5 and p3 > 4.:  # Magic number here # HACK for bi-drifter!
            #if p3 is not None:
                p3_clean_.append(p3)
                p3_err_clean_.append(p3_err)
                p3_pulse_clean_.append(i)
        freqs_.append(counts_)

    # continous p3
    #print p3_pulse_clean_[-1]
    p3_cont_ = np.zeros([p3_pulse_clean_[-1]+1])
    on_off_ = np.empty([p3_pulse_clean_[-1]+1])
    on_off_.fill(0)
    for i in xrange(len(p3_pulse_clean_)):
        ind = p3_pulse_clean_[i]
        p3_cont_[ind] = p3_clean_[i]
        on_off_[ind] = 1
    #for i in xrange(p3_pulse_clean_[-1]):
    #    on_off_[i] = np.random.ranf()

    # P3 fft
    p3_len = len(p3_cont_)
    freq = np.fft.fftfreq(p3_len, d=1.)[1:p3_len/2]  # one side frequency range
    fft = np.fft.fft(p3_cont_)[1:p3_len/2]  # fft computing
    fft = np.abs(fft)
    fft /= np.max(fft)
    fft_on = np.fft.fft(on_off_)[1:p3_len/2]  # fft computing
    fft_on = np.abs(fft_on)
    fft_on /= np.max(fft_on)
    df = fft - fft_on
    df /= np.max(df)

    # modes fft
    modes_num = len(modes)
    mfreqs = []
    mffts = []

    for i in xrange(modes_num):
        mlen = len(modes[i])
        mfreqs.append(np.fft.fftfreq(mlen, d=1.)[1:mlen/2])
        mfft = np.fft.fft(modes[i])[1:mlen/2]  # fft computing
        mffts.append(np.abs(mfft))
        mffts[-1] /= np.max(mffts[-1])

    average_ = fun.average_profile(np.array(freqs_))

    num = len(freqs_)

    grey = '#737373'
    green = '#7ac36a'
    blue = '#5a9bd4'
    cols = [blue, green]
    labs = ["B-mode", "Q-mode"]

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.15, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 3), (0, 0), rowspan=3)
    pl.minorticks_on()
    pl.locator_params(axis='x', nbins=4)
    #pl.plot(p3_, p3_pulse_, c=grey)
    #pl.errorbar(p3_, p3_pulse_, xerr=p3_err_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    pl.errorbar(p3_clean_, p3_pulse_clean_, xerr=p3_err_clean_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    pl.ylim(start, num+start)
    #pl.ylim(p3_pulse_[0], p3_pulse_[-1])
    #pl.ylim(p3_pulse_clean_[0], p3_pulse_clean_[-1])
    if exp_range is None:
        pl.xlim(0.9*np.min(p3_), 1.1*np.max(p3_))
    else:
        pl.xlim(exp_range[0], exp_range[1])
    #pl.locator_params(nbins=3)
    #pl.xlim(0.9*np.min(p3_), 1.1*np.max(p3_))
    #pl.xlim([5.5, 7.25])   # comment this hack!
    #pl.xticks([15, 17, 19])
    pl.ylabel('start period no.')
    pl.xlabel('$P_3$ (in $P$)')

    ax = pl.subplot2grid((4, 3), (0, 1), rowspan=3, colspan=2)
    pl.imshow(freqs_, origin="lower", cmap=cmap, interpolation='none', aspect='auto', vmax=brightness*np.max(freqs_), extent=[0, 0.5, 0, num])  # , vmax=700.5) 
    pl.xticks([], [])
    #pl.grid(color="white")
    #pl.axvline(x=14., lw=1., color="white")
    #exit()
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)
    #"""
    qf = True
    bf = True
    xx = 0.11
    for i, mode in enumerate(modes):
        for j, m in enumerate(mode):
            if m == 1:
                if i == 0 and qf is True:
                    pl.plot([xx + 0.01*i , xx + 0.01*i], [j-start-0.3-length/2, j-start+0.3-length/2], lw=1., color=cols[i], alpha=1.0, label=labs[i])
                    qf = False
                elif i == 1 and bf is True:
                    pl.plot([xx + 0.01*i , xx + 0.01*i], [j-start-0.3-length/2, j-start+0.3-length/2], lw=1., color=cols[i], alpha=1.0, label=labs[i])
                    bf = False
                else:
                    pl.plot([xx + 0.01*i , xx + 0.01*i], [j-start-0.3-length/2, j-start+0.3-length/2], lw=1., color=cols[i], alpha=1.0)

            #if j > (start+length):
            #    break
    #"""
    pl.xlim(0, 0.3)
    pl.ylim(0, num)
    pl.legend(loc="upper right", fontsize=5)

    ax = pl.subplot2grid((4, 3), (3, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(freq_, average_/np.max(average_), c=grey)
    x0, x1 = pl.xlim(freq_[0], freq_[-1])
    #y0, y1 = pl.ylim()
    #pl.ylim(y0-0.1*y1, 1.1*y1)
    #yt = pl.yticks()
    pl.yticks([0., 0.5])
    pl.xlim(0, 0.3)
    pl.xticks([0.05, 0.1, 0.15, 0.2, 0.25])
    pl.xlabel('$P/P_3$')
    filename = '%s_p3_evolution_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()

    p3mean = np.mean(p3_clean_)

    pl.figure()
    pl.minorticks_on()
    ax = pl.subplot2grid((2, 1), (0, 0))
    pl.errorbar(p3_pulse_clean_,p3_clean_, yerr=p3_err_clean_, color="none", lw=0.1, marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=0.1)
    mod = [p3mean, p3mean+0.2, p3mean+0.4]
    #pl.scatter(range(0, p3_len), on_off_, c="red", s=0.01, marker=",")
    pl.scatter(range(0, p3_len), -3*on_off_+mod[2], c="red", s=1.01, marker=",")
    #pl.scatter(range(0, p3_len), p3_cont_, c="red")
    #pl.ylim([5.4, 7.1])  # comment it!
    for i in xrange(modes_num):
        pl.scatter(range(0, len(modes[i])), -3*modes[i]+mod[i], c=cols[i], s=1.01, marker=",")
    pl.xlim(p3_pulse_clean_[0], p3_pulse_clean_[-1])
    if exp_range is None:
        pl.ylim(0.9*np.min(p3_), 1.1*np.max(p3_))
    else:
        pl.ylim(exp_range[0], exp_range[1])
    #pl.ylim([5.5, 7.3])
    pl.ylabel('$P_3$')

    ax = pl.subplot2grid((2, 1), (1, 0))
    pl.minorticks_on()
    #pl.axvline(x=0.005, color="red")
    pl.plot(freq, fft_on, lw=2, c="red", alpha=0.5, label="on/off")
    pl.plot(freq, fft, lw=1, c="grey", alpha=0.9, label="$P_3$")
    for i in xrange(modes_num):
        pl.plot(mfreqs[i], mffts[i], lw=2, c=cols[i], alpha=0.5, label=labs[i])

    #pl.plot(freq, df, lw=2, c="black", alpha=0.7, label="$P_3$ - on/off")
    ylims = pl.ylim()
    pl.legend()
    #pl.text(0.0025, ylims[0] + 0.85*(ylims[1]-ylims[0]), "0.0025")
    #pl.text(0.005, ylims[0] + 0.8*(ylims[1]-ylims[0]), "0.005")
    #pl.text(0.0007, ylims[0] + 0.95*(ylims[1]-ylims[0]), "0.0007")
    #pl.text(0.008, ylims[0] + 0.4*(ylims[1]-ylims[0]), "0.008")
    pl.xlim([0., 0.05])
    pl.xlabel('frequency [$1/P$]')
    filename = '%s_p3_evolution1D_st%d_le%d.pdf' % (str(name_mod), start, length)
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    print filename
    if show is True:
        pl.show()
    pl.close()

def single_b1839(cls, start=0, length=100, ph_st=None, ph_end=None, cmap="inferno", brightness=0.5, name_mod=0, modes=[], show=True):
    """
    plots single pulses (new style)
    :param cls: SinglePulseAnalysis class
    :param start: first pulse
    :param length: number of pulses to use
    :param ph_st: phase starting index
    :param ph_end: phase ending index
    :param cmap: color map (e.g. viridis, inferno, plasma, magma)
    :param name_mod: output filename prefix
    :param modes: [[0...,1], [1,...,0]] tables with defined modes
    :param show: show plot on screen?
    :return:
    """

    bins = len(cls.data_[0])
    size = len(cls.data_)
    if length is None:
        length = size - start

    single_ = cls.data_[start:start+length][:]
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
    green = '#7ac36a'
    blue = '#5a9bd4'
    cols = [blue, green]
    labs = ["B-mode", "Q-mode"]

    mp.rc('font', size=6.)
    mp.rc('legend', fontsize=6.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.15, bottom=0.08, right=0.99, top=0.99, wspace=0., hspace=0.)

    ax = pl.subplot2grid((5, 3), (0, 0), rowspan=4)
    pl.minorticks_on()
    pl.plot(counts_, pulses_, c=grey)
    pl.ylim(np.min(pulses_), np.max(pulses_))
    pl.xlim(1.1, -0.1)
    pl.xticks([0.5, 1.0])
    pl.xlabel(r'intensity')
    pl.ylabel('Pulse number')

    ax = pl.subplot2grid((5, 3), (0, 1), rowspan=4, colspan=2)
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto')
    pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto', vmax=brightness*np.max(single_))  #, clim=(0., 1.0))
    pulse_num, ph_num = single_.shape
    qf = True
    bf = True
    for i, mode in enumerate(modes):
        for j, m in enumerate(mode):
            if m == 1:
                if i == 0 and qf is True:
                    pl.plot([ph_num/21 + 2*i , ph_num/21 +2*i], [j-start-0.3, j-start+0.3], lw=1., color=cols[i], alpha=1.0, label=labs[i])
                    qf = False
                elif i == 1 and bf is True:
                    pl.plot([ph_num/21 + 2*i , ph_num/21 +2*i], [j-start-0.3, j-start+0.3], lw=1., color=cols[i], alpha=1.0, label=labs[i])
                    bf = False
                else:
                    pl.plot([ph_num/21 + 2*i , ph_num/21 +2*i], [j-start-0.3, j-start+0.3], lw=1., color=cols[i], alpha=1.0)

            if j > (start+pulse_num):
                break
    #print pulse_num, ph_num, start
    pl.xticks([], [])
    ymin, ymax = pl.ylim([0, pulse_num])
    #pl.yticks([ymin, ymax], [y_min, y_max])
    pl.tick_params(labelleft=False)
    pl.legend(loc="upper center", fontsize=5)

    ax = pl.subplot2grid((5, 3), (4, 1), colspan=2)
    pl.minorticks_on()
    pl.plot(phase_, average_ / np.max(average_), c=grey)
    x0, x1 = pl.xlim(np.min(phase_), np.max(phase_))
    #yt = pl.yticks()
    #pl.yticks(yt[0], [])
    pl.yticks([0, 0.5])
    pl.xlabel(r'longitude ($^{\circ}$)')
    pl.tick_params(labeltop=False, labelbottom=True)
    filename = '%s_single_pulses_st%d_le%d.pdf' % (str(name_mod), start, length)
    print filename
    pl.savefig(os.path.join(cls.output_dir, filename))
    pl.savefig(os.path.join(cls.output_dir, filename.replace(".pdf", ".svg")))
    if show is True:
        pl.show()
    pl.close()




