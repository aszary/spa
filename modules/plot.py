import os

import numpy as np
import matplotlib as mp
mp.use("TkAgg")
from matplotlib import pyplot as pl

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

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 1.9465637440940307))  # 8cm x 4.9443cm (golden ratio)
    pl.minorticks_on()
    pl.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.99)
    pl.plot(average_)
    filename = '%s_average_profile_st%d_le%d.svg' % (str(name_mod), start, length)
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

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
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


def single(cls, start=0, length=100, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, show=True):
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
    pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto', vmax=np.max(single_))  #, clim=(0., 1.0))
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
    pl.savefig(os.path.join(cls.output_dir, '%s_single_pulses_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_single_pulses_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()


def lrfs(cls, start=0, length=512, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, show=True):
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
    pl.imshow(np.abs(lrfs_), origin="lower", cmap=cmap, interpolation='none', aspect='auto')  # , vmax=700.5)
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
    pl.savefig(os.path.join(cls.output_dir, '%s_lrfs_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_lrfs_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()


def folded(cls, p3=8., period=1., comp_num=1, start=0, length=None, ph_st=None, ph_end=None, cmap="inferno", darkness=1., times=1, rngs=None, pthres=0.7, sthres=0.1, name_mod=0, show=True):
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

    ybins = 30
    single_ = fun.fold_single(single_, p3=p3, ybins=ybins)
    average_ = fun.average_profile(single_)

    single_ = np.array(list(single_) + (times-1) * list(single_))

    max_x_, max_y_ = fun.get_maxima(single_, comp_num, pthres=pthres, sthres=sthres, smooth=True)
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

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
    mp.rc('axes', linewidth=0.5)
    mp.rc('lines', linewidth=0.5)

    pl.figure(figsize=(3.14961, 4.33071))  # 8cm x 11cm
    pl.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.92, wspace=0., hspace=0.)

    ax = pl.subplot2grid((4, 1), (0, 0))#, colspan=2)
    pl.minorticks_on()
    # TODO no line fitting
    #pl.errorbar(xs2, vs2, yerr=es2, xerr=xes2, color="none", lw=1., marker='_', mec=red, ecolor=red, capsize=0., mfc=red, ms=6)
    pl.ylim([-1.3, 1.3])
    pl.xlim(phase_[0], phase_[-1])
    pl.ylabel(r'Drift rate [$^\circ / {\rm s}$]')
    pl.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")
    pl.xlabel(r'longitude [$^{\circ}$]')

    ax = pl.subplot2grid((4, 1), (1, 0), rowspan=2)
    pl.minorticks_on()
    pl.imshow(single_, origin="lower", cmap=cmap, aspect='auto', interpolation='bicubic', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5], vmax=darkness*np.max(single_))
    #pl.imshow(single_, origin="lower", cmap=cmap, interpolation='bicubic', aspect='auto', extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.contourf(single_, origin="lower", cmap=cmap, extent=[-0.5, len(single_[0])-0.5, -0.5, len(single_)-0.5])
    #pl.grid(color="white")

    for c in xrange(comp_num):
        pl.scatter(max_x_[c], max_y_[c], c="white", marker='x', s=10, lw=0.3)
        # TODO no line fitting
        #pl.plot(mx_[c], my_[c], c="white", lw=0.3)
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
    pl.savefig(os.path.join(cls.output_dir, '%s_folded_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_folded_st%d_le%d.pdf' % (str(name_mod), start, length)))
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
            p3_pulse_.append(i + length / 2)
            if p3 is not None:# nope p3_err < 0.5 and p3 > 4.:  # Magic number here # HACK for bi-drifter!
                p3_clean_.append(p3)
                p3_err_clean_.append(p3_err)
                p3_pulse_clean_.append(i + length / 2)
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

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
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
    #pl.xlim([5.4, 7.1])   # comment this hack!
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
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()

    pl.figure()
    pl.minorticks_on()
    ax = pl.subplot2grid((2, 1), (0, 0))
    pl.errorbar(p3_pulse_clean_,p3_clean_, yerr=p3_err_clean_, color="none", lw=1., marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=1.)
    pl.xlim(p3_pulse_clean_[0], p3_pulse_clean_[-1])
    #pl.scatter(range(0, p3_len), p3_cont_, c="red")
    #pl.ylim([5.4, 7.1])  # comment it!
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

    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution1D_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution1D_st%d_le%d.pdf' % (str(name_mod), start, length)))
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
                    p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
                    #p3, p3_err, max_ind = fun.get_p3_rahuls(counts_, freq_)
                    #p3, p3_err, max_ind = fun.get_p3_simple(counts_, x=freq_)
                    if p3 is not None:
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
    cols = ["green", "blue"]
    labs = ["B-mode", "Q-mode"]

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
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
    pl.ylim(0, end-start)
    pl.ylabel('start period no.')
    pl.xlabel('$P_3$')

    ax = pl.subplot2grid((4, 3), (0, 1), rowspan=3, colspan=2)
    pl.imshow(freqs_[1], origin="lower", cmap=cmap, interpolation='none', aspect='auto')  # , vmax=700.5)
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
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution_modes_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution_modes_st%d_le%d.pdf' % (str(name_mod), start, length)))
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
    pl.ylim([5.5, 7.3])
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

    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution1D_modes_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution1D_modes_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()


def p3_evolution_b1839(cls, length=256, start=0, end=None, step=5, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, modes=[], show=True):
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
        p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
        #p3, p3_err, max_ind = fun.get_p3_rahuls(counts_, freq_)
        #p3, p3_err, max_ind = fun.get_p3_simple(counts_, x=freq_, on_fail=1)
        if p3 is not None:
            p3_.append(p3)
            p3_err_.append(p3_err)
            p3_pulse_.append(i + length/2)
            if p3_err < 0.5 and p3 > 4.:  # Magic number here # HACK for bi-drifter!
                p3_clean_.append(p3)
                p3_err_clean_.append(p3_err)
                p3_pulse_clean_.append(i + length/2)
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

    grey = '#737373'
    cols = ["green", "blue"]
    labs = ["B-mode", "Q-mode"]

    mp.rc('font', size=7.)
    mp.rc('legend', fontsize=7.)
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
    #pl.xlim([5.5, 7.25])   # comment this hack!
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
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()

    pl.figure()
    pl.minorticks_on()
    ax = pl.subplot2grid((2, 1), (0, 0))
    pl.errorbar(p3_pulse_clean_,p3_clean_, yerr=p3_err_clean_, color="none", lw=0.1, marker='_', mec=grey, ecolor=grey, capsize=0., mfc=grey, ms=0.1)
    mod = [3.6, 3.8, 4.]
    #pl.scatter(range(0, p3_len), on_off_, c="red", s=0.01, marker=",")
    pl.scatter(range(0, p3_len), on_off_*3+mod[2], c="red", s=1.01, marker=",")
    #pl.scatter(range(0, p3_len), p3_cont_, c="red")
    #pl.ylim([5.4, 7.1])  # comment it!
    for i in xrange(modes_num):
        pl.scatter(range(0, len(modes[i])), modes[i]*3+mod[i], c=cols[i], s=1.01, marker=",")
    pl.xlim(p3_pulse_clean_[0], p3_pulse_clean_[-1])
    pl.ylim([5.5, 7.3])
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

    pl.text(0.0025, ylims[0] + 0.85*(ylims[1]-ylims[0]), "0.0025")
    pl.text(0.005, ylims[0] + 0.8*(ylims[1]-ylims[0]), "0.005")
    pl.text(0.0007, ylims[0] + 0.95*(ylims[1]-ylims[0]), "0.0007")
    #pl.text(0.008, ylims[0] + 0.4*(ylims[1]-ylims[0]), "0.008")
    pl.xlim([0., 0.05])
    pl.xlabel('frequency [$1/P$]')

    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution1D_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_p3_evolution1D_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()

def single_b1839(cls, start=0, length=100, ph_st=None, ph_end=None, cmap="inferno", name_mod=0, modes=[], show=True):
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
    cols = ["green", "blue"]
    labs = ["B-mode", "Q-mode"]

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
    pl.imshow(single_, origin="lower", cmap=cmap, interpolation='none', aspect='auto', vmax=np.max(single_))  #, clim=(0., 1.0))
    pulse_num, ph_num = single_.shape
    for i, mode in enumerate(modes):
        for j, m in enumerate(mode):
            if m == 1:
                pl.plot([ph_num/2 + 2*i , ph_num/2 +2*i], [j-start-0.3, j-start+0.3], lw=0.5, color=cols[i])
            if j > (start+pulse_num):
                break

    #print pulse_num, ph_num, start
    pl.xticks([], [])
    ymin, ymax = pl.ylim([0, pulse_num])
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
    pl.savefig(os.path.join(cls.output_dir, '%s_single_pulses_st%d_le%d.svg' % (str(name_mod), start, length)))
    pl.savefig(os.path.join(cls.output_dir, '%s_single_pulses_st%d_le%d.pdf' % (str(name_mod), start, length)))
    if show is True:
        pl.show()
    pl.close()




