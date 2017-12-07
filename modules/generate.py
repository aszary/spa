import numpy as np
import matplotlib as mp
import matplotlib.pyplot as pl

import fun


def step_function(pulses=2000, bins=1024, length=128, step=10):
    da_ = np.empty([pulses, bins])

    phi = 0.
    db1 = 10
    db2 = 30

    db = db1
    for i in xrange(pulses):
        if i > 1000:
            db = db2
        for j in xrange(bins):
            da_[i][j] = np.sin(float(j+db*i) / bins * 6.*np.pi) + np.random.ranf()

    pl.imshow(da_, origin="lower", interpolation='none', aspect='auto', vmax=np.max(da_))
    pl.savefig("output/P3_step_change.pdf")
    pl.show()
    pl.close()

    freqs_ = []
    p3_ = []
    p3_err_ = []
    p3_pulse_ = []
    # single pulse data
    for k in xrange(0, pulses-length, step):
        single_ = da_[k:k+length][:]
        lrfs_, freq_ = fun.lrfs(single_, None)
        counts_, pulses_ = fun.counts(np.abs(lrfs_))
        p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_)
        if p3 is not None:
            p3_.append(p3)
            p3_err_.append(p3_err)
            p3_pulse_.append(k+length/2)

        """
        try:
            # new approach
            p3, p3_err, max_ind = fun.get_p3(counts_, x=freq_, on_fail=0)
            #p3, p3_err, max_ind = fun.get_p3_simple(counts_, x=freq_)
            if p3 is not None:
                p3_.append(p3)
                p3_err_.append(p3_err)
                p3_pulse_.append(k)
                #pl.plot(counts_)
                #pl.show()
            else:
                pass
                #pl.imshow(single_)
                #pl.plot(counts_)
                #pl.show()
        except IndexError:
            pass
        except ValueError:
            pass
        """
        #print len(counts_)
        #if len(counts_) == 255:
        freqs_.append(counts_)

    #average_ = fun.average_profile_soft(np.array(freqs_))
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
    pl.errorbar(p3_, p3_pulse_, xerr=p3_err_, color="none", lw=1., marker='_', mec="grey", ecolor="grey", capsize=0., mfc=grey, ms=1.)
    pl.ylim(0, pulses)
    pl.ylabel('start period no.')
    pl.xlabel('$P_3$')

    ax = pl.subplot2grid((4, 3), (0, 1), rowspan=3, colspan=2)
    pl.imshow(freqs_, origin="lower", interpolation='none', aspect='auto')  # , vmax=700.5)
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
    pl.savefig("output/P3_step_change_LRFS.pdf")
    pl.show()
    pl.close()


    return da_


def main():
    step_function()
    print "Bye"

if __name__ == "__main__":
    main()
