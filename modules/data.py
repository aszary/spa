import os

import numpy as np


def load_westerbork(spa, file_name, end=None, bin_=None, extra_=True):
    """
    :param file_name: file to read
    :param end: number pulses to read
    :param bin_: number of bins
    :return:
    """

    with open(os.path.join(spa.data_dir, file_name)) as f:

        # read header
        res = f.readline().split()
        pulses = int(res[0])
        bins = int(res[1])
        resolution = float(res[2])  # deg/sample

        name = f.readline().strip()
        obs = f.readline().strip()

        res4 = f.readline().split()
        scan = res4[0]
        frequency = float(res4[2])
        bandwidth = float(res4[3])
        spa.period = float(res4[4])
        mjd_start = float(res4[5])

        if bin_ is not None:
            print "Warning! Bin number changed from %d to %d" % (bins, bin_)
            bins = bin_

        print file_name
        print 'obs: ', obs
        print 'Name: ', name
        print 'Number of pulses: ', pulses
        print 'Number of bins: ', bins
        print 'Resolution: ', resolution
        print 'Period: ', spa.period
        print 'Frequency: ', frequency
        print 'Bandwidth: ', bandwidth
        print "MJD start: ", mjd_start
        print "Scan: ", scan

        if end is None:
            end = pulses

        spa.stokes_ = np.zeros([4, pulses, bins])

        spa.off_rms_ = np.zeros(pulses)
        spa.base_ = np.zeros(pulses)

        ln = 4
        # read data
        for i in xrange(pulses):
            bin = 0
            while bin < bins:
                re0 = f.readline()
                re = re0.split()
                ln += 1
                re_ind = 0
                for j, r in enumerate(re):
                    spa.stokes_[0][i][bin] = float(r)
                    bin += 1
                    if bin == bins:
                        re_ind = j
                        """
                        if j == len(re)-1:
                            reread = True
                        else:
                        """
                        break
                # to prevent infinite loop for wrong bin number
                if len(re) == 0:
                    print re0
                    print "Error! Wrong bin number!"
                    exit()
            # to read extra parameters (off_rms, base)
            if extra_ is True:
                ex = 0
                extra = []
                for k in xrange(re_ind+1, len(re)):
                    extra.append(re[k])
                    ex += 1
                if ex < 3:
                    re = f.readline().split()
                    ln += 1
                    for r in re:
                        extra.append(r)
                        ex += 1
                spa.off_rms_[i] = extra[0]
                spa.base_[i] = extra[1]
            if i == end:
                break
    f.close()


def load_westerbork4(spa, file_name, end=None, bin_=None):
    """
    All 4 Stokes parameters
    :param file_name: file to read
    :param end: number pulses to read
    :param bin_: number of bins
    :return:
    """

    with open(os.path.join(spa.data_dir, file_name)) as f:

        # read header
        res = f.readline().split()
        pulses = int(res[0])
        bins = int(res[1])
        resolution = float(res[2])  # deg/sample

        name = f.readline().strip()
        obs = f.readline().strip()

        res4 = f.readline().split()
        scan = res4[0]
        frequency = float(res4[2])
        bandwidth = float(res4[3])
        period = float(res4[4])
        mjd_start = float(res4[5])

        if bin_ is not None:
            print "Warning! Bin number changed from %d to %d" % (bins, bin_)
            bins = bin_

        print file_name
        print 'obs: ', obs
        print 'Name: ', name
        print 'Number of pulses: ', pulses
        print 'Number of bins: ', bins
        print 'Resolution: ', resolution
        print 'Period: ', period
        print 'Frequency: ', frequency
        print 'Bandwidth: ', bandwidth
        print "MJD start: ", mjd_start
        print "Scan: ", scan

        if end is None:
            end = pulses

        spa.stokes_ = np.zeros([4, pulses, bins])

        spa.off_rms_ = np.zeros(pulses)
        spa.base_ = np.zeros(pulses)

        ln = 4
        # read data
        for i in xrange(pulses):
            ind = 0
            bin = 0
            while bin < bins:
                re0 = f.readline()
                re = re0.split()
                ln += 1
                for j, r in enumerate(re):
                    spa.stokes_[ind][i][bin] = r
                    ind += 1
                    if ind == 4:
                        ind = 0
                        bin += 1
                        if bin == bins:
                            re_ind = j
                            break
                if len(re) == 0:
                    print "Error! Wrong bin number!"
                    exit()
                #print i, bin
            ex = 0
            extra = []
            for k in xrange(re_ind+1, len(re)):
                extra.append(re[k])
                ex += 1
            if ex < 3:
                re = f.readline().split()
                ln += 1
                for r in re:
                    extra.append(r)
                    ex += 1
            spa.off_rms_[i] = extra[0]
            spa.base_[i] = extra[1]
            #pul = extra[2]
            if i == end:
                break
    f.close()

