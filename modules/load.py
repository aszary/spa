import os

import numpy as np


def westerbork(cls, file_name, end=None, bin_num=None, extra_=True):
    """
    :param file_name: file to read
    :param end: number pulses to read
    :param bin_num: number of bins
    :return:
    """

    with open(os.path.join(cls.data_dir, file_name)) as f:

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
        cls.period = float(res4[4])
        mjd_start = float(res4[5])

        if bin_num is not None:
            print "Warning! Bin number changed from %d to %d" % (bins, bin_num)
            bins = bin_num

        print file_name
        print 'obs: ', obs
        print 'Name: ', name
        print 'Number of pulses: ', pulses
        print 'Number of bins: ', bins
        print 'Resolution: ', resolution
        print 'Period: ', cls.period
        print 'Frequency: ', frequency
        print 'Bandwidth: ', bandwidth
        print "MJD start: ", mjd_start
        print "Scan: ", scan

        if end is None:
            end = pulses

        cls.data_ = np.zeros([pulses, bins])

        cls.off_rms_ = np.zeros(pulses)
        cls.base_ = np.zeros(pulses)

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
                    cls.data_[i][bin] = float(r)
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
                cls.off_rms_[i] = extra[0]
                cls.base_[i] = extra[1]
            if i == end:
                break
    f.close()


def westerbork4(cls, file_name, end=None, bin_num=None):
    """
    All 4 Stokes parameters
    :param file_name: file to read
    :param end: number pulses to read
    :param bin_num: number of bins
    :return:
    """

    with open(os.path.join(cls.data_dir, file_name)) as f:

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

        if bin_num is not None:
            print "Warning! Bin number changed from %d to %d" % (bins, bin_num)
            bins = bin_num

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

        cls.stokes_ = np.zeros([4, pulses, bins])
        cls.data_ = np.zeros([pulses, bins])

        cls.off_rms_ = np.zeros(pulses)
        cls.base_ = np.zeros(pulses)

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
                    if ind == 0:
                        cls.data_[i][bin] = r
                    cls.stokes_[ind][i][bin] = r
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
            cls.off_rms_[i] = extra[0]
            cls.base_[i] = extra[1]
            #pul = extra[2]
            if i == end:
                break
    f.close()


def psrchive(cls, file_name, end=None, bin_num=None):
    """
    New format...
    :param file_name: file to read
    :param end: number pulses to read
    :param bin_num: number of bins
    :return:
    """
    try:
        import psrchive as ps
    except:
        print "Error. Python interface to psrchive not installed (remember to ./configure --enable-shared)\nExiting..."
        exit()

    arch = ps.Archive_load(os.path.join(cls.data_dir, file_name))
    src = arch.get_source()
    bnd = arch.get_bandwidth()
    frq = arch.get_centre_frequency()
    dm = arch.get_dispersion_measure()
    bins = arch.get_nbin()
    pulses = arch.get_nsubint()
    nchan = arch.get_nchan()
    npol = arch.get_npol()
    reciver = arch.get_receiver_name()
    tel = arch.get_telescope()
    arch.fscrunch()
    print "Telescope:", tel, "reciver:", reciver
    print "Source:", src
    print "DM", dm
    print "Bandwidth:", bnd
    print "Number of chanels", nchan
    print "Frequency:", frq
    print "Number of polarizations:", npol
    print "Bins:", bins
    print "Nsubint (pulses):", pulses

    cls.data_ = np.zeros([pulses, bins])

    data = arch.get_data()
    print data.shape
    for i in xrange(pulses):
        for j in xrange(bins):
            cls.data_[i][j] = data[i][0][0][j]  # 0, 0 indexes?


   # print dir(arch)
    #print dir(data)


