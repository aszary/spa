#! /usr/bin/env python
__author__ = 'Andrzej Szary'  # peakutils by Lucas Hermann Negri
import glob
import argparse

import modules.load as load
import modules.plot as plot


class SinglePulseAnalysis:

    def __init__(self, data_dir, output_dir='output'):
        self.data_dir = data_dir
        self.output_dir = output_dir

        self.stokes_ = None  # not used in plots anymore, use data_ instead
        self.data_ = None
        self.off_rms_ = None
        self.base = None

        self.types = {'westerbork': load.westerbork, 'westerbork4': load.westerbork4, 'psrchive': load.psrchive}

    def load(self, file_name, end=None, bin_num=None, type='westerbork4'):
        try:
            fun = self.types[type]
        except:
            print 'Loading function not implemented.\nExiting...'
            exit()
        fun(self, file_name, end=end, bin_num=bin_num)

    def plot(self):
        plot.average(self)
        plot.single_old(self)
        plot.single(self)
        plot.lrfs(self)
        plot.folded(self, p3=16.7, length=300)
        plot.p3_evolution(self)


def main():
    #s = SinglePulseAnalysis(data_dir='/data/szary/J0815+0939/data/')
    #s.load('sJ0815+0939.54015ap', end=None, type='westerbork4')
    #s2.load('test.dat', end=1000, type='psrchive')  # TODO not implemented yet
    #s.plot()

    files = ['./20111105/B0943+10_L33341_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc', './20111107/B0943+10_L33339_RSP0.PrepsubbNoclip.1_DM15.31.puma.119.gg.1pol.asc', './20111127/B0943+10_L35621_RSP0.PrepsubbZerodmNoclip_DM15.31.puma.119.gg.1pol.asc','./20111201/B0943+10_L36159_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20111204/B0943+10_L36157_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20111221/B0943+10_L39707_RSP0.ZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20120111/B0943+10_L42350_RSP0.PrepdataNoclip.1.puma.119.gg.1pol.asc']


    s2 = SinglePulseAnalysis(data_dir='/data/leeuwen/drifting/0943+10/')
    s2.load(files[0], end=1000, type='westerbork')
    s2.plot()


    print "Bye"


if __name__ == "__main__":
    main()
