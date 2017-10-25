#! /usr/bin/env python
__author__ = 'Andrzej Szary'  # peakutils by Lucas Hermann Negri
import glob
import argparse

import numpy as np

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

    def plot_all(self):
        plot.average(self)
        plot.single_old(self)
        plot.single(self)
        plot.lrfs(self)
        plot.folded(self, p3=16.7, length=300)
        plot.p3_evolution(self)

    def run1(self, files, num=0):
        """
        a simple run
        :param files: files to deal with
        :param num: which file to load
        """
        self.load(files[num], end=10000, type='westerbork')
        plot.average(self, start=8000, length=1000)
        plot.single(self, start=8000, length=1000)
        plot.lrfs(self, start=8000, ph_st=1100, ph_end=1800)
        plot.folded(self, p3=2.1508, start=8000, length=1000, ph_st=1100, ph_end=1800)
        plot.p3_evolution(self, start=0, end=10000, step=50)

    def runs(self, files, num=0, pulses=1000, end=None):
        """
        go grab a coffee (or two)
        :param files: files to deal with
        :param num: which file to load
        :param pulses: number of pulses in one set
        :param end: last pulse to load (None for all)
        """
        self.load(files[num], end=end, type='westerbork')
        if end is None:
            end = len(self.data_)
        run_num = end / pulses
        rng = np.linspace(0, end, run_num)
        for i in xrange(run_num-1):
            plot.average(self, start=int(rng[i]), length=int(rng[i+1]-rng[i]), name_mod='r%d_%d' % (num, i), show=False)
            plot.single(self, start=int(rng[i]), length=int(rng[i+1]-rng[i]), name_mod='r%d_%d' % (num, i), show=False)
            plot.lrfs(self, start=int(rng[i]), length=int(rng[i+1]-rng[i]), name_mod='r%d_%d' % (num, i), show=False)
        plot.p3_evolution(self, start=0, length=512, end=end, step=50, name_mod='r%d' % num, show=False)


def main():
    #j0815()
    #b0943()
    b1839()
    #b1828()
    print "Bye"

def b1828():
    s = SinglePulseAnalysis(data_dir='/data/leeuwen/drifting/B1828-11/')
    s.load('2007-04-12-11:17:02.ar', end=None, type='psrchive')
    #plot.average(s, name_mod="B1828", show=False)
    #plot.single_old(s, name_mod="B1828", show=False)
    #plot.single(s, name_mod="B1828", show=True)
    #plot.lrfs(s, length=256, name_mod="B1828", show=False)
    plot.p3_evolution(s, length=128, step=1, start=0, name_mod="B1828", show=True)
    #plot.p3_evolution(s, name_mod="B1828", show=False)
    #plot.folded(s, p3=8, length=660, name_mod="B1828")  # TODO fix it!



def b1839():
    s = SinglePulseAnalysis(data_dir='/data/szary/B1839-04/')
    s.load('200505521.1380.debase.gg', end=None, type='psrchive') 
    #plot.average(s, name_mod="B1839", show=False)
    #plot.single_old(s, name_mod="B1839", show=False)
    #plot.single(s, start=2200, length=200, ph_st=190, ph_end=270, name_mod="B1839", show=True)
    plot.single(s, length=3000, ph_st=190, ph_end=270, name_mod="B1839", show=True)
    #plot.lrfs(s, length=256, start=2320, ph_st=190, ph_end=270, name_mod="B1839", show=False)
    #plot.p3_evolution_b1839(s, length=128, step=1, start=0, end=3004, ph_st=190, ph_end=270, name_mod="B1839", show=True)
    #plot.p3_evolution(s, end=3004, name_mod="B1839", show=False)


def b0943():
    s2 = SinglePulseAnalysis(data_dir='/data/leeuwen/drifting/0943+10/')
    s2.run1(files)

    s3 = SinglePulseAnalysis(data_dir='/data/leeuwen/drifting/0943+10/')
    s3.runs(files)


def j0815():
    s = SinglePulseAnalysis(data_dir='/data/szary/J0815+0939/data/')
    s.load('sJ0815+0939.54015ap', end=None, type='westerbork4')
    #s.load('test.dat', end=1000, type='psrchive')  # TODO not implemented yet
    s.plot_all()

    files = ['./20111105/B0943+10_L33341_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc', './20111107/B0943+10_L33339_RSP0.PrepsubbNoclip.1_DM15.31.puma.119.gg.1pol.asc', './20111127/B0943+10_L35621_RSP0.PrepsubbZerodmNoclip_DM15.31.puma.119.gg.1pol.asc','./20111201/B0943+10_L36159_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20111204/B0943+10_L36157_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20111221/B0943+10_L39707_RSP0.ZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20120111/B0943+10_L42350_RSP0.PrepdataNoclip.1.puma.119.gg.1pol.asc']



if __name__ == "__main__":
    main()
