#! /usr/bin/env python
__author__ = 'Andrzej Szary'  # peakutils by Lucas Hermann Negri
import glob
import argparse

import numpy as np

import modules.load as load
import modules.plot as plot
import modules.generate as generate


class SinglePulseAnalysis:

    def __init__(self, data_dir, output_dir='output'):
        self.data_dir = data_dir
        self.output_dir = output_dir

        self.stokes_ = None  # not used in plots anymore, use data_ instead
        self.data_ = None
        self.off_rms_ = None
        self.base = None

        self.types = {'meerkat': load.meerkat, 'westerbork': load.westerbork, 'westerbork4': load.westerbork4, 'psrchive': load.psrchive, "simple": load.simple}

    def load(self, file_name, end=None, bin_num=None, type='westerbork4'):
        try:
            fun = self.types[type]
        except:
            print 'Loading function not implemented.\nExiting...'
            exit()
        fun(self, file_name, end=end, bin_num=bin_num)


    def plot_all(self):
        plot.average(self)
        #plot.single_old(self)
        plot.single(self, start=0, length=len(self.data_))
        plot.lrfs(self)
        plot.folded(self, p3=16.7)
        plot.p3_evolution(self, step=10, length=128)  #, start=6200)

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


def test_p3():
    s = SinglePulseAnalysis(data_dir='/data/szary/B1839-04/')
    #s.load('200505521.1380.debase.gg', end=None, type='psrchive') 
    s.data_ = generate.step_function()
    #plot.p3_evolution_b1839(s, length=128, step=1, start=0, end=1000, name_mod="test", show=True)

def b1839():
    s = SinglePulseAnalysis(data_dir='/data/szary/B1839-04/')
    bright = load.modes("data/bright.txt", includes=True, mod=1.0)
    quiet = load.modes("data/ignored.txt", includes=False, mod=1.0)
    #s.load('200505521.1380.debase.gg', end=None, type='psrchive')

    # 2003
    #s.load('200309159.1380.singlepulse.gg', end=None, type='psrchive')
    #plot.average(s, name_mod="1_B1839", show=False)
    #plot.single(s, start=0, length=1025, ph_st=0, ph_end=55, name_mod="1_B1839", show=True)
    #plot.lrfs(s, length=512, start=250, ph_st=0, ph_end=55, name_mod="1_B1839", show=False)
    #plot.folded(s, p3=12.2910488557, start=370, length=100, ph_st=0, ph_end=55, name_mod="1_B1839", ybins=24)
    #plot.folded(s, p3=12.2910488557, start=0, length=700, ph_st=0, ph_end=55, name_mod="_B1839", ybins=24)
    #plot.p3_evolution_b1839(s, length=256, step=1, start=0, ph_st=0, ph_end=55, name_mod="1_B1839", exp_range=(10,15), show=True)

    # 2005-05(521)
    # HERE
    s.load('200505521.1380.singlepulse.gg', end=None, type='psrchive')  # best # in the second paper
    #plot.average(s, name_mod="2_B1839", show=False)
    plot.single(s, start=400, length=800, ph_st=190, ph_end=270, name_mod="2_B1839", show=True)
    #plot.single(s, start=2200, length=900, ph_st=190, ph_end=270, name_mod="2_B1839", show=True)
    #plot.single_b1839(s, start=2200, length=900, ph_st=190, ph_end=270, name_mod="2_B1839", modes=[bright, quiet], show=True)  # in the second paper
    #plot.single_b1839(s, start=0, length=900, ph_st=190, ph_end=270, name_mod="2_B1839", modes=[bright, quiet], show=True)
    #plot.single_b1839(s, start=3000, length=1000, ph_st=190, ph_end=270, name_mod="2_B1839", modes=[bright, quiet], show=True)
    #plot.lrfs(s, length=400, start=0, ph_st=190, ph_end=270, name_mod="2_B1839", show=False)
    #plot.folded(s, p3=12.13594191, start=0, length=400, ph_st=190, ph_end=270, name_mod="2_B1839", ybins=24)
    #plot.folded(s, p3=12.13594191, start=0, length=1000, ph_st=190, ph_end=270, name_mod="_B1839", ybins=24)
    #plot.folded_fit(s, p3=12.13594191, start=0, length=400, ph_st=190, ph_end=270, times=1, name_mod="2_B1839", ybins=24, pthres=0.5, rngs=[(8,27), (5, 23)])
    #plot.folded_fitseq(s, p3=12.13594191, start=0, length=400, ph_st=190, ph_end=270, times=2, name_mod="2_B1839", ybins=24, pthres=0.5)
    #plot.p3_evolution_b1839(s, length=256, step=1, start=900, ph_st=190, ph_end=270, name_mod="2_B1839", exp_range=(10,15), modes=[bright, quiet], show=True)
    #plot.p3_evolution_b1839(s, length=256, step=1, start=0, ph_st=190, ph_end=270, name_mod="2_B1839", exp_range=(10,15), modes=[bright, quiet], show=True) # in the second paper
    #plot.lrfs(s, length=416, start=0, ph_st=190, ph_end=270, name_mod="2_B1839", brightness=0.4, show=True) # in the second paper

    # 2005-05(598)
    #s.load('200505598.1380.singlepulse.gg', end=None, type='psrchive')  # nice? 250-500
    #plot.average(s, name_mod="3_B1839", show=False)
    #plot.single(s, start=0, length=2162, ph_st=200, ph_end=275, name_mod="3_B1839", show=True)
    #plot.lrfs(s, length=240, start=240, ph_st=200, ph_end=275, name_mod="3_B1839", show=False)
    #plot.folded(s, p3=12.2771528066, start=240, length=240, ph_st=200, ph_end=275, name_mod="3_B1839", ybins=24)
    #plot.p3_evolution_b1839(s, length=256, step=1, start=0, ph_st=200, ph_end=275, name_mod="3_B1839", exp_range=(10,15), show=True)

    # 2005-05(769)
    #s.load('200505769.1380.singlepulse.gg', end=None, type='psrchive')  # nice 1530-1970
    #plot.average(s, name_mod="4_B1839", show=False)
    #plot.single(s, start=0, length=4378, ph_st=0, ph_end=80, name_mod="4_B1839", show=True)
    #plot.lrfs(s, length=440, start=1530, ph_st=0, ph_end=80, name_mod="4_B1839", show=False)
    #plot.folded(s, p3=12.2771528066, start=1530, length=440, ph_st=0, ph_end=80, name_mod="4_B1839", ybins=24)
    #plot.p3_evolution_b1839(s, length=256, step=1, start=0, ph_st=0, ph_end=80, name_mod="4_B1839", exp_range=(10,15), show=True)


    # 2005-05(521)
    #s.data_dir = "data"
    #s.load('b1839.zap1', end=None, type='psrchive')
    #plot.average(s, start=0, name_mod="X_B1839", show=True)
    #plot.single(s, start=0, length=6008, ph_st=190, ph_end=270, name_mod="X_B1839", show=True)
    #plot.p3_evolution_b1839(s, length=256, step=1, start=0, ph_st=190, ph_end=270, name_mod="X_B1839", exp_range=(10,15), modes=[bright, quiet], show=True)
    #plot.p3_evolution_modes_b1839(s, length=128, step=1, start=0, ph_st=190, ph_end=270, name_mod="X_B1839", modes=[bright, quiet], show=True)

    # HERE 2
    #s.data_dir = "data"
    #s.load('b1839.p3fold', end=None, type='psrchive')  # this
    #s.load('b1839_11.p3fold', end=None, type='psrchive')
    #plot.average(s, start=0, name_mod="X_B1839", show=True)
    #plot.prefolded(s, start=0, ph_st=190, ph_end=270, name_mod="X_B1839")
    #plot.prefolded_fit_big(s, start=0, ph_st=190, ph_end=270, pthres=0.1, times=2, darkness=0.5, name_mod="X3_B1839", move=3) # this
    #plot.prefolded_fit(s, start=0, ph_st=190, ph_end=270, pthres=0.1, times=2, darkness=0.5, name_mod="X3_B1839", move=3)
    #plot.prefolded_fitseq(s, start=0, ph_st=190, ph_end=270, pthres=0.1,  times=3, st_inds=[[33, 39], [26, 30]], lens=[[17, 20], [16, 12]], seq=13, darkness=1.0, name_mod="X3_B1839")
    #plot.prefolded_fit(s, start=0, ph_st=190, ph_end=270, pthres=0.15, times=2, darkness=0.7, st_inds=[[24, 29], [23, 25]], lens=[[23, 15], [12, 12]], name_mod="X33_B1839")  # for b1839_11.p3fold
    #plot.prefolded_fitseq(s, start=0, ph_st=190, ph_end=270, pthres=0.15,  times=2, st_inds=[[28, 31], [23, 25]], lens=[[17, 15], [12, 12]], seq=15, darkness=1.0, name_mod="X3_B1839")  # for b1839_11.p3fold

    #s.load('b1839.zap1', end=None, type='psrchive')
    #s.load('b1839.zapped', end=None, type='psrchive')
    #s.load('b1839.zap2', end=None, type='psrchive')
    #plot.single_b1839(s, start=0, length=6008, ph_st=190, ph_end=270, name_mod="X_B1839", show=True)
    #plot.lrfs(s, length=415, start=0, ph_st=190, ph_end=270, name_mod="X_B1839", show=True)
    #plot.lrfs(s, length=1470, start=3400, ph_st=190, ph_end=270, name_mod="X_B1839", show=True)

    # HERE 3
    #s.data_dir = "data"
    #s.load('single_pulses.npy', type='simple')
    #plot.single(s, start=0, length=600, name_mod="E_B1839", show=True)
    #plot.lrfs(s, length=500, start=0, name_mod="E_B1839", show=False)
    #plot.folded(s, p3=12.2, start=0, length=None, name_mod="E_B1839", ybins=12)
    #plot.folded_fit(s, p3=11.3, start=0, length=None, times=2, name_mod="E_B1839", ybins=24, pthres=0.5, rngs=[(8,27), (5, 23)])

    # HERE 4
    #s.data_dir = "data"
    #s.load('single_pulses.p3fold', end=None, type='psrchive')  # this
    #plot.average(s, start=0, name_mod="Y_B1839", show=True)
    #plot.prefolded(s, start=0, name_mod="Y_B1839")
    #plot.prefolded_fit_model(s, start=0, pthres=0.05, times=2, darkness=0.5, name_mod="Y3_B1839", move=12) # this
    #plot.prefolded_fit_big(s, start=0, pthres=0.1, times=2, darkness=0.5, name_mod="Y3_B1839", move=3) # this



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


def b0943():
    s2 = SinglePulseAnalysis(data_dir='/data/leeuwen/drifting/0943+10/')
    s2.run1(files)

    s3 = SinglePulseAnalysis(data_dir='/data/leeuwen/drifting/0943+10/')
    s3.runs(files)

def j0815_rankin():
    s = SinglePulseAnalysis(data_dir='/data/szary/J0815+0939/data/Arecibo/')
    s.load('sJ0815+0939.54782ap.128bins', end=None, type='westerbork4')  # best?
    #s.load('sJ0815+0939.55937pa', end=None, type='westerbork4')  # not bad
    s.plot_all()

def j0815():
    s = SinglePulseAnalysis(data_dir='/data/szary/J0815+0939/data/p1769')
    s.load('0815.53161.puma.339.asc', end=None, type='westerbork')
    # new part
    #plot.single(s, start=320, length=520)
    #plot.lrfs(s, start=320, length=520)
    #plot.folded(s, p3=16.7, start=320, length=520)
    plot.p3_evolution(s, step=1, length=128, name_mod=1)  #, start=6200)
    #s.plot_all()

def j1705():
    s = SinglePulseAnalysis(data_dir='/data/szary/meerkat/J1705-1906/')
    #s.load('1070/*ar', end=None, type='meerkat')
    s.load('1498/*ar', end=None, type='meerkat')
    plot.single(s, start=0, brightness=0.5)
    plot.lrfs(s)
    #plot.folded(s, p3=10.0, name_mod="J1705")

def j0630():
    s = SinglePulseAnalysis(data_dir='/data/szary/meerkat/J0630-2834/')
    s.load('grand.debase.gg', end=None, type='psrchive')
    plot.single(s, start=0, ph_st=800, ph_end=1023, length=None, brightness=0.2)
    plot.lrfs(s, length=512, ph_st=800, ph_end=1023)
    plot.folded(s, p3=4.1, ybins=10, ph_st=800, ph_end=1023, name_mod="J0630")


def grand():
    s = SinglePulseAnalysis(data_dir='/data/szary/meerkat/')
    s.load('J1701-3006B/grand.rr', end=None, type='psrchive')
    ph_st = None
    ph_end = None
    plot.single(s, start=0, ph_st=ph_st, ph_end=ph_end, length=None, brightness=0.9)
    plot.lrfs(s, length=512, ph_st=ph_st, ph_end=ph_end)
    #plot.folded(s, p3=4.1, ybins=10, ph_st=ph_st, ph_end=ph_end, name_mod="J0630")




def main():
    #test_p3()
    #b1839()
    #j0815()
    #j0815_rankin()
    #b0943()
    #b1828()
    #j1705()
    #j0630()
    grand()
    print "Bye"


    files = ['./20111105/B0943+10_L33341_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc', './20111107/B0943+10_L33339_RSP0.PrepsubbNoclip.1_DM15.31.puma.119.gg.1pol.asc', './20111127/B0943+10_L35621_RSP0.PrepsubbZerodmNoclip_DM15.31.puma.119.gg.1pol.asc','./20111201/B0943+10_L36159_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20111204/B0943+10_L36157_RSP0.PrepsubbZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20111221/B0943+10_L39707_RSP0.ZerodmNoclip.1_DM15.31.puma.119.gg.1pol.asc','./20120111/B0943+10_L42350_RSP0.PrepdataNoclip.1.puma.119.gg.1pol.asc']

if __name__ == "__main__":
    main()
