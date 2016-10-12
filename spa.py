#! /usr/bin/env python
__author__ = 'Andrzej Szary'  # peakutils by Lucas Hermann Negri
import glob
import argparse

import modules.data as data

class SinglePulseAnalysis:

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.stokes_ = None
        self.off_rms_ = None
        self.base = None

        self.types = {'westerbork': data.load_westerbork, 'westerbork4': data.load_westerbork4}

    def load(self, file_name,  type='westerbork4'):
        try:
            fun = self.types[type]
        except:
            print 'Loading function not implemented.\nExiting...'
            exit()

        fun(self, file_name)



def main():
    s = SinglePulseAnalysis(data_dir='/data/szary/J0815+0939/data/')
    s.load('sJ0815+0939.54015ap', type='westerbork4')
    print "Bye"


if __name__ == "__main__":
    main()
