#! /usr/bin/env python
__author__ = 'Andrzej Szary and Lucas Hermann Negri (peakutils)'
import os
import glob
import argparse

class SinglePulseAnalysis:

    def __init__(self, data_dir):
        self.data_dir = data_dir



def main():
    parser = argparse.ArgumentParser(prog='spa.py', description=u'Program for pulsar single-pulse analysis', add_help=True)
    parser.add_argument('-d', '--data', metavar='D', type=str, nargs='?', default='data', help=u'sets the data dir')

    args = parser.parse_args()

    s = SinglePulseAnalysis(data_dir=args.data)
    print "Bye"


if __name__ == "__main__":
    main()
