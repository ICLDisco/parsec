#!/usr/bin/env python

import sys
import ptt_utils
import pbt2ptt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a set of PaRSEC Binary Profile files into an HDF5 file")
    parser.add_argument('--output', dest='output', help='Output file name')
    parser.add_argument('--report-progress', dest='report_progress', action='store_const',
                        const=True, default=False,
                        help='Report progress of conversion to stderr')
    parser.add_argument('--single-process', dest='multiprocess', action='store_const',
                        const=False, default=True,
                        help='Deactivate multiprocess parallelism')
    parser.add_argument('--ptt-version', dest='pttversion', action='store',
                        default=1, type=int, choices=[1, 2],
                        help='PTT format version')
    parser.add_argument('--complib', dest='complib', action='store', default='blosc',
                        choices=['zlib', 'lzo', 'bzip2', 'blosc',
                                 'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc',
                                 'blosc:snappy', 'blosc:zlib', 'blosc:zstd'],
                        help='Compression library')
    parser.add_argument('--complevel', dest='complevel', action='store',
                        default=0, type=int, choices=list(range(0, 10)),
                        help='Compression level')
    parser.add_argument('inputs', metavar='INPUT', type=str, nargs='+',
                        help='PaRSEC Binary Profile Input files')
    args = parser.parse_args()

    if args.output is None:
        groups = ptt_utils.group_trace_filenames(args.inputs)
        for f in groups:
            print("Processing {}".format(f))
            name = pbt2ptt.convert(f, multiprocess=args.multiprocess, report_progress=args.report_progress,
                                   compress=(args.complib, args.complevel),
                                   version=args.pttversion)
            print("Generated: {}".format(name))
    else:
        f = args.inputs
        print("Processing {}".format(f))
        name = pbt2ptt.convert(f, multiprocess=args.multiprocess, report_progress=args.report_progress,
                               out=args.output, compress=(args.complib, args.complevel),
                               version=args.pttversion)
        print("Generated {}".format(name))
    sys.exit(0)
