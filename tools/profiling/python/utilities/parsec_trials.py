from __future__ import print_function
import datetime as dt
import time
import cPickle
from parsec_profiling import *

class ParsecTest(object):
    class_version = 1.0 # revamped everything
    def __init__(self, ident, ex, N, cores, NB, IB, sched, test_num, perf, walltime):
        self.__version__ = self.__class__.class_version
        # parameters
        self.ex = ex
        self.N = int(N)
        self.cores = int(cores)
        self.NB = int(NB)
        self.IB = int(IB)
        self.sched = sched
        # identifiers
        self.ident = ident
        self.test_num = int(test_num)
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
        # output
        self.perf = float(perf)
        self.time = walltime
        self.extra_output = ''
    def stamp_time(self):
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
    def unique_name(self):
        return '{:_<6}_{}-{:0>3}_{:0>5}_{:0>4}_{:0>4}_{:_<3}_{:0>3}_{:0>3}_{:.2f}'.format(
            self.ex, self.ident, self.cores, self.N, self.NB, self.IB,
            self.sched, self.test_num, int(self.perf), self.unix_timestamp)
    def __repr__(self):
        return self.uniqueName()

class ParsecTrial(list):
    class_version = 1.0 # revamped everything for pandas
    # class members
    __unloaded_profile_token__ = 'not loaded' # old
    @staticmethod
    def unpickle(_file, load_profile=True):
        trial_set = cPickle.load(_file)
        return trial_set
    # object members
    def pickle(self, _file, protocol=cPickle.HIGHEST_PROTOCOL):
        cPickle.dump(self, _file, protocol)
    def __init__(self, ident, ex, N, cores=0, NB=0, IB=0, sched='LFQ', extra_args=[]):
        self.__version__ = self.__class__.class_version
        # basic parameters (always present)
        self.cores = int(cores)
        self.ex = ex
        self.N = int(N)
        self.NB = int(NB)
        self.IB = int(IB)
        self.sched = sched
        # extra parameters (could eventually be split into true Python parameters)
        self.extra_args = extra_args
        # identifiers
        self.ident = ident
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
        # stats
        self.perf_avg = 0.0
        self.perf_sdv = 0.0
        self.time_avg = 0.0
        self.time_sdv = 0.0
    def stamp_time(self):
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
    def generate_cmd(self):
        cmd = 'testing_' + self.ex
        args = []
        args.append('-N')
        args.append(str(self.N))
        args.append('-o')
        args.append(self.sched)
        if self.cores > 0:
            args.append('-c')
            args.append(str(self.cores))
        if self.NB > 0:
            args.append('-NB')
            args.append(str(self.NB))
            if self.IB > 0: # don't define IB without defining NB
                args.append('-IB')
                args.append(str(self.IB))
        if self.extra_args:
            args.extend(self.extra_args)
        return cmd, args
    def percent_sdv(self, stddev=None, avg=None):
        if not avg:
            avg = self.perf_avg
        if not stddev:
            stddev = self.perf_sdv
        if avg == 0:
            return 0
        else:
            return int(100*stddev/avg)
    def __str__(self):
        return ('{} {: <3} N: {: >5} cores: {: >3} nb: {: >4} ib: {: >4} '.format(
            self.ex, self.sched, self.N, self.cores, self.NB, self.IB) +
                ' @@@@@  time(sd/avg): {: >4.2f} / {: >5.1f} '.format(self.time_sdv, self.time_avg) +
                ' #####  gflops(sd/avg[rsd]): {: >4.2f} / {: >5.1f} [{: >2d}]'.format(
                 self.perf_sdv, self.perf_avg, self.percent_sdv()))
    def shared_name(self):
        return '{}_{:0>3}_{:_<6}_N{:0>5}_n{:0>4}_i{:0>4}_{:_<3}'.format(
            self.ident, self.cores,
            self.ex, self.N, self.NB, self.IB, self.sched)
    def name(self):
        return self.shared_name() + '_gfl{:0>3}_rsd{:0>3}_len{:0>3}'.format(
            int(self.perf_avg), self.percent_sdv(), len(self))
    def unique_name(self):
        return self.name() + '_' + str(self.unix_timestamp)


