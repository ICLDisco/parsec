import datetime as dt
import time
import cPickle

class Trial(object):
    class_version = 1.0 # added a version
    class_version = 1.1 # renamed some attributes
    def __init__(self, ident, ex, N, cores, NB, IB, sched, trial_num, perf, walltime):
        self.__version__ = self.__class__.class_version
        self.ident = ident
        self.ex = ex
        self.N = int(N)
        self.cores = int(cores)
        self.NB = int(NB)
        self.IB = int(IB)
        self.sched = sched
        self.trial_num = int(trial_num)
        self.perf = float(perf)
        self.time = walltime
        self.extra_output = ''
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
        self.profile = None # may not have one
        self.profile_event_stats = None
    def stamp_time(self):
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
    def unique_name(self):
        return '{:_<6}_({}-{:0>3})_{:0>5}_{:0>4}_{:0>4}_{:_<3}_{:0>3}_{:0>3}_{:.2f}'.format(
            self.ex, self.ident, self.cores, self.N, self.NB, self.IB,
            self.sched, int(self.perf), self.trial_num, self.unix_timestamp)
    def __repr__(self):
        return self.uniqueName()
    def __setstate__(self, dictionary): # the unpickler shim
        self.__dict__.update(dictionary)
        if not hasattr(self, '__version__'):
            self.__version__ = 1.0
            if hasattr(self, 'name'):
                self.ex = self.name
            elif hasattr(self, 'executable'):
                self.ex = self.executable
            if hasattr(self, 'scheduler'):
                self.sched = self.scheduler
            if not hasattr(self, 'ident'):
                self.ident = 'NO_ID'
        if self.__version__ < 1.1:
            self.perf = self.gflops
            self.time = self.walltime
            self.unix_timestamp = self.unix_time
            self.iso_timestamp = self.timestamp
            self.extra_output = self.extraOutput
            self.__version__ = 1.1

class TrialSet(list):
    class_version = 1.0 # added a version
    class_version = 1.1 # renamed various attributes
    # class members
    __unloaded_profile_token__ = 'not loaded'
    @staticmethod
    def unpickle(filepath, load_profile=True):
        f = open(filepath, 'r')
        trial_set = cPickle.load(f)
        if load_profile:
            # load profiles, assign them to trials
            for trial in trial_set:
                if trial.profile == TrialSet.__unloaded_profile_token__:
                    trial.profile = cPickle.load(f)
        f.close()
        return trial_set
    # object members
    def pickle(self, filepath, protocol=cPickle.HIGHEST_PROTOCOL):
        f = open(filepath, 'w')
        profile_backups = []
        for trial in self:
            profile_backups.append(trial.profile)
            if trial.profile:
                trial.profile = TrialSet.__unloaded_profile_token__
        cPickle.dump(self, f, protocol)
        for profile in profile_backups:
            if profile: # don't dump the Nones
                cPickle.dump(profile, f, protocol)
        f.close()
        # restore profiles because the user isn't necessarily done with them
        for trial, profile in zip(self, profile_backups):
            trial.profile = profile

    def __init__(self, ident, ex, N, cores=0, NB=0, IB=0, sched='LFQ', extra_args=[]):
        self.__version__ = self.__class__.class_version
        self.ident = ident
        self.cores = int(cores)
        self.ex = ex
        self.N = int(N)
        self.NB = int(NB)
        self.IB = int(IB)
        self.sched = sched
        self.perf_avg = 0.0
        self.perf_sdv = 0.0
        self.time_avg = 0.0
        self.time_sdv = 0.0
        self.iso_timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_timestamp = int(time.time())
        self.extra_args = extra_args
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
    def __setstate__(self, dictionary): # the unpickler shim
        self.__dict__.update(dictionary)
        if not hasattr(self, '__version__'):
            if not hasattr(self, 'ident'):
                self.ident = 'NO_ID'
            self.__version__ = 1.0
        if self.__version__ < 1.1:
            self.perf_avg = self.avgGflops
            self.perf_sdv = self.Gstddev
            self.time_avg = self.avgTime
            self.time_sdv = self.Tstddev
            self.unix_timestamp = self.unix_time
            self.iso_timestamp = self.timestamp
            self.__version__ = 1.1
            


