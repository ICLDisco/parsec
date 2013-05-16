import datetime as dt
import time
import cPickle

class Trial(object):
    def __init__(self, ident, ex, N, cores, NB, IB, sched, trialNum, gflops, walltime):
        self.ident = ident
        self.ex = ex
        self.N = int(N)
        self.cores = int(cores)
        self.NB = int(NB)
        self.IB = int(IB)
        self.sched = sched
        self.trialNum = int(trialNum)
        self.gflops = float(gflops)
        self.walltime = walltime
        self.extraOutput = ''
        self.timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_time = int(time.time())
        self.profile = None # may not have one
    def timestamp(self):
        self.timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_time = int(time.time())
    def getExec(self):
        if self.ex:
            return self.ex
        elif self.name:
            return self.name
        elif self.executable:
            return self.executable
        else:
            return None
    def getSched(self):
        if self.sched:
            return self.sched
        elif self.scheduler:
            return self.scheduler
        else:
            return None
    def uniqueName(self):
        return '{:_<6}_({}-{:0>3})_{:0>5}_{:0>4}_{:0>4}_{:_<3}_{:0>3}_{:0>3}_{:.2f}'.format(
            self.ex, self.ident, self.cores, self.N, self.NB, self.IB,
            self.sched, int(self.gflops), self.trialNum, self.unix_time)
    def __repr__(self):
        return self.uniqueName()

class TrialSet(list):
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
        self.ident = ident
        self.cores = int(cores)
        self.ex = ex
        self.N = int(N)
        self.NB = int(NB)
        self.IB = int(IB)
        self.sched = sched
        self.avgGflops = 0.0
        self.Gstddev = 0.0
        self.avgTime = 0.0
        self.Tstddev = 0.0
        self.timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_time = int(time.time())
        self.extra_args = extra_args
    def new_timestamp(self):
        self.timestamp = dt.datetime.now().isoformat(sep='_')
        self.unix_time = int(time.time())
    def genCmd(self):
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
    def percentStdDev(self, stddev=None, avg=None):
        if not avg:
            avg = self.avgGflops
        if not stddev:
            stddev = self.Gstddev
        if avg == 0:
            return 0
        else:
            return int(100*stddev/avg)
    def __str__(self):
        return ('{} {: <3} N: {: >5} cores: {: >3} nb: {: >4} ib: {: >4} '.format(
            self.ex, self.sched, self.N, self.cores, self.NB, self.IB) +
                ' @@@@@  time(sd/avg): {: >4.2f} / {: >5.1f} '.format(self.Tstddev, self.avgTime) +
                ' #####  gflops(sd/avg[rsd]): {: >4.2f} / {: >5.1f} [{: >2d}]'.format(
                 self.Gstddev, self.avgGflops, self.percentStdDev()))
    def name(self):
        return '{}_{:0>3}_{:_<6}_N{:0>5}_n{:0>4}_i{:0>4}_{:_<3}_gfl{:0>3}_rsd{:0>3}_len{:0>3}'.format(
            self.ident if self.ident else 'TRIALSET', self.cores,
            self.ex, self.N, self.NB, self.IB, self.sched,
            int(self.avgGflops), self.percentStdDev(), len(self))
    def uniqueName(self):
        return self.name() + '_' + str(self.unix_time)


