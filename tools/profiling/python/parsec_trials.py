import datetime as dt
import time

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
        self.unix_time = time.time()
        self.profile = None # may not have one
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
    def uniqueName(self):
        return '{}_{:0>3}_{:_<6}_{:0>5}_{:0>4}_{:0>4}_{:_<3}_gfl{:0>3}_rsd{:0>3}_len{:0>3}_{}'.format(
            self.ident if self.ident else 'TRIALSET', self.cores,
            self.ex, self.N, self.NB, self.IB, self.sched,
            int(self.avgGflops), self.percentStdDev(), len(self), self.unix_time)


