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
    def uniqueName(self):
        return '{:_<6}_({}-{:0>3})_{:0>5}_{:0>4}_{:0>4}_{:_<3}_{:0>3}_{:0>3}_{:.2f}.trial'.format(
            self.ex, self.ident, self.cores, self.N, self.NB, self.IB,
            self.sched, self.trialNum, int(self.gflops), self.unix_time)
    def __repr__(self):
        return self.uniqueName()

class TrialSet(list):
    def __init__(self, ident, ex, N, cores=0, NB=0, IB=0, sched='LFQ'):
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
        self.unix_time = time.time()
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
        return cmd, args
    def percentStdDev(self):
        return int(100*self.Gstddev/self.avgGflops) if self.avgGflops else 0
    def __str__(self):
        return ("%s %s N:%d cores:%d nb:%d ib:%d --- gflops(stddev/avg[perc]): %f / %f [%d] --- time(stddev/avg): %f / %f" %
                (self.ex, self.sched, self.N,
                 self.cores, self.NB, self.IB, self.Gstddev, self.avgGflops,
                 self.percentStdDev(), self.Tstddev, self.avgTime))
    def uniqueName(self):
        return '{}_{:0>3}_{:_<6}_{:0>5}_{:0>4}_{:0>4}_{:_<3}_gfl{:0>3}_rsd{:0>3}_len{:0>3}_{:.2f}.set'.format(
            self.ident if self.ident else 'TRIALSET', self.cores,
            self.ex, self.N, self.NB, self.IB, self.sched,
            int(self.avgGflops), self.percentStdDev(), len(self), self.unix_time)


