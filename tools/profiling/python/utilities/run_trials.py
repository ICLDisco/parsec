#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import re
import datetime as dt
import time
import shutil
from random import randint
import cPickle
import glob
import subprocess
from multiprocessing import Process, Pipe

##### global failure settings for testing ######
max_rsd = 5 # anything above this and we want to re-run
max_stddev_fails = 4 # don't re-run too many times
max_trial_failures = 5
max_test_failures = 20
tests_per_trial = 3

# PaRSEC testing output pattern
test_output_pattern = (
    ".* TIME\(s\)\s+(\d+\.\d+)\s+:\s+\w+\s+.+?N= \d+\s+NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)")

class ParsecTest(object):
    class_version = 1.0 # revamped everything
    def __init__(self, ident, exe, N, cores, NB, IB, sched, perf, walltime, test_num):
        self.__version__ = self.__class__.class_version
        # parameters
        self.exe = exe
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
            self.exe, self.ident, self.cores, self.N, self.NB, self.IB,
            self.sched, self.test_num, int(self.perf), self.unix_timestamp)
    def __repr__(self):
        return self.uniqueName()

class ParsecTrial(list):
    class_version = 1.0 # revamped everything for pandas
    # class members
    __unloaded_profile_token__ = 'not loaded' # old
    @staticmethod
    def unpickle(_file, load_profile=True):
        trial = cPickle.load(_file)
        return trial
    # object members
    def pickle(self, _file, protocol=cPickle.HIGHEST_PROTOCOL):
        cPickle.dump(self, _file, protocol)
    def __init__(self, ident, exe, N, cores=0, NB=0, IB=0, sched='LFQ', extra_args=[]):
        self.__version__ = self.__class__.class_version
        # basic parameters (always present)
        self.cores = int(cores)
        self.exe = exe
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
        cmd = 'testing_' + self.exe
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
            self.exe, self.sched, self.N, self.cores, self.NB, self.IB) +
                ' @@@@@  time(sd/avg): {: >4.2f} / {: >5.1f} '.format(self.time_sdv, self.time_avg) +
                ' #####  gflops(sd/avg[rsd]): {: >4.2f} / {: >5.1f} [{: >2d}]'.format(
                 self.perf_sdv, self.perf_avg, self.percent_sdv()))
    def shared_name(self):
        return '{}_{:0>3}_{:_<6}_N{:0>5}_n{:0>4}_i{:0>4}_{:_<3}'.format(
            self.ident, self.cores,
            self.exe, self.N, self.NB, self.IB, self.sched)
    def name(self):
        return self.shared_name() + '_gfl{:0>3}_rsd{:0>3}_len{:0>3}'.format(
            int(self.perf_avg), self.percent_sdv(), len(self))
    def unique_name(self):
        return self.name() + '_' + str(self.unix_timestamp)



def spawn_test_processes(trials, tests_per_trial, exe_dir='.', out_dir='.',
                         max_rsd=max_rsd):
    last_N = 0
    last_exe = ''
    total_fail_count = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for trial_num in range(len(trials)):
        if last_exe != trials[trial_num].exe:
            print('\n')
            print('############')
            print(trials[trial_num].exe)
            print('############')
            last_exe = trials[trial_num].exe
        if last_N != trials[trial_num].N:
            print('\n')
            print(' -_-_-_- {} -_-_-_- '.format(str(trials[trial_num].N)))
            last_N = trials[trial_num].N
        trial_done = False
        fail_count = 0
        while not trial_done:
            print()
            trial = trials[trial_num]
            try:
                my_end, their_end = Pipe()
                p = Process(target=run_trial_in_process,
                            args=(their_end, trial, tests_per_trial,
                                  exe_dir, out_dir, max_rsd))
                p.start()
                while p.is_alive():
                    p.join(2)
                if p.exitcode == 0:
                    trial_done = True
                    trials[trial_num] = None
                else:
                    fail_count += 1
                    if fail_count < max_trial_failures:
                        print('\n\nthe spawned process may have crashed. ' +
                              'Exit code was {}. Retrying!\n'.format(p.exitcode))
                    else:
                        trial.failed = True
                        trial_done = True
            except Exception:
                import traceback
                traceback.print_exc()
                fail_count += 1
                if fail_count < max_trial_failures:
                    print('An exception occurred during trial ' +
                          '{} processing. Retrying!'.format(trial.unique_name()))
                else:
                    trial_done = True
                    trial.failed = True
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('the trial {} failed '.format(trial.unique_name()) +
                          'to successfully execute after {} failures.'.format(fail_count))

def run_trial_in_process(my_pipe, trial, tests_per_trial, exe_dir, out_dir, max_rsd):
    import online_math

    test_output_re = re.compile(test_output_pattern, flags=re.DOTALL)

    stddev_fails = 0
    trial_finished = False
    extra_tests = []

    while not trial_finished:
        # abbrevs
        exe = trial.exe
        N = trial.N
        cores = trial.cores
        NB = trial.NB
        IB = trial.IB
        sched = trial.sched
        for test_num in range(0, tests_per_trial + stddev_fails):
            # in case of test executable crashes, prepare to run more than once
            for test_attempts in range(max_test_failures):
                print("%s for %dx%d matrix on %d cores, NB = %d, IB = %d; sched = %s Xargs = %s trial #%d" %
                      (exe, N, N, cores, NB, IB, sched, str(trial.extra_args), test_num))
                cmd, args = trial.generate_cmd()
                proc = subprocess.Popen([exe_dir + os.sep + cmd] + args,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # RUN
                (stdout, stderr) = proc.communicate()
                if len(stderr) != 0:
                    marker = randint(0, 99999)
                    # print("AN ERROR OCCURRED (random id: %d)" % marker)
                    # sys.stderr.write(str(marker) + ':\n' + stderr + '\n')
                match = test_output_re.match(stdout)
                if match:
                    # save successfully-parsed output
                    trial.NB = int(match.group(2))
                    perf = float(match.group(3))
                    time = float(match.group(1))
                    extra_output = match.group(4)
                    print("   -----> gflops: %f time: %f NB:%d" %
                          (perf, time, trial.NB))
                    trial = ParsecTest(trial.ident, exe, N, cores, NB,
                                       IB, sched, perf, time, test_num)
                    trial.extra_output = extra_output
                    if not os.environ.get('SUPPRESS_EXTRA_OUTPUT', None):
                       sys.stdout.write(extra_output)
                    # rename profile, if it exists
                    profiles = glob.glob(os.getcwd() + os.sep +
                                         'testing_' + exe + '*.prof-*')
                    if len(profiles) > 0:
                        moved_profiles = list()
                        for filename in profiles:
                            profile_filename = filename.replace('testing_' + exe,
                                                                out_dir
                                                                + os.sep
                                                                + trial.unique_name())
                            # print('moving {} to {}'.format(filename, profile_filename))
                            shutil.move(filename,  profile_filename)
                            moved_profiles.append(profile_filename)
                        try:
                            import parsec_binprof as pbp
                            add_info = dict()
                            if trial.exe.endswith('potrf'):
                                precision = trial.exe.replace('potrf', '')[-1].upper()
                                add_info['POTRF_PRI_CHANGE'] = os.environ[precision + 'POTRF']
                            pbp.convert(moved_profiles, add_info=add_info)
                        except ImportError:
                            pass # can't convert... ahh well
                    trial.append(test)
                    break # no more attempts are needed - we got what we came for
                else:
                    sys.stderr.write("results not properly parsed: %s\n" % stdout)
                    print('\nWe\'ll just try this one again.\n')
        print('done with trial.')
        # done with trial. now calculate trial statistics
        test_perfs = []
        test_times = []
        for test in trial:
            test_times.append(test.time)
            test_perfs.append(test.perf)
        variance, avgPerf = online_math.online_variance_mean(test_perfs)
        perf_stddev = variance ** 0.5
        rsd = trial.percent_sdv(perf_stddev, avgPerf)
        # now check whether our results are clean/good
        if rsd <= max_rsd: # clean - save and print!
            trial.perf_sdv = perf_stddev
            trial.perf_avg = avgPerf
            variance, trial.time_avg = online_math.online_variance_mean(test_times)
            trial.time_sdv = variance ** 0.5
            print(trial) # realtime progress report
            while not trial_finished: # safe against Keyboard Interrupt
                try:
                    pfilename = out_dir + os.sep + trial.unique_name() + '.trial'
                    pfile = open(pfilename, 'w')
                    trial.pickle(pfile)
                    # move 'pending' to 'rerun' in case a later re-run of the entire group is necessary
                    if 'pending.' in trial.filename:
                        rerun = trial.filename.replace('pending.', 'rerun.')
                        shutil.move(trial.filename, rerun)
                    trial_finished = True
                    # safe_unlink([out_dir + os.sep +
                    #              test.unique_name() + '.test'
                    #              for test in trial],
                    #             report_error=False)
                except KeyboardInterrupt:
                    print('Currently writing files. Cannot interrupt.')

        elif stddev_fails < max_stddev_fails: # no good, try again
            stddev_fails += 1
            extra_tests.extend(trial[:])
            del trial[:] # retry with a clean trial, in case there was interference
            print('WARNING: this trial has a large relative ' +
                  'standard deviation ({}%), and will be redone.'.format(rsd))
        else: # no good, but we've tried too many times :(
            # let's just use all of our many results, and label the trial with a warning
            trial.extend(extra_tests)
            test_perfs = []
            test_times = []
            for test in trial:
                test_times.append(test.walltime)
                test_perfs.append(test.perf)
            variance, avgPerf = online_math.online_variance_mean(test_perfs)
            perf_stddev = variance ** 0.5
            trial.perf_sdv = perf_stddev
            trial.perf_avg = avgPerf
            variance, trial.time_avg = online_math.online_variance_mean(test_times)
            trial.time_sdv = variance ** 0.5
            wfile = open(out_dir + os.sep +
                         trial.unique_name() + '.warn', 'w')
            trial.pickle(wfile)
            # leave the pending pickle so it can be easily identified
            # as never having completed later on
            trial_finished = True
    # be done.
    return 0

def safe_unlink(files, report_error = True):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            if report_error:
                print('the file {} has apparently vanished.'.format(ufile))

###########
## MAIN
###########
def run_main():
    import argparse
    parser = argparser.ArgumentParser(description='Runs PaRSEC trials in various modes.')
    parser.add_argument('exe_dir')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--convert', action='store_true')
    parser.add_argument('-t', '--tests-per-trial', type=int, default=tests_per_trial)
    parser.add_argument('-s', '--max-relstddev', type=float, default=max_rsd)
    parser.add_argument('-u', '--unlink-existing', action='store_true',
                        help='Don\'t ask before unlinking existing profiles that might interfere.')
    parser.add_argument('-b', '--best', action='store_true',
                        help='Preserve only the trial with the best runtime. ' +
                        'Default is to preserve all runs.')
    parser.add_argument('the_rest', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    extra_args = []
    filenames = []

    for arg in args.the_rest:
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            extra_args.append(arg)
    if 'None' in extra_args: # a shortcut for removing even stored extra args
        extra_args = None

    # clean up old .prof files before testing
    old_profs = glob.glob(exe_dir + os.sep + 'testing_*.prof-*')
    if len(old_profs) > 0:
        unlink = 'y'
        if not args.unlink_existing:
            unlink = raw_input('Found {} existing profiles in the exe directory.\n' +
                               'These may confuse the program. Remove? [Y/n]: ') or 'y'
        if 'y' in unlink or 'Y' in unlink:
            safe_unlink(old_profs)
    old_profs = glob.glob(os.getcwd() + os.sep + 'testing_*.prof-*')
    if len(old_profs) > 0:
        unlink = 'y'
        if not args.unlink_existing:
            unlink = raw_input('Found {} existing profiles in the current working directory.\n' +
                               'These may confuse the program. Remove? [Y/n]: ') or 'y'
        if 'y' in unlink or 'Y' in unlink:
            safe_unlink(old_profs)

    trials = []
    for filename in filenames:
        if 'rerun.' in filename:
            pending_name = filename.replace('rerun.', 'pending.')
            shutil.move(filename, pending_name)
            filename = pending_name
        pfile = open(filename, 'r')
        trial = ParsecTrial.unpickle(pfile)
        trial.filename = filename
        if extra_args == None or len(extra_args) > 0:
            trial.extra_args = extra_args
        trial.stamp_time() # timestamp the run time
        trials.append(trial)

    trials.sort(key = lambda trial: (trial.sched))
    # run the longer ones first
    trials.sort(key = lambda trial: (trial.exe, trial.N, trial.NB, trial.IB), reverse=True)

    spawn_trial_processes(trials, exe_dir=exe_dir, out_dir=out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-use PaRSEC testing utility.')
    parser.add_argument('action', type=str, choices=['generate', 'run'],
                        help='This utility presently allows the generation or the' +
                        'subsequent execution of PaRSEC testing trials.\n' +
                        'The action keyword specifies the action to perform.')
    args = parser.parse_args()

    sys.argv.pop(1) # remove the 'action' from the arguments before calling a utility

    if args.action.startswith('gen'):
        # call generate
        pass
    elif args.action.startwith('run'):
        # call run
        run_main()

