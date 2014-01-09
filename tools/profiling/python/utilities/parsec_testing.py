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

from common_utils import *

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


##### global defaults for testing #####
max_rsd = 5 # anything above this and we want to re-run the whole trial
tests_per_trial = 3
# failure (retry) defaults
max_stddev_fails = 4 # don't re-run forever, though
max_trial_failures = 5
max_test_failures = 2 # these shouldn't really fail

# PaRSEC testing output pattern
test_output_pattern = (
    ".* TIME\(s\)\s+(\d+\.\d+)\s+:\s+\w+\s+.+?N= \d+\s+NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)")

def spawn_trial_processes(trials, tests_per_trial, keep_best_test_only=False,
                          exe_dir='.', out_dir='.', max_rsd=max_rsd,
                          convert_profiles=True):
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
                # my_end, their_end = Pipe()
                p = Process(target=run_trial,
                            args=(trial, tests_per_trial, exe_dir, out_dir,
                                  max_rsd, keep_best_test_only, convert_profiles))
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
                safe_unlink(glob.glob( 'testing_' + exe + '*.prof-*'))
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

def run_trial(trial, tests_per_trial, exe_dir, out_dir,
              max_rsd, keep_best_test_only, convert_profiles):
    import online_math

    test_output_re = re.compile(test_output_pattern, flags=re.DOTALL)

    # abbrevs
    exe = trial.exe
    N = trial.N
    cores = trial.cores
    NB = trial.NB
    IB = trial.IB
    sched = trial.sched

    # counters and loop variables
    test_num = 0
    stddev_fails = 0
    trial_finished = False
    extra_tests = []

    while not trial_finished:
        test_attempts = 0
        while test_num < tests_per_trial + stddev_fails:
            if test_attempts > max_test_failures:
                test_num += 1
                test_attempts = 0
                print('Failed this test too many times. Moving on...')
                continue
            print("%s for %dx%d matrix on %d cores, NB = %d, IB = %d; sched = %s Xargs = %s trial #%d" %
                  (exe, N, N, cores, NB, IB, sched, str(trial.extra_args), test_num))
            cmd, args = trial.generate_cmd()
            proc = subprocess.Popen([exe_dir + os.sep + cmd] + args,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # RUN
            (stdout, stderr) = proc.communicate()
            # if len(stderr) != 0:
            #     marker = randint(0, 99999)
            #     print("AN ERROR OCCURRED (random id: %d)" % marker)
            #     sys.stderr.write(str(marker) + ':\n' + stderr + '\n')
            match = test_output_re.match(stdout)
            profile_filenames = glob.glob( 'testing_' + exe + '*.prof-*')
            if match:
                # save successfully-parsed output
                trial.NB = int(match.group(2))
                perf = float(match.group(3))
                time = float(match.group(1))
                extra_output = match.group(4)
                print("   -----> gflops: %f time: %f NB:%d" %
                      (perf, time, trial.NB))
                test = ParsecTest(trial.ident, exe, N, cores, NB,
                                  IB, sched, perf, time, test_num)
                test.extra_output = extra_output
                if not os.environ.get('SUPPRESS_EXTRA_OUTPUT', None):
                   sys.stdout.write(extra_output)
                # rename profile, if it exists
                if len(profile_filenames) > 0:
                    moved_profile_filenames = list()
                    for filename in profile_filenames:
                        profile_filename = filename.replace('testing_' + exe,
                                                            out_dir
                                                            + os.sep
                                                            + test.unique_name())
                        # print('moving {} to {}'.format(filename, profile_filename))
                        shutil.move(filename, profile_filename)
                        moved_profile_filenames.append(profile_filename)
                    profile_filenames = moved_profile_filenames
                trial.append((test, profile_filenames))
                test_num += 1 # no more attempts are needed - we got what we came for
            else:
                safe_unlink(profile_filenames)
                sys.stderr.write("results not properly parsed: %s\n" % stdout)
                print('\nWe\'ll just try this one again.\n')
        if tests_per_trial > 1:
            print('finished all {} tests of this trial'.format(len(trial)))
        # done with trial. now calculate trial statistics
        test_perfs = []
        test_times = []
        for test, profile_filenames in trial:
            test_times.append(test.time)
            test_perfs.append(test.perf)
        variance, avgPerf = online_math.online_variance_mean(test_perfs)
        perf_stddev = variance ** 0.5
        rsd = trial.percent_sdv(perf_stddev, avgPerf)
        # now check whether our results are clean/good
        if rsd <= max_rsd or keep_best_test_only: # clean - save and print!
            trial.perf_sdv = perf_stddev
            trial.perf_avg = avgPerf
            variance, trial.time_avg = online_math.online_variance_mean(test_times)
            trial.time_sdv = variance ** 0.5
            print(trial) # realtime progress report

            for test, profile_filenames in extra_tests:
                safe_unlink(profile_filenames) # these won't be needed anymore

            if keep_best_test_only:
                best_perf = 0
                best_index = 0
                for index, (test, profile_filenames) in enumerate(trial):
                    if test.perf > best_perf:
                        best_perf = test.perf
                        best_index = index
                print('Only keeping the profile of the test with the best performance' +
                      ' ({} gflops/s), at index {}.'.format(best_perf, best_index))

                new_list = list()
                for index, (test, profile_filenames) in enumerate(trial):
                    if index != best_index:
                        safe_unlink(profile_filenames) # remove profiles of 'not best' runs
                        new_list.append((test, list()))
                    else:
                        new_list.append((test, profile_filenames))
                del trial[:]
                trial.extend(new_list)

            if convert_profiles:
                # iterate through the list, convert the profiles, and save the new names
                new_list = list()
                while len(trial) > 0:
                    test, profile_filenames = trial.pop()
                    if len(profile_filenames) > 0:
                        try:
                            import parsec_binprof as pbp
                            add_info = add_info_to_profile(trial)
                            profile_filenames = [pbp.convert(profile_filenames,
                                                             add_info=add_info)]
                            new_list.append((test, profile_filenames))
                        except ImportError:
                            new_list.append((test, profile_filenames))
                            pass # can't convert without the module... ahh well
                    else:
                        new_list.append((test, profile_filenames))
                trial.extend(new_list) # put everything back in the trial

            while not trial_finished: # safe against Keyboard Interrupt
                try:
                    pfilename = out_dir + os.sep + trial.unique_name() + '.trial'
                    pfile = open(pfilename, 'w')
                    trial.pickle(pfile)
                    # move 'pending' to 'rerun' in case a later re-run of the entire group is necessary
                    if 'pending.' in trial.filename:
                        rerun = trial.filename.replace('pending.', 'rerun.')
                        if os.path.exists(trial.filename):
                            shutil.move(trial.filename, rerun)
                    trial_finished = True
                    # safe_unlink([out_dir + os.sep +
                    #              test.unique_name() + '.test'
                    #              for test in trial],
                    #             report_error=False)
                except KeyboardInterrupt:
                    print('Currently writing files. Cannot interrupt.')

        elif stddev_fails < max_stddev_fails: # no good, try again from beginning
            stddev_fails += 1
            test_num = 0
            extra_tests.extend(trial[:])
            del trial[:] # retry with a clean trial, in case there was interference
            print('WARNING: this trial has a large relative ' +
                  'standard deviation ({}%), and will be redone.'.format(rsd))
        else: # no good.... but we've tried too many times :(
            # (so let's just use all of our many results, and label the trial with a warning)
            trial.extend(extra_tests)
            test_perfs = []
            test_times = []
            for test, profile_filenames in trial:
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
    return trial


def add_info_to_profile(trial):
    add_info = dict()
    try:
        if trial.exe.endswith('potrf'):
            precision = trial.exe.replace('potrf', '')[-1].upper()
            add_info['POTRF_PRI_CHANGE'] = int(os.environ[precision + 'POTRF'])
    except KeyError as ke:
        add_info['POTRF_PRI_CHANGE'] = 0
        print(ke)
        print('Could not find', precision + 'POTRF', 'environment variable.\n',
              'Setting POTRF_PRI_CHANGE to 0.')
    return add_info

####### global parameter defaults for ig
# it would be nice to have different 'default experiment parameters'
# for different machines (e.g. ig, zoot).
#
# even nicer would be a more configurable way of generating trial sets.
# so that, for instance, I could specify different NB and IB params depending on
# the scheduler
default_test_NBs = {'dgeqrf': 192, 'dpotrf': 256, 'dgetrf': 256 }
default_IB_divs = {'dgeqrf': 8, 'dpotrf': 0, 'dgetrf': 0}
####### global parameter defaults
min_N = 12288
max_N = 12288
default_exes = ['dpotrf', 'dgeqrf', 'dgetrf' ]
default_scheds = ['AP', 'GD', 'LTQ', 'LFQ', 'PBQ']
default_NBs = [192, 256, 380]
N_hi_mult = 20


def generate_trials(out_dir, print_only=True, Ns=None, min_N=min_N, max_N=max_N,
                    NBs=default_NBs, exes=default_exes, scheds=default_scheds,
                    min_cores=0, max_cores=0, IB_divs=[0],
                    extra_args = []):
    IB_divs_orig = IB_divs

    import socket
    hostname = socket.gethostname().split('.')[0]
    trials = []

    if not Ns:
        generated_Ns = True
    else:
        generated_Ns = False
    # first, generate intended trial sets:
    for ex in exes:
        if not NBs:
            NBs = [default_NBs[ex]]
        for NB in NBs:
            if generated_Ns:
                if NB >= 256:
                    fact = NB
                else:
                    fact = 2 * NB
                Ns = list(range(NB*8 * N_hi_mult, min_N-1, -fact*8))
                print('generated ', Ns)
                if len(Ns) == 0:
                    Ns = [min_N]
                while len(Ns) > 1 and Ns[0] > max_N: # cutoff
                    Ns = Ns[1:]
            for N in Ns:
                sys.stderr.write("%s %d\n" % (ex.upper(), N))
                if not IB_divs_orig:
                    IB_divs = [default_IB_divs[ex]]
                elif 'potrf' in ex or 'getrf' in ex:
                    IB_divs = [0] # because they don't use IB
                else:
                    IB_divs = IB_divs_orig
                for IB_div in IB_divs:
                    if IB_div > 0:
                        if NB % IB_div == 0:
                            IB = NB / IB_div
                        else:
                            continue # this one's not a fair combo
                    else:
                        IB = 0
                    for cores in range(min_cores, max_cores + 1):
                        if cores == 0 and os.path.exists('/proc/cpuinfo'):
                            temp = subprocess.check_output(['grep',  '-c', '^processor',
                                                            '/proc/cpuinfo'])
                            try:
                                temp_int = int(temp)
                                cores = temp_int
                            except:
                                cores = 0 # didn't work, so back to default
                        for scheduler in scheds:
                            if not os.path.isdir(out_dir):
                                os.mkdir(out_dir)
                            trial = ParsecTrial(hostname, ex, N, cores,
                                                    NB, IB, scheduler, extra_args)
                            print(trial.shared_name() + ' ' + str(extra_args))
                            trials.append(trial)
    gen = 'yes'
    if print_only:
        gen = raw_input('Would you like to go ahead and generate the trial files [y/N]? ') or 'no'
    if 'y' in gen or 'Y' in gen:
        for trial in trials:
            file_ = open(out_dir + os.sep + 'pending.' +
                         trial.shared_name(), 'w')
            trial.pickle(file_)
            file_.close()
    return trials

###########
## MAIN(S)
###########
def run_main():
    import argparse
    parser = argparse.ArgumentParser(description='Runs PaRSEC trials in various modes.')
    parser.add_argument('exe_dir',
                        help='Directory containing PaRSEC testing executables.')
    parser.add_argument('out_dir',
                        help='Directory in which to place PaRSEC profiles and trial summaries.')
    parser.add_argument('-c', '--convert-profiles', action='store_true',
                        help='Convert PaRSEC Binary Profiles to Python HDF5 files.')
    parser.add_argument('-t', '--tests-per-trial', type=int, default=tests_per_trial)
    parser.add_argument('-s', '--max-rsd', type=float, default=max_rsd,
                        help='Maximum relative (% of avg) standard deviation for tests in trial.')
    parser.add_argument('-u', '--unlink-existing', action='store_true',
                        help='Don\'t ask before unlinking existing profiles that might interfere.')
    parser.add_argument('-b', '--best-test-only', action='store_true',
                        help='Preserve only the test with the best runtime ' +
                        '(default is to preserve all tests).')

    args, the_rest = parser.parse_known_args()
    extra_args = []
    filenames = []

    for arg in the_rest:
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            extra_args.append(arg)
    if 'None' in extra_args: # a shortcut for removing even stored extra args
        extra_args = None

    # clean up old .prof files before testing
    old_profs = glob.glob('testing_*.prof-*')
    if len(old_profs) > 0:
        unlink = 'y'
        if not args.unlink_existing:
            unlink = raw_input('Found {} existing profiles'.format(len(old_profs)) +
                               ' in the current working directory.\n' +
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

    spawn_trial_processes(trials, args.tests_per_trial, keep_best_test_only=args.best_test_only,
                          exe_dir=args.exe_dir, out_dir=args.out_dir, max_rsd=args.max_rsd,
                          convert_profiles=args.convert_profiles)

def generate_main():
    import argparse
    parser = argparse.ArgumentParser(description='Generates PaRSEC trials in various modes.')
    # parser.add_argument('out_dir', type=str, default='.', required=False)
    parser.add_argument('-p', '--print-only', action='store_true')
    parser.add_argument('-N', type=int, nargs='*', default=None)
    parser.add_argument('-Nrange', nargs='+', default=None)
    parser.add_argument('--maxN', type=int, default=max_N)
    parser.add_argument('--minN', type=int, default=min_N)
    parser.add_argument('-NB', nargs='+', default=default_NBs)
    parser.add_argument('-x', '--exes', nargs='+', default=default_exes)
    parser.add_argument('-o', '--scheds', nargs='+', default=default_scheds)
    args, extra_args = parser.parse_known_args()

    args.NB = smart_parse(args.NB)

    if args.Nrange and not args.N:
        args.N = smart_parse(args.Nrange)

    out_dir = '.'
    # for arg in args.extra_args:
    #     if os.path.isdir(arg):
    #         out_dir = arg
    #         args.extra_args.remove(arg)
    #         break

    trials = generate_trials(out_dir, print_only=args.print_only, max_N=max_N,
                             min_N=min_N, Ns=args.N, NBs=args.NB,
                             exes=args.exes, scheds=args.scheds,
                             extra_args=extra_args)
    print('Generated', len(trials), 'trials.')

if __name__ == '__main__':
    """This utility presently allows the generation or the
    subsequent execution of PaRSEC testing trials.\n
    The action keyword specifies the action to perform."""

    import argparse
    parser = argparse.ArgumentParser(description='Multi-use PaRSEC testing utility.')
                                     # add_help=False)
    parser.add_argument('action', type=str, choices=['gen', 'run'])
    parser.add_argument('action_args', metavar='Arguments to action', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sys.argv.pop(1) # remove the 'action' from the arguments before calling a utility

    if args.action.startswith('gen'):
        # call generate
        generate_main()
    elif args.action.startswith('run'):
        # call run
        run_main()
    else:
        print(__main__.__doc__)

