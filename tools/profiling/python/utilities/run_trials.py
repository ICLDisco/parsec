#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import re
import shutil
from random import randint
import math
import cPickle
import online_math
import glob
from parsec_trials import ParsecTrial, ParsecTrialSet
import subprocess
from multiprocessing import Process, Pipe

##### global failure settings for trial set
max_rsd = 5 # anything above this and we want to re-run
max_stddev_fails = 4 # don't re-run too many times
max_set_fails = 5
max_trial_attempts = 20
do_trials = 1

# old ?
# pattern = re.compile(".*### TIMED\s(\d+\.\d+)\s+s.+?NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)",
#                      flags=re.DOTALL)
# new
pattern = re.compile(".* TIME\(s\)\s+(\d+\.\d+)\s+:\s+\w+\s+.+?N= \d+\s+NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)",
                     flags=re.DOTALL)

def spawn_trial_set_processes(trial_sets, exe_dir='.', output_base_dir='.'):
    last_N = 0
    last_ex = ''
    total_fail_count = 0
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for set_num in range(len(trial_sets)):
        if last_ex != trial_sets[set_num].ex:
            print('\n')
            print('############')
            print(trial_sets[set_num].ex)
            print('############')
            last_ex = trial_sets[set_num].ex
        if last_N != trial_sets[set_num].N:
            print('\n')
            print(' -_-_-_- {} -_-_-_- '.format(str(trial_sets[set_num].N)))
            last_N = trial_sets[set_num].N
        set_done = False
        fail_count = 0
        while not set_done:
            print()
            trial_set = trial_sets[set_num]
            try:
                my_end, their_end = Pipe()
                p = Process(target=run_trial_set_in_process,
                            args=(their_end, exe_dir, output_base_dir))
                p.start()
                my_end.send(trial_set)
                while p.is_alive():
                    p.join(2)
                if p.exitcode == 0:
                    set_done = True
                    trial_sets[set_num] = None
                else:
                    fail_count += 1
                    if fail_count < max_set_fails:
                        print('\n\nthe spawned process may have crashed. ' +
                              'Exit code was {}. Retrying!\n'.format(p.exitcode))
                    else:
                        trial_set.failed = True
                        set_done = True
            except Exception:
                import traceback
                traceback.print_exc()
                fail_count += 1
                if fail_count < max_set_fails:
                    print('An exception occurred during trial set ' +
                          '{} processing. Retrying!'.format(trial_set.unique_name()))
                else:
                    set_done = True
                    trial_set.failed = True
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('the trial set {} failed '.format(trial_set.unique_name()) +
                          'to successfully execute after {} failures.'.format(fail_count))

def run_trial_set_in_process(my_pipe, exe_dir='.', output_base_dir='.'):
    trial_set = my_pipe.recv()
    stddev_fails = 0
    set_finished = False
    extra_trials = []
    pending_filename = 'pending.' + trial_set.shared_name()

    while not set_finished:
        # abbrevs
        ex = trial_set.ex
        N = trial_set.N
        cores = trial_set.cores
        NB = trial_set.NB
        IB = trial_set.IB
        sched = trial_set.sched
        for trialNum in range(0, do_trials + stddev_fails):
            # in case of test executable crashes, prepare to run more than once
            for trial_attempts in range(max_trial_attempts):
                print("%s for %dx%d matrix on %d cores, NB = %d, IB = %d; sched = %s Xargs = %s trial #%d" %
                      (ex, N, N, cores, NB, IB, sched, str(trial_set.extra_args), trialNum))
                cmd, args = trial_set.generate_cmd()
                proc = subprocess.Popen([exe_dir + os.sep + cmd] + args,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # RUN
                (stdout, stderr) = proc.communicate()
                if len(stderr) != 0:
                    marker = randint(0, 99999)
                    # print("AN ERROR OCCURRED (random id: %d)" % marker)
                    # sys.stderr.write(str(marker) + ':\n' + stderr + '\n')
                match = pattern.match(stdout)
                if match:
                    # save successfully-parsed output
                    trial_set.NB = int(match.group(2))
                    perf = float(match.group(3))
                    time = float(match.group(1))
                    extra_output = match.group(4)
                    print("   -----> gflops: %f time: %f NB:%d" %
                          (perf, time, trial_set.NB))
                    trial = ParsecTrial(trial_set.ident, ex, N, cores, NB,
                                        IB, sched, trialNum, perf, time)
                    trial.extra_output = extra_output
                    if not os.environ.get('SUPPRESS_EXTRA_OUTPUT', None):
                       sys.stdout.write(extra_output)
                    # rename profile, if it exists
                    profiles = glob.glob(os.getcwd() + os.sep +
                                         'testing_' + ex + '*.prof-*')
                    if len(profiles) > 0:
                        for filename in profiles:
                            profile_filename = filename.replace('testing_' + ex,
                                                                output_base_dir
                                                                + os.sep
                                                                + trial.unique_name())
                            # print('moving {} to {}'.format(filename, profile_filename))
                            shutil.move(filename,  profile_filename)
                        # try:
                        #     import py_dbpreader as dbpr
                        #     trial.profile = dbpr.readProfile(profiles)
                        #     safe_unlink(profiles) # delete binary profile(s)
                        # except ImportError as iex:
                        #     print('Unable to rename profile')
                        #     print(iex)
                    trial_set.append(trial)
                    break # no more attempts are needed - we got what we came for
                else:
                    sys.stderr.write("results not properly parsed: %s\n" % stdout)
                    print('\nWe\'ll just try this one again.\n')
        print('done with trials in set.')
        # done with trials in set. now calculate set statistics
        perfSet = []
        timeSet = []
        for trial in trial_set:
            timeSet.append(trial.time)
            perfSet.append(trial.perf)
        variance, avgPerf = online_math.online_variance_mean(perfSet)
        perf_stddev = math.sqrt(variance)
        rsd = trial_set.percent_sdv(perf_stddev, avgPerf)
        # now check whether our results are clean/good
        if rsd <= max_rsd: # clean - save and print!
            trial_set.perf_sdv = perf_stddev
            trial_set.perf_avg = avgPerf
            variance, trial_set.time_avg = online_math.online_variance_mean(timeSet)
            trial_set.time_sdv = math.sqrt(variance)
            print(trial_set) # realtime progress report
            while not set_finished:
                try:
                    pfilename = output_base_dir + os.sep + trial_set.unique_name() + '.set'
                    pfile = open(pfilename, 'w')
                    trial_set.pickle(pfile)
                    # move 'pending' to 'rerun' in case a later re-run of the entire group is necessary
                    if os.path.exists(output_base_dir + os.sep + pending_filename):
                        shutil.move(output_base_dir + os.sep + pending_filename,
                                    output_base_dir + os.sep +
                                    pending_filename.replace('pending', 'rerun'))
                    safe_unlink([output_base_dir + os.sep +
                                 trial.unique_name() + '.trial'
                                 for trial in trial_set],
                                report_error=False)
                    set_finished = True
                except KeyboardInterrupt:
                    print('Currently writing files. Cannot interrupt.')

        elif stddev_fails < max_stddev_fails: # no good, try again
            stddev_fails += 1
            extra_trials.extend(trial_set[:])
            del trial_set[:] # retry with a clean set, in case there was interference
            print('WARNING: this trial set has a large relative ' +
                  'standard deviation ({}%), and will be redone.'.format(rsd))
        else: # no good, but we've tried too many times :(
            # let's just use all of our many results, and label the set with a warning
            trial_set.extend(extra_trials)
            perfSet = []
            timeSet = []
            for trial in trial_set:
                timeSet.append(trial.walltime)
                perfSet.append(trial.perf)
            variance, avgPerf = online_math.online_variance_mean(perfSet)
            perf_stddev = math.sqrt(variance)
            trial_set.perf_sdv = perf_stddev
            trial_set.perf_avg = avgPerf
            variance, trial_set.time_avg = online_math.online_variance_mean(timeSet)
            trial_set.time_sdv = math.sqrt(variance)
            wfile = open(output_base_dir + os.sep +
                         trial_set.unique_name() + '.warn', 'w')
            trial_set.pickle(wfile)
            # leave the pending pickle so it can be easily identified
            # as never having completed later on
            set_finished = True
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
if __name__ == '__main__':
    extra_args = []
    pickles = []
    try:
        argn = 1
        if os.path.isdir(sys.argv[argn]):
            exe_dir = sys.argv[argn]
            argn += 1
        else:
            exe_dir = '.'
        if not os.path.isfile(sys.argv[argn]):
            output_base_dir = sys.argv[argn]
            argn += 1
        else:
            output_base_dir = '.'
        for arg in sys.argv[argn:]:
            if os.path.exists(arg):
                pickles.append(arg)
            else:
                extra_args.append(arg)
    except:
        print('Usage: run_tests.py EXECUTABLE_DIRECTORY ' +
              'OUTPUT_DIRECTORY TEST_PICKLES_TO_RUN ' +
              '[EXTRA ARGUMENTS TO TEST EXECUTABLE]')
        sys.exit(-1)

    # print(glob.glob(os.getcwd() + os.sep + 'testing_*.prof-*'))
    # clean up old .profile files before testing
    safe_unlink(glob.glob(exe_dir + os.sep + 'testing_*.prof-*'))
    safe_unlink(glob.glob(os.getcwd() + os.sep + 'testing_*.prof-*'))

    if 'None' in extra_args:
        extra_args = None

    trial_sets = []
    for pickle in pickles:
        pfile = open(pickle, 'r')
        trial_set = ParsecTrialSet.unpickle(pfile)
        if extra_args == None or len(extra_args) > 0:
            trial_set.extra_args = extra_args
        trial_set.stamp_time() # timestamp the run time
        trial_sets.append(trial_set)

    trial_sets.sort(key = lambda tset: (tset.sched))
    # run the longer ones first
    trial_sets.sort(key = lambda tset: (tset.ex, tset.N, tset.NB, tset.IB), reverse=True)

    spawn_trial_set_processes(trial_sets, exe_dir = exe_dir, output_base_dir=output_base_dir)
