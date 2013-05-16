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
from parsec_trials import Trial, TrialSet
import subprocess
from multiprocessing import Process, Pipe
from parsec_profile import *
# also uses dbpreader_py, if available

##### global failure settings for trial set
max_stddev_fails = 8 # don't re run too many times
max_rsd = 4 # anything above this and we want to re-run
max_trial_attempts = 20

pattern = re.compile(".*### TIMED\s(\d+\.\d+)\s+s.+?NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)",
                     flags=re.DOTALL)

def safe_unlink(files, report_error = True):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            if report_error:
                print('the file {} has apparently vanished.'.format(ufile))

####### global parameter defaults for ig            
# it would be nice to have different 'default experiment parameters'
# for different machines (e.g. ig, zoot).
#
# even nicer would be a more configurable way of generating trial sets.
# so that, for instance, I could specify different NB and IB params depending on
# the scheduler
default_NBs = {'dgeqrf': 192, 'dpotrf': 256, 'dgetrf': 256 }
default_IBdivs = {'dgeqrf': 8, 'dpotrf': 0, 'dgetrf': 0}
####### global parameter defaults
max_set_fails = 5
do_trials = 4

def generate_trial_sets(write_pending=True, extra_args = []):
    #
    # customize this section to your heart's content!
    #
    # defaults for ig:
    execs = ['dpotrf'] #, 'dgetrf_incpiv'] #, 'dpotrf', 'dgeqrf' ]
    schedulers = ['AP', 'GD', 'LTQ', 'LFQ', 'PBQ']
    minNumCores = 0 # default to using them all
    maxNumCores = 0
    minN = 6000
    maxN = 21400
    NBs = [160, 188, 200, 216, 256]
    IBdivs = None    # use defaults
    Ns = None        # generated based on tile size
    N_hi_mult = 20
    #
    # overrides
    #
#    IBdivs = [1,2,4,8,11] # None to use default per exec

#    IBdivs = [2, 4]
    NBs = [180, 380, 400]        # None to use defaults

#    Ns = [15360]
#    NBs = [180, 200, 360, 380]
#    NBs = [256]
#    IBdivs = [1,2,8]
    IBdivs = [0]
    #
    # end customizable param section
    #

    IBdivs_orig = IBdivs
    
    import socket
    hostname = socket.gethostname().split('.')[0]
    trial_sets = []

    if not Ns:
        generated_Ns = True
    else:
        generated_Ns = False
    # first, generate intended trial sets:
    for ex in execs:
        if not NBs:
            NBs = [default_NBs[ex]]
        for NB in NBs:
            if generated_Ns:
                if NB >= 256:
                    fact = NB
                else:
                    fact = 2 * NB
                Ns = list(range(NB*8 * N_hi_mult, minN, -fact*8))
                while Ns[0] >= maxN: # cutoff
                    Ns = Ns[1:]
            for N in Ns:
                sys.stderr.write("%s %d\n" % (ex.upper(), N))   
                if not IBdivs_orig:
                    IBdivs = [default_IBdivs[ex]]
                elif 'potrf' in ex or 'getrf' in ex:
                    IBdivs = [0] # because they don't use IB
                else:
                    IBdivs = IBdivs_orig
                for IBdiv in IBdivs:
                    if IBdiv > 0:
                        if NB % IBdiv == 0:
                            IB = NB / IBdiv
                        else:
                            continue # this one's not a fair combo
                    else:
                        IB = 0
                    for cores in range(minNumCores,maxNumCores + 1):
                        if cores == 0 and os.path.exists('/proc/cpuinfo'):
                            temp = subprocess.check_output(['grep',  '-c', '^processor',
                                                            '/proc/cpuinfo'])
                            try:
                                temp_int = int(temp)
                                cores = temp_int
                            except:
                                cores = 0 # didn't work, so back to default
                        for scheduler in schedulers:
                            if not os.path.isdir(outputBaseDir):
                                os.mkdir(outputBaseDir)
                            trial_set = TrialSet(hostname, ex, N, cores,
                                                 NB, IB, scheduler, extra_args)
                            print(trial_set.uniqueName() + ' ' + str(extra_args))
                            if write_pending:
                                # save planned file in case everything dies
                                trial_set.pickle(outputBaseDir + os.sep + 'pending.' +
                                                 trial_set.name())
                            trial_sets.append(trial_set)
    return trial_sets

def spawn_trial_set_processes(trial_sets, testingDir = '.'):
    last_N = 0
    last_ex = ''
    total_fail_count = 0
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
                p = Process(target=run_trial_set_in_process, args=(their_end, testingDir))
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
                          '{} processing. Retrying!'.format(trial_set.uniqueName()))
                else:
                    set_done = True
                    trial_set.failed = True
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('the trial set {} failed '.format(trial_set.uniqueName()) +
                          'to successfully execute after {} failures.'.format(fail_count))

def run_trial_set_in_process(my_pipe, testingDir='.'):
    trial_set = my_pipe.recv()
    stddev_fails = 0
    set_finished = False
    extra_trials = []
    # get this before any stats or sizes get added
    # this should someday be made more robust
    pending_filename = 'pending.' + trial_set.name()
    
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
                cmd, args = trial_set.genCmd()
                proc = subprocess.Popen([testingDir + os.sep + cmd] + args,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # RUN
                (stdout, stderr) = proc.communicate() 
                if len(stderr) != 0:
                    marker = randint(0, 99999)
                    print("AN ERROR OCCURRED %d" % marker)
                    sys.stderr.write(str(marker) + ':\n' + stderr + '\n')
                match = pattern.match(stdout)
                if match:
                    # save successfully-parsed output
                    trial_set.NB = int(match.group(2))
                    gflops = float(match.group(3))
                    time = float(match.group(1))
                    extraOutput = match.group(4)
                    print("   -----> gflops: %f time: %f NB:%d" %
                          (gflops, time, trial_set.NB))
                    trialObj = Trial(trial_set.ident, ex, N, cores, NB,
                                     IB, sched, trialNum, gflops, time)
                    trialObj.extraOutput = extraOutput
                    if not os.environ.get('SUPPRESS_EXTRA_OUTPUT', None):
                       sys.stdout.write(extraOutput)
                    # read and save profile, if it exists
                    profiles = glob.glob(testingDir + os.sep +
                                         'testing_' + ex + '*.profile')
                    if len(profiles) > 0:
                        # if False:
                        #     try:
                        #         my_end, their_end = Pipe()
                        #         p = Process(target=read_profile_in_process, args=(their_end,profiles))
                        #         p.start()
                        #         while p.is_alive():
                        #             if my_end.poll(1):
                        #                 trialObj.profile = my_end.recv()
                        #         if my_end.poll(1):
                        #             trialObj.profile = my_end.recv()
                        #         p.join()
                        #         if p.exitcode != 0 or trialObj.profile == None:
                        #             print('\n\nProfile-reading process may have failed. return code from process was {}\n'.format(p.exitcode))
                        #             print('profile null? {}'.format(trialObj.profile == None))
                        #             continue
                        #     except Exception:
                        #         import traceback
                        #         traceback.print_exc()
                        #         print('Unable to save profile.')
                        #         continue
                        # else:
                        # read profile in current process since dbpreader is now fixed
                        try:
                            import py_dbpreader as dbpr
                            profile = dbpr.readProfile(profiles)
                            trialObj.profile = profile
                            trialObj.profile_event_stats = profile.event_type_stats
                            safe_unlink(profiles) # delete binary profile(s)
                        except ImportError as iex:
                            print('Unable to save profile; dbpreader is unavailable:')
                            print(iex)
                    # saving this file takes unnecessarily long
                    # pickleFile = open(outputBaseDir + os.sep +
                    #                   trialObj.uniqueName() +'.trial', 'w')
                    # cPickle.dump(trialObj, pickleFile)
                    # pickleFile.close()
                    trial_set.append(trialObj)
                    break # no more attempts are needed - we got what we came for
                else:
                    sys.stderr.write("results not properly parsed: %s\n" % stdout)
                    print('\nWe\'ll just try this one again.\n')
        # done with trials in set. now calculate set statistics
        gflopsSet = []
        timeSet = []
        for trial in trial_set:
            timeSet.append(trial.walltime)
            gflopsSet.append(trial.gflops)
        variance, avgGflops = online_math.online_variance_mean(gflopsSet)
        gfl_stddev = math.sqrt(variance)
        rsd = trial_set.percentStdDev(gfl_stddev, avgGflops)
        # now check whether our results are clean/good
        if rsd <= max_rsd: # clean - save and print!
            trial_set.Gstddev = gfl_stddev
            trial_set.avgGflops = avgGflops
            variance, trial_set.avgTime = online_math.online_variance_mean(timeSet)
            trial_set.Tstddev = math.sqrt(variance)
            print(trial_set) # realtime progress report
            while not set_finished:
                try:
                    trial_set.pickle(outputBaseDir + os.sep + trial_set.uniqueName() + '.set')
                    # move 'pending' to 'rerun' in case a later re-run of the entire group is necessary
                    if os.path.exists(outputBaseDir + os.sep + pending_filename):
                        shutil.move(outputBaseDir + os.sep + pending_filename,
                                    outputBaseDir + os.sep +
                                    pending_filename.replace('pending', 'rerun'))
                    safe_unlink([outputBaseDir + os.sep +
                                 trial.uniqueName() + '.trial'
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
            gflopsSet = []
            timeSet = []
            for trial in trial_set:
                timeSet.append(trial.walltime)
                gflopsSet.append(trial.gflops)
            variance, avgGflops = online_math.online_variance_mean(gflopsSet)
            gfl_stddev = math.sqrt(variance)
            trial_set.Gstddev = gfl_stddev
            trial_set.avgGflops = avgGflops
            variance, trial_set.avgTime = online_math.online_variance_mean(timeSet)
            trial_set.Tstddev = math.sqrt(variance)
            trial_set.pickle(outputBaseDir + os.sep +
                             trial_set.uniqueName() + '.warn')
            # leave the pending pickle so it can be easily identified
            # as never having completed later on
            set_finished = True
    # be done.
    return 0

def read_profile_in_process(pipe, profiles):
    profile = None
    if len(profiles) > 0:
        try:
            import py_dbpreader as dbpr
            profile = dbpr.readProfile(profiles)
            safe_unlink(profiles) # delete extra files now
        except ImportError as iex:
            print('Unable to save profile; dbpreader is unavailable:')
            print(iex)
    pipe.send(profile)
    return 0
    
    

    
###########
## MAIN
###########

if __name__ == '__main__':
    list_only = False
    use_pending = False
    use_rerun = False
    use_files = False
    generate = False
    extra_args = None
    
    if len(sys.argv) > 1:
        testingDir = sys.argv[1]
    else:
        testingDir = '.'
    if len(sys.argv) > 2:
        outputBaseDir = sys.argv[2]
    else:
        outputBaseDir = '/mnt/scratch/pgaultne'
    if len(sys.argv) > 3:
        if 'l' in sys.argv[3].lower():
            list_only = True
        if 'p' in sys.argv[3].lower():
            use_pending = True
        if 'r' in sys.argv[3].lower():
            use_rerun = True
        if 'g' in sys.argv[3].lower():
            generate = True
        if 'f' in sys.argv[3].lower():
            use_files = True
    if len(sys.argv) > 4 and not use_files:
        if sys.argv[4].lower() == '-l2':
            extra_args = ['--mca-pins=papi_exec']
        elif sys.argv[4].lower() == '-l3':
            extra_args = ['--mca-pins=papi_socket']
        elif sys.argv[4].lower() == '-lboth':
            extra_args = ['--mca-pins=papi_socket,papi_exec']
        elif sys.argv[4].lower() == '-X':
            extra_args = []
        else:
            extra_args = sys.argv[4:]
            
    # clean up old .profile files before testing
    safe_unlink(glob.glob(testingDir + os.sep + 'testing_*.profile'))

    if generate or list_only:
        generate_trial_sets(write_pending = not list_only,
                                         extra_args = extra_args)
    if use_rerun or use_pending:
        pickles = []
        if use_rerun:
            while use_rerun:
                try:
                    reruns = glob.glob(outputBaseDir + os.sep + 'rerun.*')
                    for rerun in reruns:
                        shutil.move(rerun, rerun.replace('rerun', 'pending'))
                    use_rerun = False # we're done with it now
                except KeyboardInterrupt:
                    print('Currently moving files. Cannot interrupt.')
        if use_pending:
            pickles.extend(glob.glob(outputBaseDir + os.sep + 'pending.*'))
        if use_files:
            pickles.extend(sys.argv[4:])
        trial_sets = []
        for pickle in pickles:
            pfile = open(pickle, 'r')
            trial_set = cPickle.load(pfile)
            trial_set.extra_args = extra_args
            trial_set.new_timestamp() # timestamp the run time
            trial_sets.append(trial_set)
            pfile.close()
        trial_sets.sort(key = lambda tset: (tset.sched))
        trial_sets.sort(key = lambda tset: (tset.ex, tset.N, tset.NB, tset.IB), reverse=True)
            
        spawn_trial_set_processes(trial_sets, testingDir = testingDir)
