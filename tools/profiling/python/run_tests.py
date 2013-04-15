#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import re
from random import randint
import math
import cPickle
import online_math
import glob
from parsec_trials import Trial, TrialSet
import subprocess
from multiprocessing import Process, Pipe
# also uses dbpreader_py, if available

##### global failure settings for trial set
max_stddev_fails = 8 # don't re run too many times
max_rsd = 4 # anything above this and we want to re-run
max_trial_attempts = 20

pattern = re.compile(".*### TIMED\s(\d+\.\d+)\s+s.+?NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)",
                     flags=re.DOTALL)

def safe_unlink(files):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            print('the file {} has apparently vanished.'.format(ufile))

####### global parameter defaults for ig            
# it would be nice to have different 'default experiment parameters'
# for different machines (e.g. ig, zoot).
#
# even nicer would be a more configurable way of generating trial sets.
# so that, for instance, I could specify different NB and IB params depending on
# the scheduler
default_NBs = {'dgeqrf': 192, 'dpotrf': 256, 'dgetrf': 256 }
default_IBdivs = {'dgeqrf': 8, 'dpotrf': 4, 'dgetrf': 8}
####### global parameter deftauls
max_set_fails = 5
do_trials = 4

def generate_trial_sets(write_pending=True):
    #
    # customize this section to your heart's content!
    #
    extra_args = [] # or, ['--mca-pins=papi_exec']
    # defaults for ig:
    execs = ['dgeqrf'] #, 'dpotrf', 'dgeqrf' ]
    schedulers = ['AP', 'GD', 'LTQ', 'LFQ', 'PBQ']
    minNumCores = 48 # default to using them all
    maxNumCores = 48
    maxN = 22000
    NBs = [160, 188, 200, 216, 256]
    IBdivs = None    # use defaults
    Ns = None        # generated based on tile size
    N_low_mult = 4
    N_hi_mult = 15
    #
    # overrides
    #
    IBdivs = [1,2,4,8,11] # None to use default per exec

    IBdivs = [2, 4]
    NBs = [168, 188, 256, 380, 400]        # None to use defaults

    Ns = [15360]
    NBs = [192]
    IBdivs = [1,2,8]
    #
    # end param section
    #
    
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
            print(NB)
            if generated_Ns:
                if NB >= 256:
                    fact = NB
                else:
                    fact = 2 * NB
                Ns = list(range(NB*8 * N_hi_mult, 8000, -fact*8))
                while Ns[0] >= maxN: # cutoff
                    Ns = Ns[1:]
            for N in Ns:
                sys.stderr.write("%s %d\n" % (ex.upper(), N))   
                if not IBdivs:
                    IBdivs = [default_IBdivs[ex]]
                for IBdiv in IBdivs:
                    if IBdiv > 0:
                        if NB % IBdiv == 0:
                            IB = NB / IBdiv
                        else:
                            continue # this one's not a fair combo
                    else:
                        IB = 0
                    for cores in range(minNumCores,maxNumCores + 1):
                        for scheduler in schedulers:
                            if not os.path.isdir(outputBaseDir):
                                os.mkdir(outputBaseDir)
                            trial_set = TrialSet(hostname, ex, N, cores,
                                                 NB, IB, scheduler, extra_args)
                            print(trial_set.uniqueName())
                            if write_pending:
                                # save planned file in case everything dies
                                pending = open(outputBaseDir + os.sep + 'pending.' +
                                               trial_set.uniqueName(), 'w')
                                cPickle.dump(trial_set, pending)
                                pending.close()
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
                p = Process(target=run_trial_set_in_process, args=(their_end,testingDir))
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
                          'to successfully execute after {} failures.'.format(failed_count))

def run_trial_set_in_process(my_pipe, testingDir='.'):
    trial_set = my_pipe.recv()
    stddev_fails = 0
    set_finished = False
    extra_trials = []
    # get this before any stats or sizes get added
    # this should someday be made more robust
    pending_filename = 'pending.' + trial_set.uniqueName()
    
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
                    sys.stdout.write(extraOutput)
                    # read and save profile, if it exists
                    profiles = glob.glob(testingDir + os.sep +
                                         'testing_' + ex + '*.profile')
                    if len(profiles) > 0:
                        try:
                            my_end, their_end = Pipe()
                            p = Process(target=read_profile_in_process, args=(their_end,profiles))
                            print('trying to read profile...')
                            p.start()
                            while p.is_alive():
                                if my_end.poll(1):
                                    trialObj.profile = my_end.recv()
                                    print('received profile!')
                            if my_end.poll(1):
                                print('received profile 2!')
                                trialObj.profile = my_end.recv()
                            p.join()
                            if p.exitcode != 0 or trialObj.profile == None:
                                print('\n\nProfile-reading process may have failed. return code from process was {}\n'.format(p.exitcode))
                                print('profile null? {}'.format(trialObj.profile == None))
                                continue
                        except Exception:
                            import traceback
                            traceback.print_exc()
                            print('Unable to save profile.')
                            continue
                    # and now pickle for posterity
                    pickleFile = open(outputBaseDir + os.sep +
                                      trialObj.uniqueName() +'.trial', 'w')
                    cPickle.dump(trialObj, pickleFile)
                    pickleFile.close()
                    trial_set.append(trialObj)
                    break # no more attempts are needed - we got what we came for
                else:
                    sys.stderr.write("results not properly parsed: %s\n" % stdout)
                    print('\nWe\'ll just try this one again...\n')
        # done with trials in set. now calculate statistics
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
            print(trial_set) # early progress report
            pfile = open(outputBaseDir + os.sep + trial_set.uniqueName() + '.set', 'w')
            cPickle.dump(trial_set, pfile)
            pfile.close()
            safe_unlink([outputBaseDir + os.sep +
                         trial.uniqueName() + '.trial' for trial in trial_set])
            safe_unlink([outputBaseDir + os.sep + pending_filename])
            set_finished = True
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
            warn_pickle = open(outputBaseDir + os.sep +
                               trial_set.uniqueName() + '.warn' , 'w')
            cPickle.dump(trial_set, warn_pickle)
            warn_pickle.close()
            # and leave the pending pickle so it can be easily re-run later
            # safe_unlink([outputBaseDir + os.sep + pending_filename])
            set_finished = True

    # be done.
    return 0

def read_profile_in_process(pipe, profiles):
    profile = None
    if len(profiles) > 0:
        try:
            print('reading.......')
            import dbpreader_py as dbpr
            profile = dbpr.readProfile(profiles)
            print('read! returning...' )
            safe_unlink(profiles) # delete extra files now
        except ImportError:
            print('Unable to save profile; dbpreader is unavailable')
    pipe.send(profile)
    return 0
    
    

    
###########
## MAIN
###########

if __name__ == '__main__':
    list_only = False
    use_pending = False
    generate_only = False
    extra_args = []
    
    if len(sys.argv) > 1:
        testingDir = sys.argv[1]
    else:
        testingDir = '.'
    if len(sys.argv) > 2:
        outputBaseDir = sys.argv[2]
    else:
        outputBaseDir = '/mnt/scratch/pgaultne'
    if len(sys.argv) > 3:
        if sys.argv[3].lower() == '-l':
            list_only = True
        elif sys.argv[3].lower() == '-p':
            use_pending = True
        elif sys.argv[3].lower() == '-g':
            generate_only = True
    if len(sys.argv) > 4:
        if sys.argv[4].lower() == '-l2':
            extra_args = ['--mca-pins=papi_exec']
        elif sys.argv[4].lower() == '-l3':
            extra_args = ['--mca-pins=papi_socket']
    # clean up old .profile files before testing
    safe_unlink(glob.glob(testingDir + os.sep + 'testing_*.profile'))

    if not use_pending:
        trial_sets = generate_trial_sets(write_pending = not list_only)
    else:
        trial_sets = []
        pending_pickles = glob.glob(outputBaseDir + os.sep + 'pending.*')
        for pickle in pending_pickles:
            pfile = open(pickle, 'r')
            trial_set = cPickle.load(pfile)
            trial_set.extra_args.extend(extra_args)
            trial_sets.append(trial_set)
            pfile.close()
        trial_sets.sort(key = lambda tset: (tset.sched))
        trial_sets.sort(key = lambda tset: (tset.N, tset.NB, tset.IB), reverse=True)
            
    if not (list_only or generate_only):
        spawn_trial_set_processes(trial_sets, testingDir = testingDir)
