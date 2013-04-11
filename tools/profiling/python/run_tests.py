#!/usr/bin/env python

import sys
import os
import re
from random import randint
import math
import cPickle
import online_math
import glob
from parsec_trials_2_0 import Trial, TrialSet
import subprocess
from multiprocessing import Process, Pipe

max_perc_sd = 3 # anything above this and we want to re-run
max_stddev_fails = 4 # but don't re run too many times
max_set_fails = 5
max_total_fails = 40
max_trial_attempts = 3
pattern = re.compile("### TIMED\s(\d+\.\d+)\s+s.+?NB=\s+(\d+).+?(\d+\.\d+)\s+gflops\n(.*)", flags=re.DOTALL)

# it would be nice to have different 'default experiment parameters'
# for different machines (e.g. ig, zoot).
#
# even nicer would be a more configurable way of generating trial sets.
# so that, for instance, I could specify different NB and IB params depending on
# the scheduler
#
# not to mention, this would be a great way of having the script "re-do"
# trial sets that had unacceptable levels of variation
# by not saving the information to disk and simply re-running the trial set
default_NBs = {'dgeqrf': 192, 'dpotrf': 256, 'dgetrf': 256 }
default_IBdivs = {'dgeqrf': 8, 'dpotrf': 4, 'dgetrf': 8}

def safe_unlink(files):
    for ufile in files:
        try:
            os.unlink(ufile) # no need to have them hanging around anymore
        except OSError:
            print('the file {} has apparently vanished.'.format(ufile))

def run_trial_set_in_process(my_pipe):
    import dbpreader_py as dbpr
    trial_set = my_pipe.recv()
    stddev_fails = 0
    
    while True:
        name = trial_set.name
        N = trial_set.N
        cores = trial_set.cores
        NB = trial_set.NB
        IB = trial_set.IB
        sched = trial_set.sched
        for trialNum in range(0, numTrials + stddev_fails):
            # in case of test executable crashes, prepare to run more than once
            for trial_attempts in range(max_trial_attempts): 
                print("%s for %dx%d matrix on %d cores, NB = %d, IB = %d; sched = %s trial #%d" %
                      (name, N, N, cores, NB, IB, sched, trialNum))
                cmd, args = trial_set.genCmd()
                proc = subprocess.Popen([testingDir + os.sep + cmd] + args,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (stdout, stderr) = proc.communicate()
                if len(stderr) != 0:
                    marker = randint(0, 99999)
                    print("AN ERROR OCCURRED %d" % marker)
                    sys.stderr.write(str(marker) + ':\n' + stderr + '\n')
                match = pattern.match(stdout)
                if match:
                    trial_set.NB = int(match.group(2))
                    gflops = float(match.group(3))
                    time = float(match.group(1))
                    extraOutput = match.group(4)
                    print("   -----> gflops: %f time: %f NB:%d" % (gflops, time, trial_set.NB))
                    trialObj = Trial(name, N, cores, NB, IB, sched, trialNum, gflops, time)
                    trialObj.extraOutput = extraOutput
                    sys.stdout.write(extraOutput)
                    # read and save profile, if it exists
                    profiles = glob.glob(testingDir + os.sep + 'testing_' + name + '*.profile')
                    if len(profiles) > 0:
                        trialObj.profile = dbpr.readProfile(profiles)
                        # now delete the files to make room
                        safe_unlink(profiles)
                    # and now pickle for posterity
                    pickleFile = open(outputBaseDir + os.sep + trialObj.uniqueName(), 'w')
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
            timeSet.append(trial.time)
            gflopsSet.append(trial.gflops)
        trial_set.Gstddev, trial_set.avgGflops = online_math.online_variance_mean(gflopsSet)
        trial_set.Tstddev, trial_set.avgTime = online_math.online_variance_mean(timeSet)
        trial_set.Gstddev = math.sqrt(trial_set.Gstddev)
        trial_set.Tstddev = math.sqrt(trial_set.Tstddev)
        print trial_set # early progress report
        
        # now check whether our results are clean/good
        if (trial_set.percentStdDev() <= max_perc_sd
            and stddev_fails < max_stddev_fails):
            pfile = open(outputBaseDir + os.sep + trial_set.uniqueName(), 'w')
            cPickle.dump(trial_set, pfile)
            pfile.close()
            # delete individual pickles
            safe_unlink([outputBaseDir + os.sep +
                         trial.uniqueName() for trial in trial_set])
            break # out of outer while True loop b/c we're done!!
        else: # if they're no good, try again
            stddev_fails += 1
            print('WARNING: this trial set has excessive variation, and will be redone.')
            safe_unlink([outputBaseDir + os.sep +
                         trial.uniqueName() for trial in trial_set])
            # replace trial set with clean one
            trial_set = TrialSet(trial_set.name, trial_set.N,
                                 trial_set.cores, trial_set.NB,
                                 trial_set.IB, trial_set.sched)
    # be done.
    return 0

if __name__ == '__main__':
    names = ['dpotrf'] #, 'dpotrf', 'dgeqrf' ]
    NBs = [180, 192, 200, 220, 256, 380]        # None to use defaults
    NBs = [380]
    IBdivs = None      # to use defaults
    Ns = None        # to use generated defaults
    minNumCores = 48 # default to using them all
    maxNumCores = 48
    numTrials = 4
    schedulers = ['AP', 'GD', 'LTQ', 'LFQ', 'PBQ']
    # overrides
    IBdivs = [2,4,8] # [1,2,4,8,11] # None to use default per exec
    #
    IBdivs = [2]
    Ns = [10000]
    Ns = None
    
    if len(sys.argv) > 1:
        testingDir = sys.argv[1]
    else:
        testingDir = '.'
    if len(sys.argv) > 2:
        outputBaseDir = sys.argv[2]
    else:
        outputBaseDir = '/mnt/scratch/pgaultne'

    # clean up old .profile files before testing
    safe_unlink(glob.glob(testingDir + os.sep + 'testing_*.profile'))
    
    trial_sets = []
    # first, generate intended trial sets:
    for name in names:
        if not NBs:
            NBs = [default_NBs[name]]
        for NB in NBs:
            if not Ns:
                Ns = reversed(range(NB*10 * 2, NB*10 * 7, NB*10))
            if not IBdivs:
                IBdivs = [default_IBdivs[name]]
            for IBdiv in IBdivs:
                if IBdiv > 0:
                    if NB % IBdiv == 0:
                        IB = NB / IBdiv
                    else:
                        continue # this one's not a fair combo
                else:
                    IB = 0
                for N in Ns:
                    sys.stderr.write("%s %d\n" % (name.upper(), N))   
                    for cores in range(minNumCores,maxNumCores + 1):
                        for scheduler in schedulers:
                            if not os.path.isdir(outputBaseDir):
                                os.mkdir(outputBaseDir)
                            trial_sets.append(TrialSet(name, N, cores, NB, IB, scheduler))
                            print(trial_sets[-1].uniqueName())
    # now run all trial sets
    last_N = 0
    last_name = ''
    total_fail_count = 0
    for set_num in range(len(trial_sets)):
        if last_name is not trial_sets[set_num].name:
            print('\n')
            print('############')
            print(trial_sets[set_num].name)
            print('############')
            last_name = trial_sets[set_num].name
        if last_N is not trial_sets[set_num].N:
            print('\n')
            print(' -_-_-_- {} -_-_-_- '.format(str(trial_sets[set_num].N)))
            last_N = trial_sets[set_num].N
        set_done = False
        fail_count = 0
        while not set_done:
            trial_set = trial_sets[set_num]
            try:
                my_end, their_end = Pipe()
                p = Process(target=run_trial_set_in_process, args=(their_end,))
                p.start()
                my_end.send(trial_set)
                while p.is_alive():
                    p.join(10)
                if p.exitcode == 0:
                    set_done = True
                    trial_sets[set_num] = None
                else:
                    fail_count += 1
                    if fail_count < max_set_fails:
                        print('\nthe spawned process may have crashed. ' +
                              'Exit code was {}. Retrying!'.format(p.exitcode))
                    else:
                        trial_set.failed = True
                        set_done = True
            except Exception:
                import traceback
                traceback.print_exc()
                fail_count += 1
                if fail_count < max_set_fails:
                    print('An exception occurred during trial set ' + 
                          '{} processing. Retrying!'.format(trial_st.uniqueName()))
                else:
                    set_done = True
                    trial_set.failed = True
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('the trial set {} failed '.format(trial_set.uniqueName()) +
                          'to successfully execute after {} failures.'.format(failed_count))

    found_failed = False
    for trial_set in trial_sets:
        if trial_set:
            if not trial_set.failed:
                print('well, that is weird. a trial set didn\'t fail but is still here?')
            else:
                print(trial_set.uniqueName() + ' failed.')
                found_failed = True
    if found_failed:
        import datetime as dt
        # pickle these completely failed sets so they can be easily re-run later
        cPickle.dump(trial_sets, open('failed_trial_sets_{}.pickle'.format(
            str(dt.datetime.now().isoformat(sep='_')) , 'w')))
