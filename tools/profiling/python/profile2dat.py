#!/usr/bin/python
import sys
import os
import shutil
import glob
import subprocess

defaultExecutable = 'dague-trunk/tools/profiling/profile2dat'

# note to self - this should eventually take two streams instead of two filenames
def profile2dat(profileFile, outputFile, executable = defaultExecutable, unlinkAfterProcessing = False):
    proc = subprocess.Popen([executable, profileFile],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = proc.communicate()
    output = open(outputFile, 'w')
    output.write(stdout)
    output.close()
    print('done processing ' + profileFile)
    if unlinkAfterProcessing:
        os.unlink(profileFile)
        print('unlinked ' + profileFile)

# assumes suffix of .profile, and uses filename minus .profile plus .dat as output filename
def profiles2dat(directory, outputDirectory = None, executable = defaultExecutable, unlink = False):
    if outputDirectory is None:
        outputDirectory = directory
    globThing = directory + '/' + '*.profile'
    profiles = glob.glob(globThing)
    for profile in profiles:
        profile2dat(profile, outputDirectory + '/' + 
                    os.path.basename(profile).rstrip('.profile')  + '.dat', 
                    executable, unlink)

if __name__ == '__main__':
    outputDir = sys.argv[1]
    executable = defaultExecutable
    unlink = True
    if len(sys.argv) > 2:
        outputDir = sys.argv[2]
    if len(sys.argv) > 3:
        executable = sys.argv[3]
    if len(sys.argv) > 4:
        arg4 = string.lower(sys.argv[4])
        if (arg4.startswith('unl') or
            arg4.startswith('del') or
            arg4.startswith('rem') or
            arg4.startswith('rm') or
            arg4.startswith('t')):
            unlink = True
    profiles2dat(sys.argv[1], outputDir, executable, unlink)
        
