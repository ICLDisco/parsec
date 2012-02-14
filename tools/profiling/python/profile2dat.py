#!/usr/bin/python
import sys
import os
import shutil
import glob
import subprocess
import traceback

defaultExecutable = os.path.dirname(sys.argv[0]) + '/../profile2dat'

# note to self - this should eventually take two streams instead of two filenames
def profile2dat(profileFile, outputFile, executable = defaultExecutable, unlinkAfterProcessing = False):
    proc = subprocess.Popen([executable, profileFile],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = proc.communicate()
    print('done processing ' + profileFile)
    output = open(outputFile, 'w')
    output.write(stdout)
    output.close()
    print('processed profile written to ' + outputFile)
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
        basename = os.path.basename(profile)
        basename = basename[:basename.rfind('.profile')]
        profile2dat(profile, outputDirectory + '/' + 
                    basename + '.dat', 
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
    try:
        profiles2dat(sys.argv[1], outputDir, executable, unlink)
    except:
        cla, exc, trbk = sys.exc_info()
        print (exc)
        traceback.print_tb(trbk)
        if not os.path.exists("/usr/bin/php"):
            print ("A PHP interpreter may not be available on this system.");
        
