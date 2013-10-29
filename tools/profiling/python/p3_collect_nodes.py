#!/usr/bin/env python
import parsec_binprof as p3_bin
import parsec_profiling as p3
import sys, os, shutil # file utilities, etc.

ungrouped_name_piece = '.prof-'
grouped_name_piece   = '.G-prof-'

class StderrCapture(object):
    def __init__(self):
        self.captured = ''
    def write(self, data):
        self.captured += str(data)

# This function does some voodoo
def rename_with_strings(prof_group, str_pieces):
    new_chunk = grouped_name_piece
    for piece in str_pieces:
        new_chunk += str(piece) + '-'
    for prof in prof_group:
        new_name = prof['filename'].replace(ungrouped_name_piece, new_chunk)
        shutil.move(prof['filename'], new_name)

def group(filenames):
    infonly_prof_groups = dict()
    # initially group by start_time
    for filename in filenames:
        infonly_prof = p3_bin.get_info([filename])
        print(infonly_prof.last_error)
        start_time = infonly_prof.information['start_time']
        if start_time not in infonly_prof_groups:
            infonly_prof_groups[start_time] = list()
        infonly_prof_groups[start_time].append({'prof':infonly_prof,
                                                'filename':filename})
    print('')
    # now that we've done the initial grouping, check for conflicts
    for key, prof_group in infonly_prof_groups.iteritems():
        print(key)
        comparers = {'worldsize': None, 'SYNC_TIME_ELAPSED': None,
                     'cwd': None, 'exe': None}
        for key in comparers.keys():
            if key not in prof_group[0]['prof'].information:
                del comparers[key] # eliminate things they don't have
            else:
                comparers[key] = prof_group[0]['prof'].information[key]
        for prof in prof_group[:1]:
            for key, value in comparers.iteritems():
                if prof['prof'].information[key] != value:
                    print('We seem to have found a conflict.')
                    try:
                        prof_group.conflicts.append(prof)
                    except:
                        prof_group.conflicts = [prof]
                    prof_group.remove(prof)

        # now do the 'real' test: try to make a joint profile out
        # of these files. If there are no warnings, then we're in
        # good shape. If there are warnings, try again with
        # the excluded ones?
        group_filenames = []
        for prof in prof_group:
            group_filenames.append(prof['filename'])
        joint_prof = p3_bin.get_info(group_filenames)
        if joint_prof.last_error != 0:
            print('error! dbpreader.c error # {}'.format(joint_prof.last_error))
            # try again with excluded?
            for prof in prof_group.conflicts:
                group_filenames.append(prof['filename'])
                prof_group.append(prof) # for later renaming
            joint_prof = p3_bin.get_info(group_filenames)
            if joint_prof.last_error != 0:
                print('error! dbpreader.c error # {}'.format(joint_prof.last_error))
                print('skipping this set of files:')
                for filename in group_filenames:
                    print('    ' + filename)
                continue # skip this set altogether
        # now do the renaming stuff
        str_pieces = [prof_group[0]['prof'].start_time]
        if 'SYNC_TIME_ELAPSED' in comparers:
            str_pieces.append(int(prof_group[0]['prof'].SYNC_TIME_ELAPSED))
        rename_with_strings(prof_group, str_pieces)

if __name__ == '__main__':
    # get filenames
    filenames = list()
    args = list()
    for arg in sys.argv[1:]: # skip our executable name
        print(arg)
        if os.path.exists(arg) and ungrouped_name_piece in arg:
            filenames.append(arg)
        else:
            args.append(arg)
    # pass to converter function
    group(filenames)

