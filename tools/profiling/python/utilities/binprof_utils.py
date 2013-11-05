#!/usr/bin/env python
import parsec_binprof as p3_bin
import parsec_profiling as p3
import sys, os, shutil, re # file utilities, etc.

ungrouped_name_piece = '.prof-'
grouped_name_piece   = '.G-prof-'

class ProfAndName():
    def __init__(self, profile, filename):
        self.profile = profile
        self.filename = filename

# This function does some voodoo
def rename_with_strings(profile, filenames, infos=['gflops'], dry_run=False):
    new_chunk = grouped_name_piece + str(profile.start_time) + '-'
    for info in infos:
        try:
            value = profile.__getattr__(info)
        except AttributeError as ae:
            print(ae)
            continue
        try:
            if '.' in str(value):
                new_chunk += '{:.1f}'.format(value)
            else:
                new_chunk += str(value)
        except:
            new_chunk += str(value)
        new_chunk += '{}-'.format(info[:3])

    for filename in filenames:
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        filename = os.path.basename(filename)
        grouped_name = filename.replace(ungrouped_name_piece, new_chunk)
        if not dry_run:
            shutil.move(dirname + os.sep + filename,
                        dirname + os.sep + grouped_name)
        else:
            print('Would move {} '.format(dirname + os.sep + filename) +
                  'to {}.'.format(dirname + os.sep + grouped_name))

def group(filenames, dry_run=False, infos=['gflops']):
    infonly_prof_groups = dict()
    # initially group by start_time
    for filename in filenames:
        infonly_prof = p3_bin.get_info([filename])
        start_time = infonly_prof.information['start_time']
        if start_time not in infonly_prof_groups:
            infonly_prof_groups[start_time] = list()
        infonly_prof_groups[start_time].append(ProfAndName(infonly_prof, filename))
    # now that we've done the initial grouping, check for conflicts
    for key, prof_group in infonly_prof_groups.iteritems():
        comparers = {'worldsize': None, 'SYNC_TIME_ELAPSED': None,
                     'cwd': None, 'exe': None}
        for key in comparers.keys():
            if key not in prof_group[0].profile.information:
                del comparers[key] # eliminate things they don't have
            else:
                comparers[key] = prof_group[0].profile.information[key]
        for prof in prof_group[:1]:
            for key, value in comparers.iteritems():
                if prof.profile.information[key] != value:
                    print('We seem to have found a conflict.')
                    print('   the value {} of key {} '.format(value, key) +
                          'in file {} '.format(prof.filename) +
                          'does not match the value {} '.format(prof_group[0].profile.information[key]) +
                          'in file {}.'.format(prof_group[0].filename))
                    print('File {} will be skipped on the first grouping attempt.'.format(
                          prof.filename))
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
            group_filenames.append(prof.filename)
        joint_prof = p3_bin.get_info(group_filenames)
        if joint_prof.last_error != 0:
            print('error! dbpreader.c error # {}'.format(joint_prof.last_error))
            print('retrying with apparently conflicting files...')
            # try again with excluded?
            for prof in prof_group.conflicts:
                group_filenames.append(prof.filename)
            joint_prof = p3_bin.get_info(group_filenames)
            if joint_prof.last_error != 0:
                print('error! dbpreader.c error # {}'.format(joint_prof.last_error))
                print('skipping this set of files:')
                for filename in group_filenames:
                    print('    ' + filename)
                continue # skip this set altogether
        # now do the renaming stuff
        rename_with_strings(joint_prof, group_filenames, dry_run=dry_run)

ungrouper = re.compile('(\w+)' + grouped_name_piece + '.*-(\w+)')
def ungroup(filenames, dry_run=False):
    for filename in filenames:
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        filename = os.path.basename(filename)
        m = ungrouper.match(filename)
        if m:
            original_name = m.group(1) + ungrouped_name_piece + m.group(2)
            if not dry_run:
                shutil.move(dirname + os.sep + filename,
                            dirname + os.sep + original_name)
            else:
                print('Would move {} '.format(dirname + os.sep + filename) +
                      'to {}.'.format(dirname + os.sep + original_name))
        else:
            print('Could not figure out how to revert {} '.format(dirname + os.sep + filename) +
                  'to original name. Skipping...')

name_grouper = re.compile('(\w+)\.G-prof-(\d+)-.*\w+')
def group_by_name(filenames):
    by_start_time = dict()
    singletons = list()
    for filename in filenames:
        m = name_grouper.match(filename)
        if m:
            name = m.group(1) + m.group(2)
            try:
                by_start_time[name].append(filename)
            except:
                by_start_time[name] = [filename]
        else: # assume singleton
            singletons.append([filename])
    return by_start_time.values() + singletons

def autoload_profiles(filenames, convert=True, unlink=False):
    groups = group_by_name(filenames)
    profiles = list()

    for group in groups:
        h5_name = group[0].replace('.prof-', '.h5-')
        if len(group) == 1 and '.h5-' in group[0]:
            print('loading H5 {}'.format(group[0]))
            profile = p3.ParsecProfile.from_hdf(group[0])
        elif len(group) == 1 and '.prof-' in group[0] and os.path.exists(h5_name):
            print('auto-selecting h5 version of profile '+ group[0])
            profile = p3.ParsecProfile.from_hdf(h5_name)
        else:
            import parsec_binprof as p3_bin # don't do this if not necessary
            if convert:
                print('converting binprof group {} to H5 format'.format(group[0]))
                profile = p3_bin.convert(group, unlink=unlink)
            else:
                print('reading binprof group {}'.format(group[0]))
                profile = p3_bin.read(group)
        profiles.append(profile)
    return profiles

def print_help():
    print('')
    print('  -- DESCRIPTION -- ')
    print(' This utility allows the user to automatically group and rename PaRSEC binary profiles.')
    print(' PBPs of different ranks but the same run share certain characteristics, ')
    print(' and this utility will attempt to group the profiles by those characteristics.')
    print(' Discovered groupings have their binary profiles renamed with shared descriptor strings,')
    print(' which means that even singleton binary profiles will benefit from this utility as,')
    print(' by default, the utility embeds the GFLOPS/s into the profile name, if available.')
    print('')
    print(' usage: p3_group_profiles.py [--dry-run] [--ungroup] [FILENAMES] [naming keys]')
    print('')
    print(' Options and arguments:')
    print(' --ungroup   : used to revert the profiles to their original names')
    print(' --dry-run   : prints the file renamings without performing them')
    print('')
    print('  -- NAMING KEYS -- ')
    print(' Any other argument not recognized as a filename will be used')
    print(' as an additional piece of the naming scheme, if a corresponding key')
    print(' can be found in the info dictionary of the profile group')
    print('')

if __name__ == '__main__':
    # get filenames
    filenames = list()
    already_grouped_filenames = list()
    args = list()
    dry_run = False
    for arg in sys.argv[1:]: # skip our executable name
        if os.path.exists(arg):
            if ungrouped_name_piece in arg:
                filenames.append(arg)
            elif grouped_name_piece in arg:
                already_grouped_filenames.append(arg)
        else:
            args.append(arg)

    if '--help' in args:
        print_help()
        sys.exit(0)
    if '--dry-run' in args:
        dry_run = True
        print('Doing a dry run...')
        args.remove('--dry-run')
    if '--ungroup' in args:
        args.remove('--ungroup')
        print('Attempting to revert filenames to originals...')
        ungroup(already_grouped_filenames, dry_run=dry_run)
    else:
        # pass to converter function
        if len(args) > 0:
            group(filenames, dry_run=dry_run, infos=args)
        else:
            group(filenames, dry_run=dry_run)

