#!/usr/bin/env python
import parsec_binprof as p3_bin
import parsec_profiling as p3
import sys, os, shutil, re # file utilities, etc.

profile_name_core = '.prof-'
old_renamed_piece   = '-pbp.prof-'

default_name_infos = ['gflops', 'N', 'sched']
group_by_defaults = ['worldsize', 'SYNC_TIME_ELAPSED', 'cwd', 'exe']

class ProfAndName():
    def __init__(self, profile, filename):
        self.profile = profile
        self.filename = filename

pbp_filename_regex = re.compile('(\w+)\.prof(.*)-([a-zA-Z0-9]{6})')

# renames raw binary profiles that are part of a single overall profile
# into names that are easier to read in a shell.
# NOT designed to operate on lists of filenames that do not belong to the same profile.
def rename_with_strings(filenames, infos=default_name_infos, dry_run=False, force=False):
    renamed_files = list()
    profile = p3_bin.get_info(filenames)
    if profile.last_error != 0:
        print('{} does not appear to be a reasonable set of filenames.'.format(filenames))
        if not force:
            return -1
    # the profile should ALWAYS have a 'start_time' info
    new_chunk = str(profile.start_time) + '-'
    for info in infos:
        try:
            value = profile.__getattr__(info)
        except AttributeError as ae:
            print(ae)
            continue
        new_chunk += '{}'.format(info[:3].lower())
        try:
            if '.' in str(value):
                new_chunk += '{:.1f}'.format(value)
            else:
                new_chunk += str(value).upper()
        except:
            new_chunk += str(value).upper()
        new_chunk += '-'

    for filename in filenames:
        full_filename = os.path.abspath(filename)
        dirname = os.path.dirname(full_filename)
        filename = os.path.basename(full_filename)
        match = pbp_filename_regex.match(filename)
        if match:
            new_filename = match.group(1) + profile_name_core + new_chunk + match.group(3)
        else:
            # old-style (more likely to do over-renaming)
            if grouped_name_piece in filename:
                print('Warning - this file ({}) appears to have been renamed previously.'.format(filename))
                if not force:
                    print('Ignoring file {} to prevent from polluting the name.'.format(filename))
                    continue
            new_filename = filename.replace(profile_name_core, grouped_name_piece + new_chunk)
        if not dry_run:
            renamed_files.append(dirname + os.sep + new_filename)
            shutil.move(dirname + os.sep + filename,
                        dirname + os.sep + new_filename)
        else:
            renamed_files.append(dirname + os.sep + filename)
            print('Would move {} '.format(dirname + os.sep + filename) +
                  'to {}.'.format(dirname + os.sep + new_filename))
    return renamed_files

def group(filenames, group_by=group_by_defaults):
    filename_groups = list()
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
        # set up grouper
        comparers = dict()
        for key in group_by:
            comparers[key] = None
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
        filename_groups.append(group_filenames)
    return filename_groups

old_unrenamer = re.compile('(\w+)' + old_renamed_piece + '.*-(\w+)')
def unrename(filenames, dry_run=False):
    for filename in filenames:
        full_filename = os.path.abspath(filename)
        dirname = os.path.dirname(full_filename)
        filename = os.path.basename(full_filename)
        match = pbp_filename_regex.match(filename)
        if match:
            original_name = match.group(1) + profile_name_core + match.group(3)
        else:
            m = ungrouper.match(filename)
            if m:
                original_name = m.group(1) + profile_name_core + m.group(2)
            else:
                print('Could not figure out how to revert {} '.format(dirname + os.sep + filename) +
                      'to original name. Skipping...')
                continue
        if not dry_run:
            shutil.move(dirname + os.sep + filename,
                        dirname + os.sep + original_name)
        else:
            print('Would move {} '.format(dirname + os.sep + filename) +
                  'to {}.'.format(dirname + os.sep + original_name))

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

def autoload_profiles(filenames, convert=True, unlink=False, info_only=False):
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
                profile = p3_bin.read(group, info_only=info_only)
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
    args = list()
    dry_run = False
    no_convert = False
    unlink = False
    for arg in sys.argv[1:]: # skip our executable name
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            args.append(arg)

    if '--help' in args:
        print_help()
        sys.exit(0)
    if '--dry-run' in args:
        dry_run = True
        print('Doing a dry run...')
        args.remove('--dry-run')
    if '--unrename' in args:
        args.remove('--unrename')
        print('Attempting to revert filenames to originals...')
        unrename(filenames, dry_run=dry_run)
    if '--no-convert' in args:
        no_convert = True
    if '--unlink' in args:
        unlink = True
    else:
        groups = group(filenames)
        # pass to converter function
        for group in groups:
            if len(args) > 0:
                renamed_files = rename_with_strings(group, infos=args, dry_run=dry_run)
            else:
                renamed_files = rename_with_strings(group, dry_run=dry_run)
            if not no_convert and not dry_run:
                print('Converting {}'.format(renamed_files))
                p3_bin.convert(renamed_files, unlink=unlink)

