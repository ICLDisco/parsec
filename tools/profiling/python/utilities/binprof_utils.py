#!/usr/bin/env python
import parsec_binprof as pbp
import parsec_profiling as p3
import sys, os, shutil, re # file utilities, etc.
import multiprocessing
import functools

filename_regex = re.compile('(\w+).*(\.prof|\.h5)(.*)-([a-zA-Z0-9]{6})')

default_name_infos = ['gflops', 'N', 'sched']
group_by_defaults = ['worldsize', 'SYNC_TIME_ELAPSED', 'cwd', 'exe']

class ProfAndName():
    def __init__(self, profile, filename):
        self.profile = profile
        self.filename = filename

def preprocess_profiles(filenames, name_infos=default_name_infos, dry_run=False,
                        enhance_filenames=True, convert=True, unlink=False, force_enhance=False):
    """ preprocesses PBP files. Intended for use everywhere a P3 is needed.

    Preprocesses Parsec Binary Profiles (PBPs) and returns a list of filename lists, if
    not asked to convert, or a list of filenames, if asked to convert to P3.
    Conversion to P3 is highly recommended.
    Each filename returned is suitable for loading info a ParsecProfile object with
    ParsecProfile.from_hdf(), and each filename list returned is suitable for passing
    to parsec_binprof.read().
    """

    fname_groups = group_profile_filenames(filenames)
    processed_filenames = list()

    # pass to converter function
    for fn_group in fname_groups:
        if enhance_filenames:
            fn_group = enhance_profile_filenames(fn_group, name_infos=name_infos,
                                                 dry_run=dry_run,
                                                 force_enhance=force_enhance)
        if convert:
            if not dry_run:
                fn_group = pbp.convert(fn_group, unlink=unlink, report_progress=True)
            else:
                print('Would convert {}'.format(fn_group))
        processed_filenames.append(fn_group)
    return processed_filenames

def group_profile_filenames(filenames, group_by=group_by_defaults):
    """ groups PBP filenames of different ranks

    This function is primarily necessary for use when dealing with
    PaRSEC Binary Profiles (PBPs) that are part of a distributed run.
    Since PaRSEC generates a separate PBP for each rank, it is necessary
    to match up these separate files when attempting to read the full PBP.
    However, the parsec_binprof module can read a single file at a time
    in order to gather the necessary information to match these files
    automatically, saving the user from unnecessary manual grouping.

    P3s that are passed to this function will be simply added as a
    single-element group and not processed further.
    """

    filename_groups = list()
    infonly_prof_groups = dict()
    for filename in filenames:
        if p3.p3_core in filename:
            # skip P3s
            filename_groups.append([filename])
        else:
            # initially group by start_time
            infonly_prof = pbp.get_info([filename])
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
        joint_prof = pbp.get_info(group_filenames)
        if joint_prof.last_error != 0:
            print('error! dbpreader.c error # {}'.format(joint_prof.last_error))
            print('retrying with apparently conflicting files...')
            # try again with excluded?
            for prof in prof_group.conflicts:
                group_filenames.append(prof.filename)
            joint_prof = pbp.get_info(group_filenames)
            if joint_prof.last_error != 0:
                print('error! dbpreader.c error # {}'.format(joint_prof.last_error))
                print('skipping this set of files:')
                for filename in group_filenames:
                    print('    ' + filename)
                continue # skip this set altogether
        filename_groups.append(group_filenames)
    return filename_groups

old_renamed_piece   = '.G-prof-'
def enhance_profile_filenames(filenames, name_infos=default_name_infos,
                              dry_run=False, force_enhance=False):
    """ Renames PBPs to a filename with more useful information than the default.

    Not designed to operate on lists of filenames that do not belong to the same profile,
    but may work anyway, depending on various factors.
    """
    renamed_files = list()
    if p3.p3_core in filenames[0]:
        profile = p3.ParsecProfile.from_hdf(filenames[0], skeleton_only=True)
    else:
        profile = pbp.get_info(filenames)
        if profile.last_error != 0:
            print('{} does not appear to be a reasonable set of filenames.'.format(filenames))
            if not force_enhance:
                return filenames
    # the profile should ALWAYS have a 'start_time' info
    new_chunk = str(profile.start_time) + '-'
    for info in name_infos:
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
    profile.close()

    for filename in filenames:
        full_filename = os.path.abspath(filename)
        dirname = os.path.dirname(full_filename)
        filename = os.path.basename(full_filename)
        match = filename_regex.match(filename)
        if match:
            new_filename = match.group(1).strip('_') + match.group(2) + '-' + new_chunk + match.group(4)
        else:
            # old-style (more likely to do over-renaming)
            if old_renamed_piece in filename:
                print('Warning - this file ({}) appears to have been renamed previously.'.format(filename))
                if not force_enhance:
                    print('Ignoring file {} to prevent from polluting the name.'.format(filename))
                    continue
            new_filename = filename.replace(pbp.pbp_core, old_renamed_piece + new_chunk)
        if not dry_run and filename != new_filename:
            if not os.path.exists(dirname + os.sep + new_filename):
                renamed_files.append(dirname + os.sep + new_filename)
                shutil.move(dirname + os.sep + filename,
                            dirname + os.sep + new_filename)
                # also rename already-converted files if they exist
                p3_name = filename.replace(pbp.pbp_core, p3.p3_core)
                new_p3_name = new_filename.replace(pbp.pbp_core, p3.p3_core)
                if (os.path.exists(dirname + os.sep + p3_name) and
                    not os.path.exists(dirname + os.sep + new_p3_name)):
                    print('Also renaming corresponding P3 file {}'.format(p3_name))
                    shutil.move(dirname + os.sep + p3_name,
                                dirname + os.sep + new_p3_name)
            else:
                print('WARNING: enhance would have overwritten file {} !'.format(dirname + os.sep +
                                                                                 new_filename))
                print('This move operation has been skipped.')
                renamed_files.append(dirname + os.sep + filename) # still return filename
        else:
            renamed_files.append(dirname + os.sep + filename)
            if filename != new_filename:
                print('Would move {} in directory {} '.format(filename, dirname + os.sep) +
                      'to {}.'.format(new_filename))
    return renamed_files

old_unrenamer = re.compile('(\w+)' + old_renamed_piece + '.*-(\w+)')
def revert_profile_filenames(filenames, dry_run=False):
    """ Reverts 'enhanced' filenames to their originals, where possible. """
    for filename in filenames:
        full_filename = os.path.abspath(filename)
        dirname = os.path.dirname(full_filename)
        filename = os.path.basename(full_filename)
        match = pbp_filename_regex.match(filename)
        if match:
            original_name = match.group(1).strip('_') + pbp.pbp_core + match.group(3)
        else:
            m = ungrouper.match(filename)
            if m:
                original_name = m.group(1).strip('_') + pbp.pbp_core + m.group(2)
            else:
                print('Could not figure out how to revert {} '.format(dirname + os.sep + filename) +
                      'to original name. Skipping...')
                continue
        if not dry_run and filename != original_name:
            if not os.path.exists(dirname + os.sep + original_name):
                shutil.move(dirname + os.sep + filename,
                            dirname + os.sep + original_name)
            else:
                print('WARNING: revert would have overwritten file {} !'.format(dirname + os.sep +
                                                                                original_name))
                print('This move operation has been skipped.')
        elif filename != original_name:
            print('Would move {} in directory {} '.format(filename, dirname + os.sep) +
                  'to {}.'.format(original_name))
        else:
            print('File {} already has its original name.'.format(original_name))

def autoload_profiles(filenames, convert=True, unlink=False,
                      enhance_filenames=False, skeleton_only=False):
    """ Provides a single interface for all attempts to load P3s from the filesystem.

    Whether from P3, PBP, or a combination of the two, you should be able to
    throw a huge list of filenames at this function and receive a whole mess of
    coherent P3 profiles back. Give it a whirl, and be sure to report any bugs!
    """
    profiles = list()

    groups_or_names = preprocess_profiles(filenames, convert=convert, unlink=unlink,
                                          enhance_filenames=enhance_filenames)

    # TODO: after preprocessing, should we check for a converted version anyway?

    if convert: # if we converted in the previous preprocessing step
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        print('loading P3s...')
        partial_from_hdf = functools.partial(p3.from_hdf, skeleton_only=skeleton_only)
        profiles = pool.map(partial_from_hdf, groups_or_names)
        print('loaded all P3s.')
        # profile = p3.ParsecProfile.from_hdf(group, skeleton_only=skeleton_only)
    else: # we didn't convert, so these are PBP filename lists
        for group in groups_or_names:
            import parsec_binprof as pbp # don't do this if not necessary
            print('loading PBP group {}'.format(group))
            profile = pbp.read(group, skeleton_only=skeleton_only)
            profiles.append(profile)
    return profiles

def get_partner_name(filename):
    if p3.p3_core in filename:
        return filename.replace(filename.replace(p3.p3_core, pbp.pbp_core))
    elif pbp.pbp_core in filename:
        return filename.replace(filename.replace(pbp.pbp_core, p3.p3_core))
    else:
        return filename

def compress_h5(filename, clevel=5):
    if p3.p3_core in filename:
        os.system('h5repack -f GZIP={} {} {}'.format(clevel, filename, filename + '.ctmp'))
        shutil.move(filename + '.ctmp', filename)

def print_help():
    print('')
    print('  -- DESCRIPTION -- ')
    print(' This utility allows the user to automatically group, and convert PaRSEC binary profiles.')
    print(' PBPs of different ranks but the same run share certain characteristics, ')
    print(' and this utility will attempt to group the profiles by those characteristics.')
    print(' Discovered groupings have their binary profiles renamed with shared descriptor strings,')
    print(' which means that even singleton binary profiles will benefit from this utility as,')
    print(' by default, the utility embeds the GFLOPS/s into the profile name, if available.')
    print('')
    print('By default, conversion to the Python pandas/HDF5 format (aka PPP) is done also.')
    print('This conversion may take some time, but it is generally recommended for ease of ')
    print('further interaction with the PaRSEC profile. To disable the conversion, ')
    print('pass the --no-convert flag to the utility.')
    print('')
    print(' usage: binprof_utils.py [--dry-run] [--no-convert] [--no-enhance] [--force-enhance]')
    print('                  [--unlink] [--unenhance] [FILENAMES] [extra naming keys]')
    print('')
    print(' Options and arguments:')
    print(' --unenhance   : used to revert the profiles to their original names')
    print(' --dry-run     : prints the file renamings without performing them')
    print(' --no-convert  : does not perform PBP->PPP conversion.')
    print('')
    print('  -- EXTRA NAMING KEYS -- ')
    print(' Any other argument not recognized as a filename will be used')
    print(' as an additional piece of the naming scheme, if a corresponding key')
    print(' can be found in the info dictionary of the profile group')
    print('')

if __name__ == '__main__':
    dry_run = False
    convert = True
    unlink = False
    enhance_filenames = True
    force_enhance = False

    filenames = list()
    args = list()

    for arg in sys.argv[1:]: # skip our executable name
        if os.path.exists(arg):
            filenames.append(arg)
        else:
            args.append(arg)

    if '--dry-run' in args:
        dry_run = True
        print('Doing a dry run...')
        args.remove('--dry-run')
    if '--revert' in args:
        args.remove('--revert')
        response = raw_input('Attempt to revert filenames to original names? [Y/n]: ')
        if 'n' not in response.lower():
            revert_profile_filenames(filenames, dry_run=dry_run)
        else:
            print('You chose not to revert the filenames. Utility now exiting.')
        sys.exit(0)
    if '--compress' in args:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(compress_h5, filenames)
        sys.exit(0)

    if '--help' in args:
        print_help()
        sys.exit(0)

    if '--force-enhance' in args:
        force_enhance = True
        args.remove('--force-enhance')
    if '--no-enhance' in args:
        enhance_filenames = False
        args.remove('--no-enhance')
    if '--no-convert' in args:
        convert = False
        args.remove('--no-convert')
    if '--unlink' in args:
        unlink = True
        args.remove('--unlink')

    if len(args) > 0:
        name_infos = args
    else:
        name_infos = default_name_infos

    preprocess_profiles(filenames, dry_run=dry_run, convert=convert,
                        enhance_filenames=enhance_filenames, unlink=unlink,
                        force_enhance=force_enhance,
                        name_infos=name_infos)
