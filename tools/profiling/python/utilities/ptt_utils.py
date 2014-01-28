#!/usr/bin/env python
import pbt2ptt as pbt
import parsec_trace_tables as ptt
import sys, os, shutil, re # file utilities, etc.
import multiprocessing
import functools

filename_regex = re.compile('(\w+).*(\.prof|\.h5)(.*)-([a-zA-Z0-9]{6})')

default_name_infos = ['gflops', 'N', 'sched']
group_by_defaults = ['worldsize', 'SYNC_TIME_ELAPSED', 'cwd', 'exe']

class TraceAndName():
    def __init__(self, trace, filename):
        self.trace = trace
        self.filename = filename

def preprocess_traces(filenames, name_infos=default_name_infos, dry_run=False,
                        enhance_filenames=True, convert=True, unlink=False, force_enhance=False):
    """ preprocesses PBT files. Intended for use everywhere a PTT is needed.

    Preprocesses Parsec Binary Traces (PBTs) and returns a list of filename lists, if
    not asked to convert, or a list of filenames, if asked to convert to PTT.
    Conversion to PTT is highly recommended.
    Each filename returned is suitable for loading info a ParsecTraceTables object with
    ParsecTraceTables.from_hdf(), and each filename list returned is suitable for passing
    to pbt2ptt.read().
    """

    fname_groups = group_trace_filenames(filenames)
    processed_filenames = list()

    # pass to converter function
    for fn_group in fname_groups:
        if enhance_filenames:
            fn_group = enhance_trace_filenames(fn_group, name_infos=name_infos,
                                                 dry_run=dry_run,
                                                 force_enhance=force_enhance)
        if convert:
            if not dry_run:
                fn_group = pbt.convert(fn_group, unlink=unlink, report_progress=True)
            else:
                print('Would convert {}'.format(fn_group))
        processed_filenames.append(fn_group)
    return processed_filenames

def group_trace_filenames(filenames, group_by=group_by_defaults):
    """ groups PBT filenames of different ranks

    This function is primarily necessary for use when dealing with
    PaRSEC Binary Traces (PBTs) that are part of a distributed run.
    Since PaRSEC generates a separate PBT for each rank, it is necessary
    to match up these separate files when attempting to read the full PBT.
    However, the pbt2ptt module can read a single file at a time
    in order to gather the necessary information to match these files
    automatically, saving the user from unnecessary manual grouping.

    PTTs that are passed to this function will be simply added as a
    single-element group and not processed further.
    """

    filename_groups = list()
    infonly_trace_groups = dict()
    for filename in filenames:
        if ptt.ptt_core in filename:
            # skip PTTs
            filename_groups.append([filename])
        else:
            # initially group by start_time
            infonly_trace = pbt.read([filename], skeleton_only=True)
            start_time = infonly_trace.information['start_time']
            if start_time not in infonly_trace_groups:
                infonly_trace_groups[start_time] = list()
            infonly_trace_groups[start_time].append(TraceAndName(infonly_trace, filename))

    # now that we've done the initial grouping, check for conflicts
    for key, trace_group in infonly_trace_groups.iteritems():
        # set up grouper
        comparers = dict()
        for key in group_by:
            comparers[key] = None
        for key in comparers.keys():
            if key not in trace_group[0].trace.information:
                del comparers[key] # eliminate things they don't have
            else:
                comparers[key] = trace_group[0].trace.information[key]
        for trace in trace_group[:1]:
            for key, value in comparers.iteritems():
                if trace.trace.information[key] != value:
                    print('We seem to have found a conflict.')
                    print('   the value {} of key {} '.format(value, key) +
                          'in file {} '.format(trace.filename) +
                          'does not match the value {} '.format(trace_group[0].trace.information[key]) +
                          'in file {}.'.format(trace_group[0].filename))
                    print('File {} will be skipped on the first grouping attempt.'.format(
                          trace.filename))
                    try:
                        trace_group.conflicts.append(trace)
                    except:
                        trace_group.conflicts = [trace]
                    trace_group.remove(trace)

        # now do the 'real' test: try to make a joint trace out
        # of these files. If there are no warnings, then we're in
        # good shape. If there are warnings, try again with
        # the excluded ones?
        group_filenames = []
        for trace in trace_group:
            group_filenames.append(trace.filename)
        joint_trace = pbt.read(group_filenames, skeleton_only=True)
        if joint_trace.last_error != 0:
            print('error! dbpreader.c error # {}'.format(joint_trace.last_error))
            print('retrying with apparently conflicting files...')
            # try again with excluded?
            for trace in trace_group.conflicts:
                group_filenames.append(trace.filename)
            joint_trace = pbt.read(group_filenames, skeleton_only=True)
            if joint_trace.last_error != 0:
                print('error! dbpreader.c error # {}'.format(joint_trace.last_error))
                print('skipping this set of files:')
                for filename in group_filenames:
                    print('    ' + filename)
                continue # skip this set altogether
        filename_groups.append(group_filenames)
    return filename_groups

old_renamed_piece   = '.G-prof-'
def enhance_trace_filenames(filenames, name_infos=default_name_infos,
                              dry_run=False, force_enhance=False):
    """ Renames PBTs to a filename with more useful information than the default.

    Not designed to operate on lists of filenames that do not belong to the same trace,
    but may work anyway, depending on various factors.
    """
    renamed_files = list()
    if ptt.ptt_core in filenames[0]:
        trace = ptt.ParsecTraceTables.from_hdf(filenames[0], skeleton_only=True)
    else:
        trace = pbt.read(filenames, skeleton_only=True)
        if trace.last_error != 0:
            print('{} does not appear to be a reasonable set of filenames.'.format(filenames))
            if not force_enhance:
                return filenames
    # the trace should ALWAYS have a 'start_time' info
    new_chunk = str(trace.start_time) + '-'
    for info in name_infos:
        try:
            value = trace.__getattr__(info)
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
    trace.close()

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
            new_filename = filename.replace(pbt.pbt_core, old_renamed_piece + new_chunk)
        if not dry_run and filename != new_filename:
            if not os.path.exists(dirname + os.sep + new_filename):
                renamed_files.append(dirname + os.sep + new_filename)
                shutil.move(dirname + os.sep + filename,
                            dirname + os.sep + new_filename)
                # also rename already-converted files if they exist
                ptt_name = filename.replace(pbt.pbt_core, ptt.ptt_core)
                new_ptt_name = new_filename.replace(pbt.pbt_core, ptt.ptt_core)
                if (os.path.exists(dirname + os.sep + ptt_name) and
                    not os.path.exists(dirname + os.sep + new_ptt_name)):
                    print('Also renaming corresponding PTT file {}'.format(ptt_name))
                    shutil.move(dirname + os.sep + ptt_name,
                                dirname + os.sep + new_ptt_name)
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
def revert_trace_filenames(filenames, dry_run=False):
    """ Reverts 'enhanced' filenames to their originals, where possible. """
    for filename in filenames:
        full_filename = os.path.abspath(filename)
        dirname = os.path.dirname(full_filename)
        filename = os.path.basename(full_filename)
        match = pbt_filename_regex.match(filename)
        if match:
            original_name = match.group(1).strip('_') + pbt.pbt_core + match.group(3)
        else:
            m = ungrouper.match(filename)
            if m:
                original_name = m.group(1).strip('_') + pbt.pbt_core + m.group(2)
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

def autoload_traces(filenames, convert=True, unlink=False,
                    enhance_filenames=False, skeleton_only=False):
    """ Provides a single interface for all attempts to load PTTs from the filesystem.

    Whether from PTT, PBT, or a combination of the two, you should be able to
    throw a huge list of filenames at this function and receive a whole mess of
    coherent PTT traces back. Give it a whirl, and be sure to report any bugs!
    """
    traces = list()

    groups_or_names = preprocess_traces(filenames, convert=convert, unlink=unlink,
                                        enhance_filenames=enhance_filenames)

    # TODO: after preprocessing, should we check for a converted version anyway?

    if convert: # if we converted in the previous preprocessing step
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        print('loading PTTs...')
        partial_from_hdf = functools.partial(ptt.from_hdf, skeleton_only=skeleton_only)
        traces = pool.map(partial_from_hdf, groups_or_names)
        print('loaded all PTTs.')
        # trace = ptt.ParsecTraceTables.from_hdf(group, skeleton_only=skeleton_only)
    else: # we didn't convert, so these are PBT filename lists
        for group in groups_or_names:
            import pbt2ptt as pbt # don't do this if not necessary
            print('loading PBT group {}'.format(group))
            trace = pbt.read(group, skeleton_only=skeleton_only)
            traces.append(trace)
    return traces

def get_partner_name(filename):
    if ptt.ptt_core in filename:
        return filename.replace(filename.replace(ptt.ptt_core, pbt.pbt_core))
    elif pbt.pbt_core in filename:
        return filename.replace(filename.replace(pbt.pbt_core, ptt.ptt_core))
    else:
        return filename

def compress_many(filenames, clevel=5):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(compress_h5, filenames)

def compress_h5(filename, clevel=5):
    if ptt.ptt_core in filename:
        os.system('h5repack -f GZIP={} {} {}'.format(clevel, filename, filename + '.ctmp'))
        shutil.move(filename + '.ctmp', filename)

def print_help():
    print('')
    print('  -- DESCRIPTION -- ')
    print(' This utility allows the user to automatically group, and convert PaRSEC binary traces.')
    print(' PBTs of different ranks but the same run share certain characteristics, ')
    print(' and this utility will attempt to group the traces by those characteristics.')
    print(' Discovered groupings have their binary traces renamed with shared descriptor strings,')
    print(' which means that even singleton binary traces will benefit from this utility as,')
    print(' by default, the utility embeds the GFLOPS/s into the trace name, if available.')
    print('')
    print('By default, conversion to the Python pandas/HDF5 format (aka PPP) is done also.')
    print('This conversion may take some time, but it is generally recommended for ease of ')
    print('further interaction with the PaRSEC trace. To disable the conversion, ')
    print('pass the --no-convert flag to the utility.')
    print('')
    print(' usage: ptt_utils.py [--dry-run] [--no-convert] [--no-enhance] [--force-enhance]')
    print('                  [--unlink] [--unenhance] [FILENAMES] [extra naming keys]')
    print('')
    print(' Options and arguments:')
    print(' --unenhance   : used to revert the traces to their original names')
    print(' --dry-run     : prints the file renamings without performing them')
    print(' --no-convert  : does not perform PBT->PPP conversion.')
    print('')
    print('  -- EXTRA NAMING KEYS -- ')
    print(' Any other argument not recognized as a filename will be used')
    print(' as an additional piece of the naming scheme, if a corresponding key')
    print(' can be found in the info dictionary of the trace group')
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
            revert_trace_filenames(filenames, dry_run=dry_run)
        else:
            print('You chose not to revert the filenames. Utility now exiting.')
        sys.exit(0)

    if '--compress' in args:
        compress_many(filenames)
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

    preprocess_traces(filenames, dry_run=dry_run, convert=convert,
                      enhance_filenames=enhance_filenames, unlink=unlink,
                      force_enhance=force_enhance,
                      name_infos=name_infos)
