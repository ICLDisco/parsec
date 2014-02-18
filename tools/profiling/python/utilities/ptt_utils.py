#!/usr/bin/env python
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

class GroupedList(list):
    def __init__(self, init_list=None):
        if init_list:
            self.extend(init_list)

def preprocess_traces(filenames, name_infos=default_name_infos, dry_run=False,
                      enhance_filenames=True, force_enhance=False):
    """ Preprocesses PBT files. Returns list of lists of filenames suitable for autoload_traces().

    Preprocesses Parsec Binary Trace files (PBTs) and returns a list of filename lists.
    By default, enhances the filenames to provide more information about each trace file
    via command line examination (e.g., ls).

    It is recommended in most cases that the lists returned by this function
    be submitted to the autoload_traces function for safer and simpler trace loading.
    """

    fname_groups = group_trace_filenames(filenames)

    if enhance_filenames:
        for fn_group in fname_groups[:]:
            new_group = enhance_trace_filenames(fn_group, name_infos=name_infos,
                                                dry_run=dry_run,
                                                force_enhance=force_enhance)
            fname_groups.remove(fn_group)
            fname_groups.append(new_group)

    return fname_groups

def group_trace_filenames(filenames, group_by=group_by_defaults):
    """ groups PBT filenames from different ranks of the same trace.

    This function is primarily necessary for use when dealing with
    PaRSEC Binary Traces (PBTs) that are part of a distributed run.
    Since PaRSEC generates a separate PBT for each rank, it is necessary
    to match up these separate files when attempting to read the full PBT.
    The pbt2ptt module can read a single file at a time
    in order to gather the necessary information to match these files
    automatically, saving the user from unnecessary manual grouping.

    PTTs in the filenames list will be put into a single-element list,
    and lists in the filename list will be passed over if marked as
    previously-grouped, or unpacked and re-grouped if unmarked.
    """

    if hasattr(filenames, '__grouped_marker__'):
        return filenames # do not attempt to regroup previously-grouped filenames

    finished_groups = GroupedList()
    unfinished_groups = dict()

    # this code filters out things that are lists, not filenames
    # and either turns them into filenames or notes that they have
    # already been marked as grouped.
    for filename in filenames[:]:
        if not isinstance(filename, basestring):
            filenames.remove(filename)
            if hasattr(filename, '__grouped_marker__'):
                finished_groups.append(filename)
            else:
                # this is probably an ungrouped list - not sure how this got here.
                filenames.extend(filename)

    for filename in filenames:
        if ptt.is_ptt(filename):
            finished_groups.append([filename])
        else:
            # initially group by start_time
            import pbt2ptt
            infonly_trace = pbt2ptt.read([filename], skeleton_only=True)
            if hasattr(infonly_trace, 'start_time'):
                start_time = infonly_trace.information['start_time']
                if start_time not in unfinished_groups:
                    unfinished_groups[start_time] = list()
                unfinished_groups[start_time].append(TraceAndName(infonly_trace, filename))
            else: # ungroupable - fail fast.
                print('One of the traces does not have a start_time information attribute.')
                print('As a result, these traces cannot be accurately grouped.')
                return [filenames] # we must return a list of lists

    # now that we've done the initial grouping, check for conflicts
    for key, unfinished_group in unfinished_groups.iteritems():
        import pbt2ptt
        # set up grouper
        comparers = dict()
        for key in group_by:
            comparers[key] = None
        for key in comparers.keys():
            if key not in unfinished_group[0].trace.information:
                del comparers[key] # eliminate things they don't have
            else:
                comparers[key] = unfinished_group[0].trace.information[key]
        for trace_and_name in unfinished_group[:1]:
            for key, value in comparers.iteritems():
                if trace_and_name.trace.information[key] != value:
                    print('We seem to have found a conflict.')
                    print('   the value {} of key {} '.format(value, key) +
                          'in file {} '.format(trace_and_name.filename) +
                          'does not match the value {} '.format(
                              unfinished_group[0].trace.information[key]) +
                          'in file {}.'.format(unfinished_group[0].filename))
                    print('File {} will be skipped on the first grouping attempt.'.format(
                          trace_and_name.filename))
                    try:
                        unfinished_group.conflicts.append(trace_and_name)
                    except:
                        unfinished_group.conflicts = [trace_and_name]
                    unfinished_group.remove(trace_and_name)

        # now do the 'real' test: try to make a full trace out
        # of these files. If there are no warnings, then we're in
        # good shape. If there are warnings, try again with
        # the excluded ones?
        only_the_filenames = GroupedList([trace_and_name.filename for trace_and_name
                                          in unfinished_group])
        full_trace = pbt2ptt.read(only_the_filenames, skeleton_only=True)
        if full_trace.last_error != 0:
            print('error! dbpreader.c error # {}'.format(full_trace.last_error))
            print('retrying with apparently conflicting files...')
            # try again with excluded?
            for trace_and_name in unfinished_group.conflicts:
                only_the_filenames.append(trace_and_name.filename)
            full_trace = pbt2ptt.read(only_the_filenames, skeleton_only=True)
            if full_trace.last_error != 0:
                print('error! dbpreader.c error # {}'.format(full_trace.last_error))
                print('skipping this set of files:')
                for filename in only_the_filenames:
                    print('    ' + filename)
                continue # skip this set altogether

        only_the_filenames.__grouped_marker__ = True # mark as correctly grouped
        finished_groups.append(only_the_filenames)

    finished_groups.__grouped_marker__ = True # mark entire list as correctly grouped
    return finished_groups

# NOTE TO SELF:
# FIX ENHANCE TO USE NEW FILE EXTENSION
# CREATE OLD TO NEW CONVERTER
# FIX UNENHANCE TO RECOGNIZE OLD AND NEW - then maybe don't need to write new converter...?

old_renamed_piece   = '.G-prof-'
def enhance_trace_filenames(filenames, name_infos=default_name_infos,
                            dry_run=False, force_enhance=False):
    """ Renames PBTs to a filename with more useful information than the default.

    Not designed to operate on lists of filenames that do not belong to the same trace,
    but may work anyway, depending on various factors.
    """
    renamed_files = list()
    if ptt.is_ptt( filenames[0] ):
        trace = ptt.ParsecTraceTables.from_hdf(filenames[0], skeleton_only=True)
    else:
        import pbt2ptt
        trace = pbt2ptt.read(filenames, skeleton_only=True)
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
        if match: # new style
            new_filename = match.group(1).strip('_') + match.group(2) + '-' + new_chunk + match.group(4)
        else: # old-style (more likely to do over-renaming)
            if old_renamed_piece in filename:
                print('Warning - this file ({}) appears to have been renamed previously.'.format(filename))
                if not force_enhance:
                    print('Ignoring file {} to prevent from polluting the name.'.format(filename))
                    continue
            new_filename = filename.replace(ptt.pbt_core, old_renamed_piece + new_chunk)
        if not dry_run and filename != new_filename:
            if not os.path.exists(dirname + os.sep + new_filename):
                renamed_files.append(dirname + os.sep + new_filename)
                shutil.move(dirname + os.sep + filename,
                            dirname + os.sep + new_filename)
                # also rename already-converted files if they exist
                ptt_name = ptt.ptt_name(filename)
                new_ptt_name = ptt.ptt_name(new_filename)
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
        match = filename_regex.match(filename)
        if match:
            original_name = match.group(1).strip('_') + ptt.pbt_core + match.group(3)
        else:
            m = ungrouper.match(filename)
            if m:
                original_name = m.group(1).strip('_') + ptt.pbt_core + m.group(2)
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
                    enhance_filenames=False, skeleton_only=False,
                    report_progress=True, force_reconvert=False,
                    multiprocess=True):
    """ Provides a single interface for all attempts to load PTTs from the filesystem.

    Whether from PTT, PBT, or a combination of the two, you should be able to
    throw a huge list of filenames at this function and receive a whole mess of
    coherent PTT traces back. Give it a whirl, and be sure to report any bugs!
    """

    pbt_groups = group_trace_filenames(filenames)
    ptts = list()

    # convert or separate into PTTs and PBTs
    if convert: # then turn everything into a PTT
        for fn_group in pbt_groups[:]:
            if len(fn_group) == 1 and ptt.is_ptt(fn_group[0]):
                ptts.append(fn_group[0])
                pbt_groups.remove(fn_group)
            else:
                import pbt2ptt
                # do convert on all -- already-converted filenames will simply be returned
                converted_filename = pbt2ptt.convert(
                    fn_group, unlink=unlink, report_progress=report_progress,
                    force_reconvert=force_reconvert, multiprocess=multiprocess)
                ptts.append(converted_filename)
                pbt_groups.remove(fn_group)
    else: # separate into already-PTTs and PBTs
        for fn_group in pbt_groups[:]:
            ptt_name = ptt.ptt_name(fn_group[0])
            h5_conflicts = find_h5_conflicts(fn_group)
            if ptt.is_ptt(fn_group[0]):
                ptts.append(fn_group)
                pbt_groups.remove(fn_group)
            elif os.path.exists(ptt_name): # passed a PBT name, but previous conversion exists
                ptts.append([ptt_name])
                pbt_groups.remove(fn_group)
            elif len(h5_conflicts) > 0:
                ptts.append([h5_conflicts[0]])
                pbt_groups.remove(fn_group)

    # LOAD PTTs
    if len(ptts) > 0:
        # prepare to multithread the loads
        if multiprocess:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
        else:
            pool = multiprocessing.Pool( 1 )

        if report_progress:
            print('loading PTTs...')
        partial_from_hdf = functools.partial(ptt.from_hdf, skeleton_only=skeleton_only)

        # do load
        traces = pool.map(partial_from_hdf, ptts)

        if report_progress:
            print('loaded all PTTs.')
    else:
        traces = list()

    # LOAD PBTs
    for group in pbt_groups:
        import pbt2ptt # don't do this if not necessary
        if report_progress:
            print('loading PBT group {}'.format(group))
        trace = pbt2ptt.read(
            group, skeleton_only=skeleton_only, multiprocess=multiprocess,
            report_progress=report_progress)
        traces.append(trace)

    return traces

def compress_many(filenames, clevel=5):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(compress_h5, filenames)

def compress_h5(filename, clevel=5):
    if ptt.is_ptt(filename):
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
    print('By default, conversion to the Python pandas/HDF5 format (aka PTT) is done also.')
    print('This conversion may take some time, but it is generally recommended for ease of ')
    print('further interaction with the PaRSEC trace. To disable the conversion, ')
    print('pass the --no-convert flag to the utility.')
    print('')
    print(' usage: ptt_utils.py [--dry-run] [--no-convert] ')
    print('                  [--unlink] [FILENAMES] [extra naming keys]')
    print('')
    print(' Options and arguments:')
    # print(' --unenhance   : used to revert the traces to their original names')
    print(' --dry-run     : prints the file renamings without performing them')
    print(' --no-convert  : does not perform PBT->PTT conversion.')
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
        print('Filename reversion is currently not supported.')
        sys.exit(-1)
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

    processed_filename_groups = group_trace_filenames(filenames)

    # processed_filename_groups = preprocess_traces(filenames, dry_run=dry_run,
    #                                               enhance_filenames=False,
    #                                               force_enhance=force_enhance,
    #                                               name_infos=name_infos)

    if convert:
        for fn_group in processed_filename_groups:
            import pbt2ptt
            fn_group = pbt2ptt.convert(fn_group, unlink=unlink, report_progress=True,
                                       force_reconvert=False, multiprocess=True)

