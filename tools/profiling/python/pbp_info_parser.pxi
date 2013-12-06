# add parsing clauses to this function to get infos.
cdef parse_info(builder, event_type, unique_id, void * cinfo):
    cdef papi_exec_info_t * cast_exec_info = NULL
    cdef select_info_t * cast_select_info = NULL
    cdef papi_core_socket_info_t * cast_core_socket_info = NULL
    cdef papi_core_select_info_t * cast_core_select_info = NULL
    cdef papi_core_exec_info_t * cast_core_exec_info = NULL

    event_info = None

    event_name = builder.event_names[event_type]
    # this is where users must add code to translate
    # their own info objects
    if event_name == 'PINS_EXEC':
        cast_exec_info = <papi_exec_info_t *>cinfo
        event_info = {
            'kernel_type':
            cast_exec_info.kernel_type,
            'exec_info':
            [cast_exec_info.values[x] for x
             in range(NUM_EXEC_EVENTS)]}
    elif event_name == 'PINS_SELECT':
        cast_select_info = <select_info_t *>cinfo
        event_info = {
            'kernel_type':
            cast_select_info.kernel_type,
            'victim_vp_id':
            cast_select_info.victim_vp_id,
            'victim_thread_id':
            cast_select_info.victim_th_id,
            'exec_context':
            cast_select_info.exec_context,
            'values':
            [cast_select_info.values[x] for x
             in range(NUM_SELECT_EVENTS)]}
    elif event_name == 'PINS_ADD':
        cast_core_exec_info = <papi_core_exec_info_t *>cinfo
        event_info = {
            'unique_id':
            unique_id,
            'kernel_type':
            cast_core_exec_info.kernel_type,
            'PAPI_L1':
            cast_core_exec_info.evt_values[0],
            'PAPI_L2':
            cast_core_exec_info.evt_values[1],
            'PAPI_L3':
            cast_core_exec_info.evt_values[2],
        }
    elif event_name == 'PINS_L12_EXEC':
        cast_core_exec_info = <papi_core_exec_info_t *>cinfo
        event_info = {
            'unique_id':
            unique_id,
            'kernel_type':
            cast_core_exec_info.kernel_type,
            'PAPI_L1':
            cast_core_exec_info.evt_values[0],
            'PAPI_L2':
            cast_core_exec_info.evt_values[1],
            'PAPI_L3':
            cast_core_exec_info.evt_values[2],
        }
    elif event_name == 'PINS_L12_SELECT':
        cast_core_select_info = <papi_core_select_info_t *>cinfo
        event_info = {
            'unique_id':
            unique_id,
            'kernel_type':
            cast_core_select_info.kernel_type,
            'victim_vp_id':
            cast_core_select_info.victim_vp_id,
            'victim_thread_id':
            cast_core_select_info.victim_th_id,
            'starvation':
            cast_core_select_info.selection_time,
            'exec_context':
            cast_core_select_info.exec_context,
            'PAPI_L1':
            cast_core_select_info.evt_values[0],
            'PAPI_L2':
            cast_core_select_info.evt_values[1],
        }
    elif event_name == 'PINS_L123':
        cast_core_socket_info = <papi_core_socket_info_t *>cinfo
        event_info = {
            'unique_id':
            unique_id,
            'PAPI_L1':
            cast_core_socket_info.evt_values[0],
            'PAPI_L2':
            cast_core_socket_info.evt_values[1],
            'PAPI_L3':
            cast_core_socket_info.evt_values[2],
        }
    # events originating from current papi_L123 module
    elif event_name.startswith('PAPI'):
        lbls = None
        try:
            lbls = papi_core_evt_value_lbls[event_name]
        except:
            # remove first part of label
            if '_CORE_' in event_name:
                lbls = papi_core_evt_value_lbls['_'.join(event_name.split('_')[3:])]
            elif '_SOCKET_' in event_name:
                lbls = papi_core_evt_value_lbls['_'.join(event_name.split('_')[2:])]
            
        if ('_EXEC' in event_name or 
            '_COMPL' in event_name or 
            '_ADD' in event_name or
            '_PREP' in event_name):
            cast_core_exec_info = <papi_core_exec_info_t *>cinfo
            event_info = {
                'unique_id': unique_id,
                'kernel_type': cast_core_exec_info.kernel_type,
            }
            for idx, lbl in enumerate(lbls):
                event_info.update({lbl: cast_core_exec_info.evt_values[idx]})
        elif '_SEL' in event_name:
            cast_core_select_info = <papi_core_select_info_t *>cinfo
            if cast_core_select_info.selection_time < 0:
                # then this event has been marked as invalid
                return None
            if cast_core_select_info.selection_time > 1000000000:
                print(event_name)
                print(cast_core_select_info.selection_time)
            event_info = {
                'unique_id':
                unique_id,
                'kernel_type':
                cast_core_select_info.kernel_type,
                'victim_vp_id':
                cast_core_select_info.victim_vp_id,
                'victim_thread_id':
                cast_core_select_info.victim_th_id,
                'selection_time':
                cast_core_select_info.selection_time,
                'exec_context':
                cast_core_select_info.exec_context,
            }
            for idx, lbl in enumerate(lbls):
                event_info.update({lbl: cast_core_select_info.evt_values[idx]})
        elif '_THREAD' in event_name or '_SOCKET' in event_name:
            cast_core_socket_info = <papi_core_socket_info_t *>cinfo
            event_info = {
                'unique_id':
                unique_id,
            }
            for idx, lbl in enumerate(lbls):
                event_info.update({lbl: cast_core_socket_info.evt_values[idx]})
    # elif event_name == '<EVENT NAME>':
    #   event_info = <write some code to make it into a simple Python dict>
    else:
        dont_print = True
        if not dont_print:
            print('No parser in pbp_info_parser.pxi for event of type \'{}\''.format(event_name))

    return event_info

papi_core_evt_value_lbls = {
    'PAPI_L12_ADD'              : ['PAPI_L1', 'PAPI_L2', 'PAPI_L3'],
    'PAPI_L12_COMPLETE_EXEC'    : ['PAPI_L1', 'PAPI_L2', 'PAPI_L3'],
    'PAPI_L12_EXEC'             : ['PAPI_L1', 'PAPI_L2', 'PAPI_L3'],
    'PAPI_L12_SELECT'           : ['PAPI_L1', 'PAPI_L2'],
    'PAPI_L123_THREAD'          : ['PAPI_L1', 'PAPI_L2', 'PAPI_L3'],
    'PAPI_SOCKET'               : ['PAPI_L1', 'PAPI_L2', 'PAPI_L3'],
    'PAPI_CORE_EXEC'            : ['PAPI_L1', 'PAPI_L2'],
    'PAPI_CORE_SEL'             : ['PAPI_L1', 'PAPI_L2'],
    'PAPI_CORE_PREP'            : ['PAPI_L1', 'PAPI_L2'],
    'PAPI_CORE_COMPL'           : ['PAPI_L1', 'PAPI_L2'],
    'PL3'        : ['PAPI_L1', 'PAPI_L2', 'PAPI_L3'],
    'TLB_EV'     : ['TLB_MISS', 'L3_EVICT', 'L3_MISS'],
    '23T'           : ['L2_DMISS', 'L3_DMISS', 'TLB_MISS'],
    'RE_IO_L3AC' : ['DATA_CACHE_REFILLS-SYS', 'CPU_TO_MEM', 'L3_MISS-AC'],
    'PREF_L3MOD' : ['NODE_PREFETCH', 'NODE_PREFETCH_MISS', 'L3_WMISS'],
    'PREFMISS'   : ['NODE_PREFETCH_MISS'],
    'PREFETCHES' : ['PREF_INSTR_DISP-L', 'INEFF_SW_PREF', 'DATA_PREF-ATT'], 
    'MEMCONT_CBLOCK': ['MEM_CONT_REQ-R', 'MEM_CONT_REQ-PREF', 'CACHE_BLOCK-A'],
    'L3M_MRQNC_DCRL2': ['L3_WMISS', 'MEM_REQ-NONC', 'DATA_CACHE_REFILLS-L2D'],
    'L3_LLC-LM_MIS-AC': ['L3_MISSES:ANY', 'LLC-LOAD-MISSES', 'MISALIGNED_ACCESSES'],
    'MWAIT_STL-SEGL_PREF-HIT-L2': ['MAB_WAIT_CYCLES', 'DISP_STALL_SEG_LOAD',
                                   'INEFF_SW_PREF-HIT_L2'],
    'INPREF-HIT-L1_STL-RSFULL_RREQ-L3': ['INEFF_SW_PREF-HIT_L1',
                                         'DSTALL_RES_ST_FULL',
                                         'READ_REQ_L3'],
    'L2-HW-PREF_MAB-REQ_DSTL-LS-F': ['L2_MISS-HWPREF', 'MAB_REQ', 'DSTL_LS_FULL'],
    'DS-LS-FULL_L23': ['DSTL_LS_FULL', 'L2_DMISS', 'L3_MISS'],
    'DS_L2_ISP' : ['DSTL_LOAD-STORE_FULL', 'L2_DMISS', 'INEFF_SW_PREF-HIT_L1'],
}
