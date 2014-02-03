import papi_core_utils
known_pcore_labels = papi_core_utils.PAPICoreEventValueLabelGetter()

# add parsing clauses to this function to get infos.
cdef parse_info(builder, event_type, void * cinfo):
    # cdef papi_exec_info_t * cast_exec_info = NULL
    # cdef select_info_t * cast_select_info = NULL
    # cdef papi_core_socket_info_t * cast_core_socket_info = NULL
    # cdef papi_core_select_info_t * cast_core_select_info = NULL
    # cdef papi_core_exec_info_t * cast_core_exec_info = NULL

    event_info = None

    # event_name = builder.event_names[event_type]
    # # this is where users must add code to translate
    # # their own info objects
    # if event_name == 'PINS_EXEC':
    #     cast_exec_info = <papi_exec_info_t *>cinfo
    #     event_info = {
    #         'kernel_type':
    #         cast_exec_info.kernel_type,
    #         'exec_info':
    #         [cast_exec_info.values[x] for x
    #          in range(NUM_EXEC_EVENTS)]}
    # elif event_name == 'PINS_SELECT':
    #     cast_select_info = <select_info_t *>cinfo
    #     event_info = {
    #         'kernel_type':
    #         cast_select_info.kernel_type,
    #         'victim_vp_id':
    #         cast_select_info.victim_vp_id,
    #         'victim_thread_id':
    #         cast_select_info.victim_th_id,
    #         'exec_context':
    #         cast_select_info.exec_context,
    #         'values':
    #         [cast_select_info.values[x] for x
    #          in range(NUM_SELECT_EVENTS)]}
    # elif event_name == 'PINS_ADD':
    #     cast_core_exec_info = <papi_core_exec_info_t *>cinfo
    #     event_info = {
    #         'kernel_type':
    #         cast_core_exec_info.kernel_type,
    #         'PAPI_L1':
    #         cast_core_exec_info.evt_values[0],
    #         'PAPI_L2':
    #         cast_core_exec_info.evt_values[1],
    #         'PAPI_L3':
    #         cast_core_exec_info.evt_values[2],
    #     }
    # elif event_name == 'PINS_L12_EXEC':
    #     cast_core_exec_info = <papi_core_exec_info_t *>cinfo
    #     event_info = {
    #         'kernel_type':
    #         cast_core_exec_info.kernel_type,
    #         'PAPI_L1':
    #         cast_core_exec_info.evt_values[0],
    #         'PAPI_L2':
    #         cast_core_exec_info.evt_values[1],
    #         'PAPI_L3':
    #         cast_core_exec_info.evt_values[2],
    #     }
    # elif event_name == 'PINS_L12_SELECT':
    #     cast_core_select_info = <papi_core_select_info_t *>cinfo
    #     event_info = {
    #         'kernel_type':
    #         cast_core_select_info.kernel_type,
    #         'victim_vp_id':
    #         cast_core_select_info.victim_vp_id,
    #         'victim_thread_id':
    #         cast_core_select_info.victim_th_id,
    #         'starvation':
    #         cast_core_select_info.selection_time,
    #         'exec_context':
    #         cast_core_select_info.exec_context,
    #         'PAPI_L1':
    #         cast_core_select_info.evt_values[0],
    #         'PAPI_L2':
    #         cast_core_select_info.evt_values[1],
    #     }
    # elif event_name == 'PINS_L123':
    #     cast_core_socket_info = <papi_core_socket_info_t *>cinfo
    #     event_info = {
    #         'PAPI_L1':
    #         cast_core_socket_info.evt_values[0],
    #         'PAPI_L2':
    #         cast_core_socket_info.evt_values[1],
    #         'PAPI_L3':
    #         cast_core_socket_info.evt_values[2],
    #     }
    # # events originating from current papi_L123 module
    # elif event_name.startswith('PAPI'):
    #     lbls = known_pcore_labels[event_name]
    #     if ('_EXEC' in event_name or 
    #         '_COMPL' in event_name or 
    #         '_ADD' in event_name or
    #         '_PREP' in event_name):
    #         cast_core_exec_info = <papi_core_exec_info_t *>cinfo
    #         event_info = {
    #             'kernel_type': cast_core_exec_info.kernel_type,
    #         }
    #         for idx, lbl in enumerate(lbls):
    #             event_info.update({lbl: cast_core_exec_info.evt_values[idx]})
    #     elif '_SEL' in event_name:
    #         cast_core_select_info = <papi_core_select_info_t *>cinfo
    #         if cast_core_select_info.selection_time < 0:
    #             # then this event has been marked as invalid
    #             return None
    #         if cast_core_select_info.selection_time > 1000000000:
    #             print(event_name)
    #             print(cast_core_select_info.selection_time)
    #         event_info = {
    #             'kernel_type':
    #             cast_core_select_info.kernel_type,
    #             'victim_vp_id':
    #             cast_core_select_info.victim_vp_id,
    #             'victim_thread_id':
    #             cast_core_select_info.victim_th_id,
    #             'selection_time':
    #             cast_core_select_info.selection_time,
    #             'exec_context':
    #             cast_core_select_info.exec_context,
    #         }
    #         for idx, lbl in enumerate(lbls):
    #             event_info.update({lbl: cast_core_select_info.evt_values[idx]})
    #     elif '_THREAD' in event_name or '_SOCKET' in event_name:
    #         cast_core_socket_info = <papi_core_socket_info_t *>cinfo
    #         event_info = dict()
    #         for idx, lbl in enumerate(lbls):
    #             event_info.update({lbl: cast_core_socket_info.evt_values[idx]})
    # # elif event_name == '<EVENT NAME>':
    # #   event_info = <write some code to make it into a simple Python dict>
    # else:
    #     dont_print = True
    #     if not dont_print:
    #         print('No parser in pbt_info_parser.pxi for event of type \'{}\''.format(event_name))

    return event_info

