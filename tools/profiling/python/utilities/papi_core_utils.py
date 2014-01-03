known_papi_core_event_value_labels = {
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

class PAPICoreEventValueLabelGetter(object):
    def __init__(self):
        self.exists = True
    def __getattr__(self, event_name):
        return self.__getitem__(event_name)
    def __getitem__(self, event_name):
        lbls = None
        try:
            lbls = known_papi_core_event_value_labels[event_name]
        except:
            # remove first part of label
            if '_CORE_' in event_name:
                lbls = known_papi_core_event_value_labels['_'.join(event_name.split('_')[3:])]
            elif '_SOCKET_' in event_name:
                lbls = known_papi_core_event_value_labels['_'.join(event_name.split('_')[2:])]
        return lbls
