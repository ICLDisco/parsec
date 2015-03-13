from collections import namedtuple
import struct

cdef class ExtendedEvent:
    cdef object ev_struct
    cdef object aev
    cdef char* fmt
    cdef int event_len

    def __init__(self, event_name, event_conv, event_len):
        fmt = '@'
        tup = []
        for ev in str.split(event_conv, ':'):
            if 0 == len(ev):
                continue
            ev_list = str.split(ev, '{', 2)
            if len(ev_list) > 1:
                [ev_name, ev_type] = ev_list[:2]
                ev_type = ev_type.replace('}', '')
            else:
                ev_name = ev_list[0] if len(ev_list) == 1 else ''
                ev_type = ''
            if 0 == len(ev_name):
                continue
            tup.append(ev_name)
            if ev_type == 'int32_t' or ev_type == 'int':
                fmt += 'i'
            elif ev_type == 'int64_t':
                fmt += 'l'
            elif ev_type == 'double':
                fmt += 'd'
            elif ev_type == 'float':
                fmt += 'f'
        self.aev = namedtuple(event_name, tup)
        #print('event[{0} = {1} fmt \'{2}\''.format(event_name, tup, fmt))
        self.ev_struct = struct.Struct(fmt)
        if event_len != len(self):
            print('Expected length differs from provided length for {0} extended event ({1} != {2})'.format(event_name, len(self), event_len))
            event_len = event_len if event_len < len(self) else len(self)
        self.event_len = event_len
    def __len__(self):
        return self.ev_struct.size
    def unpack(self, pybs):
        return self.aev._make(self.ev_struct.unpack(pybs))

# add parsing clauses to this function to get infos.
cdef parse_info(builder, event_type, char * cinfo):
    cdef papi_exec_info_t * cast_exec_info = NULL
    cdef select_info_t * cast_select_info = NULL
    cdef papi_core_socket_info_t * cast_core_socket_info = NULL
    cdef papi_core_select_info_t * cast_core_select_info = NULL
    cdef papi_core_exec_info_t * cast_core_exec_info = NULL

    cdef bytes pybs;

    if None == builder.event_convertors[event_type]:
       return None
    event_info = None

    try:
        pybs = cinfo[:len(builder.event_convertors[event_type])]
        return builder.event_convertors[event_type].unpack(pybs)._asdict()
    except Exception as e:
        print(e)
        return None
