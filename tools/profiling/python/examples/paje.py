#!/usr/bin/env python

from types import *
import collections

MinimalPajeEventDefs = collections.OrderedDict()

MinimalPajeEventDefs['PajeDefineContainerType'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDefineContainerType']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineContainerType']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineContainerType']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeDefineStateType'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDefineStateType']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineStateType']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineStateType']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeDefineEventType'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDefineEventType']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineEventType']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineEventType']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeDefineVariableType'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDefineVariableType']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineVariableType']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineVariableType']['Color'] = 'color'
MinimalPajeEventDefs['PajeDefineVariableType']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeDefineLinkType'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDefineLinkType']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineLinkType']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineLinkType']['StartContainerType'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineLinkType']['EndContainerType'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineLinkType']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeDefineEntityValue'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDefineEntityValue']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineEntityValue']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeDefineEntityValue']['Color'] = 'color'
MinimalPajeEventDefs['PajeDefineEntityValue']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeCreateContainer'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeCreateContainer']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeCreateContainer']['Time'] = 'date'
MinimalPajeEventDefs['PajeCreateContainer']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeCreateContainer']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeCreateContainer']['Alias'] = ['string', 'int']

MinimalPajeEventDefs['PajeDestroyContainer'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeDestroyContainer']['Name'] = ['string', 'int']
MinimalPajeEventDefs['PajeDestroyContainer']['Time'] = 'date'
MinimalPajeEventDefs['PajeDestroyContainer']['Type'] = ['string', 'int']

MinimalPajeEventDefs['PajeSetState'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeSetState']['Time'] = 'date'
MinimalPajeEventDefs['PajeSetState']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeSetState']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeSetState']['Value'] = ['string', 'int']

MinimalPajeEventDefs['PajePushState'] = collections.OrderedDict()
MinimalPajeEventDefs['PajePushState']['Time'] = 'date'
MinimalPajeEventDefs['PajePushState']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajePushState']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajePushState']['Value'] = ['string', 'int']

MinimalPajeEventDefs['PajePopState'] = collections.OrderedDict()
MinimalPajeEventDefs['PajePopState']['Time'] = 'date'
MinimalPajeEventDefs['PajePopState']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajePopState']['Container'] = ['string', 'int']

MinimalPajeEventDefs['PajeResetState'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeResetState']['Time'] = 'date'
MinimalPajeEventDefs['PajeResetState']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeResetState']['Container'] = ['string', 'int']

MinimalPajeEventDefs['PajeNewEvent'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeNewEvent']['Time'] = 'date'
MinimalPajeEventDefs['PajeNewEvent']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeNewEvent']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeNewEvent']['Value'] = ['string', 'int']

MinimalPajeEventDefs['PajeSetVariable'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeSetVariable']['Time'] = 'date'
MinimalPajeEventDefs['PajeSetVariable']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeSetVariable']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeSetVariable']['Value'] = 'double'

MinimalPajeEventDefs['PajeAddVariable'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeAddVariable']['Time'] = 'date'
MinimalPajeEventDefs['PajeAddVariable']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeAddVariable']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeAddVariable']['Value'] = 'double'

MinimalPajeEventDefs['PajeSubVariable'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeSubVariable']['Time'] = 'date'
MinimalPajeEventDefs['PajeSubVariable']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeSubVariable']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeSubVariable']['Value'] = 'double'

MinimalPajeEventDefs['PajeStartLink'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeStartLink']['Time'] = 'date'
MinimalPajeEventDefs['PajeStartLink']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeStartLink']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeStartLink']['StartContainer'] = ['string', 'int']
MinimalPajeEventDefs['PajeStartLink']['Value'] = ['string', 'int']
MinimalPajeEventDefs['PajeStartLink']['Key'] = ['string', 'int']

MinimalPajeEventDefs['PajeEndLink'] = collections.OrderedDict()
MinimalPajeEventDefs['PajeEndLink']['Time'] = 'date'
MinimalPajeEventDefs['PajeEndLink']['Type'] = ['string', 'int']
MinimalPajeEventDefs['PajeEndLink']['Container'] = ['string', 'int']
MinimalPajeEventDefs['PajeEndLink']['EndContainer'] = ['string', 'int']
MinimalPajeEventDefs['PajeEndLink']['Value'] = ['string', 'int']
MinimalPajeEventDefs['PajeEndLink']['Key'] = ['string', 'int']

TraceFile = None
next_alias = 0

class PajeException(Exception):
    def __init__(self, value):
        self.value=value
    def __str__(self):
        return repr(self.value)

class PajeDef:
    """Base Paje Event Definition Class"""
    next_def_key   = 0

    def __init__(self, event_type = None, **kwarg):
        global TraceFile

        if TraceFile is None:
            raise PajeException("Trace not started")

        if event_type not in MinimalPajeEventDefs.keys():
            raise PajeException("'%s' is not a possible PajeEvent Definition" % (event_type))

        self.paje_fields = collections.OrderedDict()

        minimal_keys = MinimalPajeEventDefs[event_type]
        for k, v in kwarg.items():
            if k in minimal_keys.keys():
                if isinstance(minimal_keys[k], list):
                    if v not in minimal_keys[k]:
                        raise PajeException("Field %s of %s must be one of %s" % (k, event_type, minimal_keys[k]))
                else:
                    if v != v:
                        raise PajeException("Field %s of %s must be %s" % (k, event_type, vt))
            else:
                if v not in ['date', 'int', 'double', 'hex', 'string', 'color']:
                    raise PajeException("Field %s of %s is declared of type %s which is not a valid Paje type" % (k, event_type, v))
            self.paje_fields[k] = v
        for k, v in minimal_keys.items():
            if k not in kwarg.keys():
                if isinstance(v, list):
                    self.paje_fields[k] = v[0]
                else:
                    self.paje_fields[k] = v

        self.def_key = PajeDef.next_def_key
        PajeDef.next_def_key = PajeDef.next_def_key+1
        self.event_type = event_type

        TraceFile.write('%%EventDef %s %d\n' % (self.event_type, self.def_key))
        for k, v in self.paje_fields.iteritems():
            TraceFile.write('%% %s %s\n' % (k, v))
        TraceFile.write('%EndEventDef\n')

    def PajeEvent(self, **kwarg):
        global next_alias
        alias = None
        TraceFile.write("%d " % (self.def_key))
        for k, v in self.paje_fields.items():
            if k=="Alias" and k not in kwarg.keys():
                val = "A%d" % next_alias
                alias = val
                next_alias = next_alias+1
            elif k not in kwarg.keys():
                raise PajeException("PajeEventDef %s requires the field %s to be defined" % (self.event_type, k))
            else:
                val = kwarg[k]
                if k == "Alias":
                    alias = val

            if v == 'date':
                assert type(val) is FloatType, "PajeEvent: Key %s is supposed to be a date" % (k)
                TraceFile.write("%g " % (val))
            elif v == 'int':
                assert type(val) is IntType, "PajeEvent: Key %s is supposed to be an integer" % (k)
                TraceFile.write("%d " % (val))
            elif v == 'double':
                assert type(val) is FloatType, "PajeEvent: Key %s is supposed to be a double" % (k)
                TraceFile.write("%g " % (val))
            elif v == 'hex':
                assert type(val) is StringType, "PajeEvent: Key %s is supposed to be an hexadecimal number" % (k)
                TraceFile.write("%s " % (val))
            elif v == 'string':
                assert type(val) is StringType, "PajeEvent: Key %s is supposed to be a string" % (k)
                TraceFile.write("\"%s\" " % (val))
            else:
                assert type(val) is StringType, "PajeEvent: Key %s is supposed to be a color" % (k)
                TraceFile.write("\"%s\" " % (val))
        TraceFile.write("\n")
        return alias

    def __str__(self):
        return "PajeEvent %d: %s (%s)" % (self.def_key, self.event_type, str(self.paje_fields))

def startTrace(filename):
    global TraceFile
    if not (TraceFile is None):
        raise PajeException("%s already open" % (TraceFile.filename()))
    TraceFile = open(filename, 'w')

def endTrace():
    global TraceFile
    if TraceFile is None:
        raise PajeException("Can't end a trace that was not started")
    TraceFile.close()
    TraceFile = None
