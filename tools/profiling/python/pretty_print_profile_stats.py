#!/usr/bin/env python

# the concept behind this module is to allow for pretty column printing
# of disparate types of objects, relying on the basic interface of
# row() and row_header(). If an object supports these two methods
# and returns a string of equal length for both of them, a complex print
# operation can be easily rearranged or modified for a user's printing convenience.

class LinePrinter(list):
    def __init__(self):
        self.sorter = self
        self.name = ''
    def row(self):
        line = ''
        for item in self:
            line += str(item) + ' '
        return line
    def row_header(self):
        hdr = ''
        for item in self:
            hdr += item.row_header() + ' '
        return hdr
    def __repr__(self):
        return self.row()

class ItemPrinter(object):
    def __init__(self, item, header, length = 10):
        self.item = item
        self.length = length
        self.hdr = header
    def row(self):
        return ('{: >' + str(self.length) + '}').format(self.item)
    def row_header(self):
        return ('{: >' + str(self.length) + '}').format(self.hdr)
    def __repr__(self):
        return self.row()

