#!/usr/bin/python
__author__="alvaro"
__date__ ="$Sep 2, 2010 10:09:19 AM$"

import sys;
import re;
import shlex;
import os;
from os import path;
from optparse import OptionParser;
from subs import subs;

KEYWORD = '@precisions';
REGEX = '^.*'+KEYWORD+'\s+((\w+,?)+)\s+(\w+)\s+->\s*((\s\w+)+).*$';

def relpath(p):
  p = path.realpath(p);
  return p.replace(path.realpath('.')+'/','');

class Conversion:
  debug = False;
  test = False;
  make = False;
  files_in = [];
  files_out = [];
  def __init__(self, file, match, content):
    self.content = content;
    file = path.realpath(file);
    rel = relpath(file);
    self.file = list(path.split(file));
    self.date = path.getmtime(file);
    if path.samefile(path.join(self.file[0],self.file[1]),sys.argv[0]):
      raise ValueError('Let\'s just forget codegen.py');
    try:
      # normal,all,mixed
      self.types = match[0].split(',');
      # z
      self.precision = match[2].lower();
      # ['c','d','s']
      self.precisions = match[3].lower().split();
    except:
      raise ValueError(path.join(self.file[0],self.file[1])+' : Invalid conversion string');
    self.files_in.append(rel);
      
      
  def run(self):
    if self.convert_names() and not self.test:
      self.convert_data();
      self.export_data();
    
    
  def convert_names(self):
    self.names = [];
    self.dates = [];
    self.converted = [];
    load = False;
    if self.debug: print '|'.join(self.types), self.precision, relpath(path.join(self.file[0],self.file[1]));
    for precision in self.precisions:
      new_file = self.convert(self.file[1], precision);
      if self.debug: print precision,':',
      if new_file <> self.file[1]:
        if self.make:
          print new_file+':',self.file[1];
          print "\t"+path.realpath(sys.argv[0]),"--file",path.realpath(self.file[1]);
        self.names.append(new_file);
        conversion = path.join(self.file[0], new_file);
        self.files_out.append(relpath(conversion));
        if self.debug: print relpath(conversion), ':',
        try:
          date = path.getmtime(conversion);
          diff = self.date - date;
          self.dates.append(diff);
          if self.debug:
            if diff > 0: print 'Old',
            else: print 'Current',
            print diff;
          if diff > 0: load = True;
        except:
          if self.debug: print 'Missing';
          self.dates.append(None);
          load = True;
      else:
        if self.debug: print '<No Change>',':';
        else: print >> sys.stderr, new_file, 'had no change for', precision;
        self.names.append(None);
        self.dates.append(None);
    return load;
    
  def export_data(self):
    for i in range(len(self.names)):
      name = self.names[i];
      data = self.converted[i];
      if data is None or name is None: continue;
      fd = open(path.join(self.file[0],name), 'w');
      fd.write(data);
      fd.close();
    #if self.debug: print 'Exported', ', '.join(self.names);
    
    
  def convert_data(self):
    for i in range(len(self.precisions)):
      precision = self.precisions[i];
      name = self.names[i];
      date = self.dates[i];
      if name is not None and (date is None or date > 0):
        self.converted.append(self.convert(self.content, precision));
      else: self.converted.append(None);
      
      
  def substitute(self, sub_type, data, precision):
    try:
      work = subs[sub_type];
      prec_to = work[0].index(precision);
      prec_from = work[0].index(self.precision);
    except:
      return data;
    for i in range(1,len(work)):
      try:
        search = work[i][prec_from];
        replace = work[i][prec_to];
        if not search: continue;
        replace.replace('\*','*');
        data = re.sub(search, replace, data);
      except:
        print 'Bad replacement pair ',i,'in',sub_type;
        continue;
    return data;
    
    
  def convert(self, data, precision):
    try:
      data = self.substitute('all', data, precision);
    except: pass;
    for sub_type in self.types:
      if sub_type == 'all': continue;
      try:
        data = self.substitute(sub_type, data, precision);
      except Exception, e:
        raise ValueError('I encountered an unrecoverable error while working in subtype:',sub_type+'.');
    data = re.sub('@precisions '+','.join(self.types)+'.*', '@generated '+precision, data); 
    return data;

def grep(string,list):
    expr = re.compile(string)
    return filter(expr.search,list)
    
parser = OptionParser();
parser.add_option('--debug', help='Print debugging messages.', action='store_true', dest='debug', default=False);
parser.add_option('--in-files','--in', help='Print the filenames of files for precision generation.', action='store_true', dest='in_print', default=False);
parser.add_option('--out-files','--out', help='Print the filenames for the precision generated files.', action='store_true', dest='out_print', default=False);
parser.add_option('--out-clean','--clean', help='Remove the files that are the product of generation.', action='store_true', dest='out_clean', default=False);
parser.add_option('--threads', help='Enter the number of threads to use for conversion.', action='store', type='int', dest='threads', default=1);
parser.add_option('--file','-f', help='Specify a file(s) on which to operate.', action='append', dest='files', type='string', default=[]);
parser.add_option('--make', help='Spew a GNU Make friendly file to standard out.', action='store_true', dest='make', default=False);
parser.add_option('--test', help='Don\'t actually do any work.', action='store_true', dest='test', default=False);
(options, args) = parser.parse_args();

rex = re.compile(REGEX);
work = [];

def check_gen(file):
  fd = open(path.realpath(file), 'r');
  lines = fd.readlines();
  fd.close();
  for line in lines:
    m = rex.match(line);
    if m is None: continue;
    work.append((file, m.groups(), ''.join(lines)));

if len(options.files):
  for file in options.files:
    check_gen(file);
else:
  startDir = '.';
  for root, dirs, files in os.walk(startDir, True, None):
    for file in files:
      if file.startswith('.'): continue;
      if not file.endswith('.c') and not file.endswith('.h') and not file.endswith('.f'):
        continue;
      check_gen(path.join(root,file));
    if '.svn' in dirs:
      dirs.remove('.svn');

if options.threads > 1: sem = BoundedSemaphore(value=options.threads);
if options.debug: Conversion.debug = True;
if options.make: Conversion.make = True;
if options.out_print or options.out_clean or options.in_print or options.make or options.test:
  Conversion.test = True;
if options.make:
  print 'generation: gen';

for tuple in work:
  try:
    c = Conversion(tuple[0], tuple[1], tuple[2]);
    c.run();
  except Exception, e:
    print >> sys.stderr, str(e);
    continue;

if options.make:
  print 'cleangen:';
  print '\trm -f '+' '.join(c.files_out);
  print 'gen:',' '+' '.join(c.files_out);
  print '.PHONY: cleangen gen generation';
if options.in_print: print ' '.join(c.files_in);
if options.out_print: print ' '.join(c.files_out);
if options.out_clean:
  for file in c.files_out:
    if not path.exists(file): continue;
    os.remove(file);
