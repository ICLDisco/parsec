CC=/home/lemarini/ompi/bin/mpicc
LINKER=/home/lemarini/ompi/bin/mpif90

#
# Required dependencies.
#
PLASMA_DIR    = /home/lemarini/plasma/build/plasma_2.0.0/
LIBBLAS       = -L/home/lemarini/plasma/lib -lrefblas
# Include directory
INC        = -I$(PLASMA_DIR)/include -I$(PLASMA_DIR)/src

# Location of the libraries.
LIBDIR     = -L$(PLASMA_DIR)/lib/

# Location and name of the PLASMA library.
LIBCBLAS      = $(PLASMA_DIR)/lib/libcblas.a
LIBCORELAPACK = $(PLASMA_DIR)/lib/libcorelapack.a
LIBCOREBLAS   = $(PLASMA_DIR)/lib/libcoreblas.a
LIBPLASMA     = $(PLASMA_DIR)/lib/libplasma.a

#  All libraries required by the tester.
LIB        = $(LIBDIR) -lplasma -lcoreblas -lcorelapack -lcblas $(LIBBLAS) -lpthread -lm


CFLAGS=-Wall -pedantic -g -I. -I.. -I$(PLASMA_DIR)/include -I$(PLASMA_DIR)/src -std=c99 -DADD_
LDFLAGS=-g $(LIB)

TARGETS=dist_exec

OBJECTS= dist_exec.o data_management.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dist_exec: $(OBJECTS)
	$(LINKER) -o $@ $^ $(LDFLAGS)

parser: $(OBJECTS) parser.o
	$(CLINKER) -o $@ $^ $(LDFLAGS)

tools/buildDAG:$(OBJECTS) $(BUILDDAG_OBJECTS)
	$(CLINKER) -o $@ $^ $(LDFLAGS)

cholesky/dposv:$(OBJECTS) $(CHOLESKY_OBJECTS)
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<


clean:
	rm -f $(OBJECTS) $(TARGETS) *~
