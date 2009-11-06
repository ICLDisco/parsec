#
# Required dependencies.
#
PLASMA_DIR    = /Users/bosilca/tools/plasma-installer/build/plasma_2.0.0/
LIBBLAS       = -framework veclib
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



CC = gcc
CLINKER = gcc
LINKER = /usr/local/bin/gfortran
YACC=yacc -d -y --verbose
LEX=flex # -d
#
# Add -DDPLASMA_EXECUTE in order to integrate DPLASMA as a scheduler for PLASMA.
# Add -D_DEBUG to be verbose
#
CFLAGS=-Wall -pedantic -g -I. $(INC) -std=c99 -DADD_
LDFLAGS=-g

TARGETS=cholesky/dposv parser tools/buildDAG

OBJECTS=dplasma.o symbol.o assignment.o expr.o \
	params.o dep.o lex.yy.o dplasma.tab.o

CHOLESKY_OBJECTS=cholesky/cholesky_hook.o \
	cholesky/dposv.o

BUILDDAG_OBJECTS=tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

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

lex.yy.o: lex.yy.c dplasma.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.l
	$(LEX) dplasma.l

clean:
	rm -f $(OBJECTS) $(CHOLESKY_OBJECTS) $(BUILDDAG_OBJECTS) $(TARGETS) test-expr lex.yy.c y.tab.c y.tab.h
