#
# Required dependencies.
#
PLASMA_DIR    = /Users/bosilca/tools/plasma-installer/build/plasma_2.0.0
LIBBLAS       = -framework veclib
# Include directory
INC        = -I$(PLASMA_DIR)/include -I$(PLASMA_DIR)/src

# Location of the libraries.
LIBDIR     = -L$(PLASMA_DIR)/lib

# Location and name of the PLASMA library.
LIBCBLAS      = $(PLASMA_DIR)/lib/libcblas.a
LIBCORELAPACK = $(PLASMA_DIR)/lib/libcorelapack.a
LIBCOREBLAS   = $(PLASMA_DIR)/lib/libcoreblas.a
LIBPLASMA     = $(PLASMA_DIR)/lib/libplasma.a

#  All libraries required by the tester.
LIB        = $(LIBDIR) -lplasma -lcoreblas -lcorelapack -lcblas $(LIBBLAS) -lpthread -lm



CC=/usr/local/bin/gcc
LINKER = /usr/local/bin/gfortran
YACC=yacc -d -y --verbose
LEX=flex # -d
#
# Add -DDPLASMA_EXECUTE in order to integrate DPLASMA as a scheduler for PLASMA.
#
CFLAGS=-Wall -pedantic -O3 -I. $(INC) -std=c99 -DADD_ -DDPLASMA_EXECUTE
LDFLAGS=

OBJECTS=dplasma.o \
	symbol.o \
	assignment.o \
	expr.o \
	params.o \
	dep.o \
	tools/buildDAG.o \
	cholesky_hook.o \
	example_dposv.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: parse

%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

parse: lex.yy.o dplasma.tab.o $(OBJECTS)
	$(LINKER) -o parse $^ $(LDFLAGS) $(LIB)

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<

lex.yy.o: lex.yy.c dplasma.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.l
	$(LEX) dplasma.l

clean:
	rm -f *.o test-expr lex.yy.c y.tab.c y.tab.h
