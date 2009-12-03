
include ../../make.inc

CC = mpicc
LINKER = mpif77
MPILIB =  

TARGETS = dist_exec

OBJECTS= dist_exec.o data_management.o ../../dplasma.o ../../symbol.o ../../assignment.o ../../expr.o \
        ../../params.o ../../dep.o ../../lex.yy.o ../../dplasma.tab.o \
        ../../scheduling.o cholesky_hook.o
 

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dist_exec: $(OBJECTS)
	$(LINKER) -o $@ $^ $(LDFLAGS) $(MPILIB) $(LIB)

%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<


clean:
	rm -f $(OBJECTS) $(TARGETS) *~
