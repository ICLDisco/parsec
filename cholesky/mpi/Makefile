
include ../../make.inc

CHOLESKY_OBJECTS=cholesky_mpi.o

MPI_DIR = $(HOME)/ompi
CC = $(MPI_DIR)/bin/mpicc
LINKER = $(MPI_DIR)/bin/mpif77
MPILIB = -L $(MPI_DIR)/lib

TARGETS = dist_exec

OBJECTS= dist_exec.o data_management.o ../../dplasma.o ../../symbol.o ../../assignment.o ../../expr.o \
        ../../params.o ../../dep.o ../../scheduling.o


.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dist_exec: $(OBJECTS) $(CHOLESKY_OBJECTS)
	$(LINKER) -o $@ $^ $(LDFLAGS) $(MPILIB) $(LIB)


cholesky_mpi.c: cholesky_mpi.jdf ../../dpc
	../../dpc cholesky_mpi.jdf > cholesky_mpi.c



%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<


clean:
	rm -f $(OBJECTS) $(TARGETS) $(CHOLESKY_OBJECTS)  cholesky_mpi.c *~
