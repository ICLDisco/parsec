
include ../../make.inc

CHOLESKY_OBJECTS=../cholesky.o

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

%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<


clean:
	rm -f $(OBJECTS) $(TARGETS) *~
