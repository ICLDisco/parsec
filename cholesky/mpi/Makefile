
include ../../make.inc

CC = mpicc
MPILIB = -lmpi -lopen-rte -lopen-pal -L${HOME}/opt/mpi/lib

TARGETS = dist_exec

OBJECTS= dist_exec.o data_management.o

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
