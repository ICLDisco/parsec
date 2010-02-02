
.PHONY: clean all
.DEFAULT_GOAL = all

CFLAGS = -D_GNU_SOURCE -Wall -pedantic -I. $(INC) -std=c99 -DREENTRANT
LDFLAGS = -lrt 

include make.inc

TESTING_TARGETS = cholesky/dposv_ll cholesky/dposv_rl QR/dgels
TOOL_TARGETS = grapher dpc tools/buildDAG #cholesky/timeenumerator

include cholesky/mpi/make.inc

TARGETS = dplasma.a $(TOOL_TARGETS) $(TESTING_TARGETS)
all: $(TARGETS)

LIBRARY_OBJECTS=dplasma.o symbol.o assignment.o expr.o \
	params.o dep.o scheduling.o profiling.o remote_dep.o barrier.o
COMPILER_OBJECTS=lex.yy.o dplasma.tab.o precompile.o 
CHOLESKY_OBJECTS=cholesky/dposv.o
QR_OBJECTS = QR/dgels.o QR/QR.o
ENUMERATOR_OBJECTS=cholesky/timeenumerator.o cholesky/cholesky-norun.o
GRAPHER_OBJECTS=grapher.o
BUILDDAG_OBJECTS=tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .a .h


remote_dep.o: $(wildcard remote_dep*.c)
MPI_OBJECTS += remote_dep.o

dplasma.a: dplasma.a($(LIBRARY_OBJECTS))
dplasma-single.a: dplasma-single.a($(patsubst %.o, %-single.o, $(LIBRARY_OBJECTS)))

dpc: $(patsubst %.o, %-single.o, $(COMPILER_OBJECTS) dpc.o) dplasma-single.a
	$(CLINKER) -o $@ $^ $(LDFLAGS) $(LIB)

grapher: $(patsubst %.o, %-single.o, $(GRAPHER_OBJECTS) $(COMPILER_OBJECTS)) dplasma-single.a
	$(CLINKER) -o $@ $^ $(LDFLAGS) $(LIB)

tools/buildDAG: $(patsubst %.o, %-single.o, $(COMPILER_OBJECTS) $(BUILDDAG_OBJECTS)) dplasma-single.a
	$(CLINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/cholesky-norun.o: cholesky/cholesky_ll.c
	$(CC) $(CFLAGS) -UDUSE_MPI -UDPLASMA_EXECUTE -c $^ -o $@

cholesky/dposv_ll: $(patsubst %.o, %-single.o, $(OBJECTS) $(CHOLESKY_OBJECTS) cholesky/cholesky_ll.o) dplasma-single.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/dposv_rl: $(patsubst %.o, %-single.o, $(OBJECTS) $(CHOLESKY_OBJECTS) cholesky/cholesky_rl.o) dplasma-single.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/timeenumerator: $(patsubst %.o, %-single.o, $(OBJECTS) $(ENUMERATOR_OBJECTS)) dplasma-single.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

QR/dgels: $(patsubst %.o, %-single.o, $(OBJECTS) $(QR_OBJECTS)) dplasma-single.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

dposv-graph.svg: dposv.dot xslt-is-hard.sh
	dot -Tsvg dposv.dot > dposv-tmp.svg
	./xslt-is-hard.sh ./dposv-tmp.svg ./dposv-graph.svg
	@rm -f dposv-tmp.svg

DPC = $(realpath dpc)

ifeq "$(strip $(findstring -DUSE_MPI , $(CFLAGS)))" ""
MPICLINKER = $(CLINKER)
MPILINKER = $(LINKER)
else
$(MPI_OBJECTS): %.o: %.c $(wildcard *.h) $(wildcard $(dir $(realpath %.o))/*.h) make.inc
	$(MPICC) -o $@ $(CFLAGS) -c $<
endif

%-single.o: %.c $(wildcard *.h) $(wildcard $(dir $(realpath %.o))/*.h) make.inc
	$(CC) -o $@ $(subst -DUSE_MPI,-UDUSE_MPI, $(CFLAGS)) -c $<

%.o: %.c $(wildcard *.h) $(wildcard $(dir $(realpath %.o))/*.h) make.inc
	$(CC) -o $@ $(CFLAGS) -c $<

%.c: %.jdf dpc
	$(DPC) $< $@

%.tab.h %.tab.c: %.y
	$(YACC) -o $(*F).tab.c $<

lex.yy.o: lex.yy.c dplasma.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.l dplasma.tab.h
	$(LEX) dplasma.l


clean:
	rm -f dplasma.a $(CLEAN_OBJECTS) $(PARSER_OBJECTS) $(CHOLESKY_OBJECTS) $(BUILDDAG_OBJECTS) \
           $(LIBRARY_OBJECTS) $(GRAPHER_OBJECTS) $(TARGETS) $(COMPILER_OBJECTS) \
           $(QR_OBJECTS) cholesky/cholesky-norun.o dpc.o lex.yy.c y.tab.c y.tab.h \
	   cholesky/cholesky_ll.o cholesky/cholesky_rl.o dplasma.tab.h lex.yy.c
