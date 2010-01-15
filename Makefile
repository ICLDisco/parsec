include make.inc

TESTING_TARGETS=cholesky/dposv_ll cholesky/dposv_rl QR/dgels
TARGETS=grapher dpc tools/buildDAG cholesky/timeenumerator $(TESTING_TARGETS)

LIBRARY_OBJECTS=dplasma.o symbol.o assignment.o expr.o \
	params.o dep.o scheduling.o profiling.o remote_dep.o barrier.o

COMPILER_OBJECTS=lex.yy.o dplasma.tab.o precompile.o 

CHOLESKY_OBJECTS=cholesky/dposv.o
QR_OBJECTS = QR/dgels.o QR/QR.o

ENUMERATOR_OBJECTS=cholesky/timeenumerator.o cholesky/cholesky-norun.o

GRAPHER_OBJECTS=grapher.o

BUILDDAG_OBJECTS=tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dplasma.a: $(LIBRARY_OBJECTS)
	$(AR) rcs $@ $(LIBRARY_OBJECTS)

dpc: $(COMPILER_OBJECTS) dpc.o dplasma.a
	$(CLINKER) -o $@ $^ $(LDFLAGS) $(LIB)

grapher: $(GRAPHER_OBJECTS) $(COMPILER_OBJECTS) dplasma.a
	$(CLINKER) -o $@ $^ $(LDFLAGS) $(LIB)

tools/buildDAG:$(COMPILER_OBJECTS) dplasma.a $(BUILDDAG_OBJECTS)
	$(CLINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/cholesky_ll.c: cholesky/cholesky_ll.jdf dpc
	./dpc ./cholesky/cholesky_ll.jdf $@

cholesky/cholesky_rl.c: cholesky/cholesky_rl.jdf dpc
	./dpc ./cholesky/cholesky_rl.jdf $@

cholesky/cholesky-norun.o: cholesky/cholesky_ll.c
	$(CC) $(CFLAGS) -UDPLASMA_EXECUTE -c cholesky/cholesky_ll.c -o cholesky/cholesky-norun.o

cholesky/dposv_ll:$(OBJECTS) $(CHOLESKY_OBJECTS) $(LIBRARY_OBJECTS) cholesky/cholesky_ll.o dplasma.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/dposv_rl:$(OBJECTS) $(CHOLESKY_OBJECTS) $(LIBRARY_OBJECTS) cholesky/cholesky_rl.o dplasma.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/timeenumerator:$(OBJECTS) $(ENUMERATOR_OBJECTS) dplasma.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

QR/QR.c: QR/QR.jdf dpc
	./dpc ./QR/QR.jdf QR/QR.c

QR/dgels: $(OBJECTS) $(QR_OBJECTS) dplasma.a
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

remote_dep.c: $(wildcard remote_dep_*.c)

%.tab.h %.tab.c: %.y
	$(YACC) -o $(*F).tab.c $<

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<

lex.yy.o: lex.yy.c dplasma.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.l
	$(LEX) dplasma.l

clean:
	rm -f $(PARSER_OBJECTS) $(CHOLESKY_OBJECTS) $(BUILDDAG_OBJECTS) \
           $(LIBRARY_OBJECTS) $(GRAPHER_OBJECTS) $(TARGETS) $(COMPILER_OBJECTS) \
           $(QR_OBJECTS) cholesky/cholesky-norun.o dpc.o lex.yy.c y.tab.c y.tab.h \
	   cholesky/cholesky_ll.o cholesky/cholesky_rl.o
