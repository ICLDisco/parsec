include make.inc

TARGETS=grapher dpc tools/buildDAG cholesky/dposv cholesky/timeenumerator

LIBRARY_OBJECTS=dplasma.o symbol.o assignment.o expr.o \
	params.o dep.o scheduling.o profiling.o

COMPILER_OBJECTS=lex.yy.o dplasma.tab.o precompile.o

CHOLESKY_OBJECTS=cholesky/dposv.o cholesky/cholesky.o

ENUMERATOR_OBJECTS=cholesky/timeenumerator.o cholesky/cholesky-norun.o

GRAPHER_OBJECTS=grapher.o

BUILDDAG_OBJECTS=tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dpc: $(LIBRARY_OBJECTS) $(COMPILER_OBJECTS) dpc.o
	$(LINKER) -o $@ $^ $(LDFLAGS)

grapher: $(LIBRARY_OBJECTS) $(GRAPHER_OBJECTS) $(COMPILER_OBJECTS)
	$(CLINKER) -o $@ $^ $(LDFLAGS)

tools/buildDAG:$(COMPILER_OBJECTS) $(LIBRARY_OBJECTS) $(BUILDDAG_OBJECTS)
	$(CLINKER) -o $@ $^ $(LDFLAGS)

cholesky/cholesky.c: cholesky/cholesky.jdf dpc
	./dpc ./cholesky/cholesky.jdf cholesky/cholesky.c

cholesky/cholesky-norun.o: cholesky/cholesky.c
	$(CC) $(CFLAGS) -UDPLASMA_EXECUTE -c cholesky/cholesky.c -o cholesky/cholesky-norun.o

cholesky/dposv:$(OBJECTS) $(CHOLESKY_OBJECTS) $(LIBRARY_OBJECTS)
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

cholesky/timeenumerator:$(OBJECTS) $(ENUMERATOR_OBJECTS) $(LIBRARY_OBJECTS)
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LIB)

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
           dpc.o lex.yy.c dplasma.tab.c dplasma.tab.h
