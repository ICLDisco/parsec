include make.inc

TARGETS=grapher dpc tools/buildDAG cholesky/dposv 

LIBRARY_OBJECTS=dplasma.o symbol.o assignment.o expr.o \
	params.o dep.o scheduling.o profiling.o

COMPILER_OBJECTS=lex.yy.o dplasma.tab.o

CHOLESKY_OBJECTS=cholesky/dposv.o

GRAPHER_OBJECTS=grapher.o

BUILDDAG_OBJECTS=tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dpc: $(LIBRARY_OBJECTS) $(COMPILER_OBJECTS) dpc.o
	$(LINKER) -o $@ $^ $(LDFLAGS)

cholesky.c: cholesky/cholesky.jdf dpc
	./dpc < cholesky/cholesky.jdf > cholesky.c

grapher: $(LIBRARY_OBJECTS) grapher.o $(COMPILER_OBJECTS)
	$(CLINKER) -o $@ $^ $(LDFLAGS)

tools/buildDAG:$(COMPILER_OBJECTS) $(LIBRARY_OBJECTS) $(BUILDDAG_OBJECTS)
	$(CLINKER) -o $@ $^ $(LDFLAGS)

cholesky/dposv:$(OBJECTS) $(CHOLESKY_OBJECTS) cholesky.o $(LIBRARY_OBJECTS)
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
	rm -f $(OBJECTS) $(PARSER_OBJECTS) $(CHOLESKY_OBJECTS) $(BUILDDAG_OBJECTS) $(TARGETS) test-expr lex.yy.c y.tab.c y.tab.h
