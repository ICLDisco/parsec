include make.inc

TARGETS=cholesky/dposv parser dpc tools/buildDAG

OBJECTS=dplasma.o symbol.o assignment.o expr.o \
	params.o dep.o lex.yy.o dplasma.tab.o

CHOLESKY_OBJECTS=cholesky/cholesky_hook.o \
	cholesky/dposv.o

PARSER_OBJECTS=parser.o

BUILDDAG_OBJECTS=tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: $(TARGETS)

dpc: $(OBJECTS) dpc.o
	$(LINKER) -o $@ $^ $(LDFLAGS)

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
	rm -f $(OBJECTS) $(PARSER_OBJECTS) $(CHOLESKY_OBJECTS) $(BUILDDAG_OBJECTS) $(TARGETS) test-expr lex.yy.c y.tab.c y.tab.h
