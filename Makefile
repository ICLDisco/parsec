CC=gcc
YACC=yacc -d -y --verbose
LEX=flex # -d
CFLAGS=-Wall -pedantic -g -I.
LDFLAGS=

OBJECTS=dplasma.o \
	symbol.o \
	assignment.o \
	expr.o \
	params.o \
	dep.o \
	tools/buildDAG.o

.SUFFIXES:
.SUFFIXES: .c .o .h

all: parse

%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

parse: lex.yy.o dplasma.tab.o $(OBJECTS)
	$(CC) -o parse $^ $(LDFLAGS)

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<

lex.yy.o: lex.yy.c dplasma.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.l
	$(LEX) dplasma.l

clean:
	rm -f *.o test-expr lex.yy.c y.tab.c y.tab.h
