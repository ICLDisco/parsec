CC=gcc
YACC=yacc -d -y --verbose
LEX=flex # -d
CFLAGS=-Wall -pedantic -ansi -g
LDFLAGS=

OBJECTS=dplasma.o \
	symbol.o \
	expr.o \
	params.o \
	dep.o

all: parse

parse: lex.yy.o y.tab.o $(OBJECTS)
	$(CC) -o parse lex.yy.o y.tab.o $(OBJECTS) $(LDFLAGS)

%.o: %.c $(wildcard *.h)
	$(CC) -o $@ $(CFLAGS) -c $<

y.tab.h y.tab.c: dplasma.y
	$(YACC) dplasma.y

lex.yy.o: lex.yy.c y.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.l
	$(LEX) dplasma.l

clean:
	rm -f *.o test-expr lex.yy.c y.tab.c y.tab.h
