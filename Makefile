CC=gcc
YACC=yacc -d -y --verbose
LEX=flex -d
CFLAGS=-Wall -pedantic -ansi -g
LDFLAGS=

all: test-expr

test-expr: test-expr.o expr.o lex.yy.o y.tab.o
	$(CC) -o test-expr test-expr.o expr.o $(LDFLAGS)

test-expr.o: test-expr.c $(wildcard *.h)
	$(CC) -o test-expr.o $(CFLAGS) -c test-expr.c

expr.o: expr.c $(wildcard *.h)
	$(CC) -o expr.o $(CFLAGS) -c expr.c

y.tab.o: y.tab.c $(wildcard *.h)
	$(CC) -o y.tab.o $(CFLAGS) -c y.tab.c

y.tab.h y.tab.c: dplasma.yy
	$(YACC) dplasma.yy

lex.yy.o: lex.yy.c y.tab.h $(wildcard *.h)
	$(CC) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: dplasma.ll
	$(LEX) dplasma.ll

clean:
	rm -f *.o test-expr lex.yy.c y.tab.c y.tab.h
