CC=gcc
CFLAGS=-Wall -pedantic -Werror -ansi -g
LDFLAGS=

all: test-expr

test-expr: test-expr.o expr.o
	$(CC) -o test-expr test-expr.o expr.o $(LDFLAGS)

test-expr.o: test-expr.c $(wildcard *.h)
	$(CC) -o test-expr.o $(CFLAGS) -c test-expr.c

expr.o: expr.c $(wildcard *.h)
	$(CC) -o expr.o $(CFLAGS) -c expr.c

clean:
	rm -f *.o test-expr
