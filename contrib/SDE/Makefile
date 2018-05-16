CC=gcc
CFLAGS=-g -fPIC -Wall -I$(PAPIDIR)/include
LDFLAGS=-g
LIBS=-lpthread -L$(PAPIDIR)/lib -lpapi

all: sde_quick_test.so

sde_quick_test.o: sde_quick_test.c
	$(CC) -o sde_quick_test.o $(CFLAGS) -c sde_quick_test.c

sde_quick_test.so: sde_quick_test.o
	$(CC) $(LDFLAGS) -shared -o sde_quick_test.so sde_quick_test.o $(LIBS)

clean:
	rm -f sde_quick_test.o sde_quick_test.so

