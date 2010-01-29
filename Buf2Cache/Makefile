CC=gcc
CCOPTS=-O3
LDOPTS=-lpthread

all: buf2cache.c driver.c
	$(CC) buf2cache.c -c -Wall
	$(CC) $(CCOPTS) $(LDOPTS) -o driver buf2cache.o driver.c -Wall
