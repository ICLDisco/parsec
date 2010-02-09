CC=gcc-mp-4.3
CCOPTS=-O3
LDOPTS=-lpthread

all: buf2cache.c driver.c
	$(CC) $(CCOPTS) buf2cache.c -c -Wall -I../
	$(CC) $(CCOPTS) $(LDOPTS) -o driver buf2cache.o driver.c -Wall -I../
