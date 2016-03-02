CC=mpicc
CFLAGS=-Wall `pkg-config --cflags parsec` -g3
LDFLAGS=`pkg-config --libs parsec` -pthread -lm -lrt -ldl -lcudart

DAGUEPP=daguepp
DAGUEPPFLAGS=

EXAMPLES= Ex00_StartStop        \
        Ex01_HelloWorld

default: ${EXAMPLES}


Ex00_StartStop: Ex00_StartStop.c
	${CC} -o $@ $< ${CFLAGS} ${LDFLAGS}

%.c: %.jdf
	${DAGUEPP} -i $< -o `basename $@ .c` ${DAGUEPPFLAGS} -- ${CFLAGS}

%.o: %.c
	${CC} -o $@ -c $< ${CFLAGS}

Ex01_HelloWorld: Ex01_HelloWorld.o
	${CC} -o $@ $< ${CFLAGS} ${LDFLAGS}

clean:
	rm -f *.o

cleanall: clean
	rm -f ${EXAMPLES}
