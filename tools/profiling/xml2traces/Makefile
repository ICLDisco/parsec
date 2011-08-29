all:
	gcc -g -Wall $(shell xml2-config --cflags --libs) \
		$(shell pkg-config --cflags --libs gtg) \
		-o xml2traces xml2traces.c
