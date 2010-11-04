LDFLAGS=-L$(OMEGA_BASE)/omega_lib/obj
OBJECTS=driver.o utility.o omega_interface.o
INCL=-I$(OMEGA_BASE)/omega_lib/include -I$(OMEGA_BASE)/basic/include
LIB=-lomega


%.tab.h %.tab.c: %.y
	$(YACC) $< -o $(*F).tab.c

parseQuark: lex.yy.o q2j.tab.o $(OBJECTS)
	$(LINKER) -o ../parseQuark $^ $(LDFLAGS) $(LIB)

%.o: %.c $(wildcard *.h)
	$(CC) $(CCFLAGS) -o $@ $(CFLAGS) $(INCL) -c $<

%.o: %.cpp $(wildcard *.h)
	$(CC) $(CCFLAGS) -o $@ $(CFLAGS) $(INCL) -c $<

lex.yy.o: lex.yy.c q2j.tab.h $(wildcard *.h)
	$(CC) $(CCFLAGS) -o lex.yy.o $(CFLAGS) -c lex.yy.c

lex.yy.c: q2j.l
	$(LEX) q2j.l

clean:
	@rm -f *.o lex.yy.c q2j.tab.c q2j.tab.h q2j.output
