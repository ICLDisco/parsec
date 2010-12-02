#include <stdio.h>

extern int yyparse (void);

int main(int argc, char **argv){
    return yyparse();
}
