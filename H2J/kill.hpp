#include <string>
#include <list>
#include <set>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctype.h>

using namespace std;

typedef struct{
    string type;
    string source;
    string sink;
    int srcLine;
    int dstLine;
    string srcArray;
    string dstArray;
    string loop;
    string sets;
} dep_t;


class SetIntersector{
    private:
//        static const int OPERATOR = 0x1;
//        static const int LOGICAL  = 0x2;
//        static const int VARIABLE = 0x3;
        dep_t f_dep, f_dep2, o_dep;
        list<string> composed_deps;
//        list<dep_t> o_deps;
//        set<string> good_vars;
        set<string> symbolic_vars;
        string var;
        int offset;
        int cmpsd_counter;


        int token_type(string token);
        string cleanAndOffsetVariables(string var_sequence);
        string offset_var( string var );
        bool isVarGood( string var);
        string rel_to_set(string rel);
        string itos(int num);



    public:
        void setFD1(dep_t fd);
//        void addOD(dep_t od);
        void setFD2(dep_t fd);
        void setOD(dep_t od);
        void compose_FD2_OD();
        string subtract();
//        string intersect(void);
        SetIntersector(set<string> sym_vars){
            symbolic_vars = sym_vars;
            cmpsd_counter = 0;
        }
};
