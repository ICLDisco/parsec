#include <string>
#include <list>
#include <set>
#include <map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctype.h>

using namespace std;

typedef struct{
    string       localVar;
    string       type; 
    string       localAlias;
    list<string> remoteAliases;
    list<string> inEdges;
    list<string> outEdges;
} final_dep_t;

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

typedef struct{
    string name;
    list<string> paramSpace;
    map<string, string> symbolicVars;
    list<string> outDeps;
    list<string> inDeps;
    list<final_dep_t> deps;
} task_t;


class SetIntersector{
    private:
        dep_t f_dep, f_dep2, o_dep;
        list<string> composed_deps;
        set<string> symbolic_vars;
        string var;
        int offset;
        int cmpsd_counter;


        int token_type(string token);
        string cleanAndOffsetVariables(string var_sequence);
        string offset_var( string var );
//        bool isVarGood( string var);
        string rel_to_set(string rel);
        string itos(int num);



    public:
        void setFD1(dep_t fd);
        void setFD2(dep_t fd);
        void setOD(dep_t od);
        void compose_FD2_OD();
        string subtract();
        string inverse(string set);
        string intersect(string set1, string set2);
        string simplify(string set);


        SetIntersector(set<string> sym_vars){
            symbolic_vars = sym_vars;
            cmpsd_counter = 0;
        }
};
