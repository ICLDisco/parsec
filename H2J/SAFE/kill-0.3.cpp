#include <string>
#include <list>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

typedef struct{
    string type;
    string source;
    string sink;
    int srcLine;
    int dstLine;
    string srcArray;
    string dstArray;
    string sets;
} dep_t;


list<dep_t> flow_deps, output_deps, merged_deps;

// Forward Function Declarations
int parse_petit_output(std::ifstream &ifs);
int readNextSource(string line, string &source, std::ifstream &ifs);
int readNextDestination(string line, string source, std::ifstream &ifs);
bool isEOR(string line);
void store_dep(list<dep_t> &depList, dep_t dep);
string skipToNext(std::ifstream &ifs);
void dumpList(list<dep_t> depList);
void mergeLists(void);
void dumpDep(dep_t dep);

// Code
int main(int argc, char **argv){
    char *fName;

    if( argc < 2 ){
        cerr << "Usage: "<< argv[0] << " Pettit_Output_File" << endl;
        exit(-1);
    }

    fName = argv[1];
    ifstream ifs( fName );
    if( !ifs ){
        cerr << "File \""<< fName <<"\" does not exist" << endl;
        exit(-1);
    }

    parse_petit_output(ifs);
    return 0;
}

int parse_petit_output(ifstream &ifs){
    int i;
    stringstream ss;
    string line, source;

    flow_deps.clear();
    output_deps.clear();
    i=0;
    while( getline(ifs,line) ){

        if( readNextSource(line, source, ifs) ){ return -1; }
        line = skipToNext(ifs);
        while( 1 ){
            if( readNextDestination(line, source, ifs) ){ return -1; }
            line = skipToNext(ifs);
            if( line.empty() || line.find("#####") != string::npos ){ break; }
        }

//        killDeadFlowDeps(flow_deps, output_deps);

    }

    mergeLists();

/*
    cout << "----- Flow -----" << endl;
    dumpList(flow_deps);
    cout << "----- Out -----" << endl;
    dumpList(output_deps);
    cout << "" << endl;
*/

    return 0;
}

int readNextSource(string line, string &source, ifstream &ifs){
    unsigned int pos;

    pos = line.find("### SOURCE:");
    while( pos == string::npos ){
        if( !getline(ifs,line) ) return 0;
        pos = line.find("### SOURCE:");
    }

    pos = line.find(":");
    // pos+2 to skip the ":" and the empty space following it
    line = line.substr(pos+2);
    if( line.empty() ){
        cerr << "Empty SOURCE" << endl;
        return -1;
    }else{
        source = line;
//        cout << "New Source Found: "<< source << endl;
    }

    return 0;
}


// Sample format of flow/output dependencies
//
// --> DTSTRF
// flow    10: A(k,k)          -->  16: A(k,k)          (0)             [ M]
// {[k] -> [k,m] : 0 <= k < m < BB}
// exact dd: {[0]}
//
// --> DTSTRF
// output  10: A(k,k)          -->  17: A(k,k)          (0)             [ M]
// {[k] -> [k,m] : 0 <= k < m < BB}
// exact dd: {[0]}
//
int readNextDestination(string line, string source, ifstream &ifs){
    stringstream ss;
    string sink, type, srcLine, junk, dstLine;
//    string sink, type, srcLine, srcArray, junk, dstLine, dstArray;
    dep_t dep;

    // read the sink of this dependency
    unsigned int pos = line.find("===>");
    if( pos != string::npos ){
        pos = line.find(">");
        // pos+2 to skip the ">" and the empty space following it
        line = line.substr(pos+2);
        if( line.empty() ){
            cerr << "Empty sink" << endl;
            return -1;
        }else{
            sink = line;
//            cout << "New Sink Found: "<< source << " --> " << sink << endl;
        }
    }

    // read the details of this dependency
    if( !getline(ifs,line) || isEOR(line) ){ return 0; }

    dep.source = source;
    dep.sink = sink;
    ss << line;
    ss >> type >> srcLine >> dep.srcArray >> junk >> dstLine >> dep.dstArray;

    dep.type = type;

    // Remove the ":" from the source and destination line.
    pos = srcLine.find(":");
    if( pos != string::npos ){
        dep.srcLine = atoi( srcLine.substr(0,pos).c_str() );
    }
    pos = dstLine.find(":");
    if( pos != string::npos ){
        dep.dstLine = atoi( dstLine.substr(0,pos).c_str() );
    }

    // read the sets of values for the index variables
    if( !getline(ifs,line) || isEOR(line) ){ return 0; }
    dep.sets = line;

    // Store this dependency into the map.
    if( !type.compare("flow") ){
        store_dep(flow_deps, dep);
    }else if( !type.compare("output") ){
        store_dep(output_deps, dep);
    }else{
        cerr << "Unknown type of dependency. ";
        cerr << "Only \"flow\" and \"output\" types are accepted" << endl;
        return -1;
    }

    return 0;
}


void store_dep(list<dep_t> &depList, dep_t dep){
    depList.push_back(dep);
    return;
}

// is this line an End Of Record
bool isEOR(string line){
    unsigned int pos = line.find("#########");
    if( pos != string::npos || line.empty() ){
        return true;
    }
    return false;
}

string skipToNext(ifstream &ifs){
    string line;
    while( 1 ){
        if( !getline(ifs,line) ) return string("");
        unsigned int pos = line.find("===>");
        if( pos != string::npos ){ break; }
        pos = line.find("#######");
        if( pos != string::npos ){ break; }
    }
    return line;
}

void dumpDep(dep_t dep){
        if( !dep.type.compare("flow") )
            cout << "f ";
        else
            cout << "d ";
        cout << dep.source << " ";
        cout << dep.sink << " ";
        cout << dep.srcLine << " ";
        cout << dep.dstLine << " ";
        cout << dep.srcArray << " ";
        cout << dep.dstArray << " ";
        cout << dep.sets << endl;
}

void dumpList(list<dep_t> depList){
    list<dep_t>::iterator itr;

    for(itr=depList.begin(); itr != depList.end(); ++itr) {
        dep_t dep = *itr;
        dumpDep(dep);
/*
//        cout << dep.type << " ";
        cout << dep.source << " ";
        cout << dep.sink << " ";
        cout << dep.srcLine << " ";
        cout << dep.dstLine << " ";
        cout << dep.srcArray << " ";
        cout << dep.dstArray << " ";
        cout << dep.sets << endl;
*/
    }
}

void mergeLists(void){
    bool found;
    list<dep_t>::iterator fd_itr; // flow dep iterator
    list<dep_t>::iterator od_itr; // out dep iterator
    list<dep_t>::iterator fd2_itr; // second flow dep iterator

    for(fd_itr=flow_deps.begin(); fd_itr != flow_deps.end(); ++fd_itr) {
        found = false;
        dep_t f_dep = *fd_itr;
        int fd_srcLine = f_dep.srcLine;
        int fd_dstLine = f_dep.dstLine;
        for(od_itr=output_deps.begin(); od_itr != output_deps.end(); ++od_itr) {
            dep_t o_dep = *od_itr;
            int od_srcLine = o_dep.srcLine;
            int od_dstLine = o_dep.dstLine;
            if( fd_srcLine == od_srcLine ){
                for(fd2_itr=flow_deps.begin(); fd2_itr != flow_deps.end(); ++fd2_itr) {
                    dep_t f2_dep = *fd2_itr;
                    int fd2_srcLine = f2_dep.srcLine;
                    int fd2_dstLine = f2_dep.dstLine;
                    if( od_dstLine == fd2_srcLine && fd_dstLine == fd2_dstLine ){
                        dumpDep(f_dep);
                        dumpDep(o_dep);
                        dumpDep(f2_dep);
                        cout << "" << endl;
                        found = true;
                        break;
                    }
                }
            }
        }
        dumpDep(f_dep);
        cout << "" << endl;
    }

    return;
}
