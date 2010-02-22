#include <string>
#include <list>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

typedef struct{
int srcLine;
int dstLine;
string srcArray;
string dstArray;
string sets;
} dep_t;


// Forward Function Declarations
int parse_petit_output(std::ifstream &ifs);
int readNextSource(string line, string &source, std::ifstream &ifs);
int readNextDestination(string line, string source, std::ifstream &ifs);
bool isEOR(string line);
void store_dep(list<dep_t> depList, dep_t dep);
string skipToNext(std::ifstream &ifs);


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
    int i, pos, intvalue=0;
    stringstream ss;
    string line, source;

    i=0;
    while( getline(ifs,line) ){

//cout << "Before Source: " << line << endl;
        if( readNextSource(line, source, ifs) ){ return -1; }
        line = skipToNext(ifs);
        while( 1 ){
            if( readNextDestination(line, source, ifs) ){ return -1; }
            line = skipToNext(ifs);
            if( line.empty() || line.find("#####") != string::npos ){ break; }
        }

    }
}

int readNextSource(string line, string &source, ifstream &ifs){
    int pos;
//    while( pos == string::npos ){
//        if( !getline(ifs,line) ) return 0;
//        pos = line.find("### SOURCE:");
//    }
//    while( isEOR(line) ){
//        // If we hit the end of the file we can return
//        if( !getline(ifs,line) ) return 0;
//    }

    pos = line.find("### SOURCE:");
    while( pos == string::npos ){
        if( !getline(ifs,line) ) return 0;
        pos = line.find("### SOURCE:");
//cout << line << endl;
    }

    pos = line.find(":");
    // pos+2 to skip the ":" and the empty space following it
    line = line.substr(pos+2);
    if( line.empty() ){
        cerr << "Empty SOURCE" << endl;
        return -1;
    }else{
        source = line;
        cout << "New Source Found: "<< source << endl;
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
    list<dep_t> flow_deps, output_deps;
    stringstream ss;
    string sink, type, srcLine, junk, dstLine;
//    string sink, type, srcLine, srcArray, junk, dstLine, dstArray;
    dep_t dep;

/*
    while( 1 ){
        int pos = line.find("#########");
        if( pos != string::npos ){ return 0; }
        pos = line.find("===>");
        if( pos != string::npos ){ break; }
//        if( !line.empty() ){ break; }
        if( !getline(ifs,line) ) return 1;
    }
*/

/*
    // Move to the next non-EOR line
    while( isEOR(line) ){
        // If we hit the end of the file we can return
        if( !getline(ifs,line) ) return 1;
    }
*/

    // read the sink of this dependency
    int pos = line.find("===>");
    if( pos != string::npos ){
        pos = line.find(">");
        // pos+2 to skip the ">" and the empty space following it
        line = line.substr(pos+2);
        if( line.empty() ){
            cerr << "Empty sink" << endl;
            return -1;
        }else{
            sink = line;
            cout << "New Sink Found: "<< source << " --> " << sink << endl;
        }
    }

    // read the details of this dependency
    if( !getline(ifs,line) || isEOR(line) ){ return 0; }

    ss << line;
    ss >> type >> srcLine >> dep.srcArray >> junk >> dstLine >> dep.dstArray;

    // Remove the ":" from the source and destination line.
    pos = srcLine.find(":");
    if( pos != string::npos ){
        dep.srcLine = atoi( srcLine.substr(0,pos).c_str() );
    }
    pos = dstLine.find(":");
    if( pos != string::npos ){
        dep.dstLine = atoi( dstLine.substr(0,pos).c_str() );
    }

//    cout << "type: " << type << " src line: " << srcLine << " src array: " << srcArray
//         << " dst line: " << dstLine << " dst array: " << dstArray << endl;

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

/*
    // Move to the next EOR line
    while( !isEOR(line) ){
        // If we hit the end of the file we can return
        if( !getline(ifs,line) ) return 0;
    }
*/

    return 0;
}


void store_dep(list<dep_t> depList, dep_t dep){
    list<dep_t>::iterator itr;
    depList.push_back(dep);
/*
    for(itr=depList.begin(); itr != depList.end(); ++itr) {
        dep_t *tmp = dynamic_cast<dep_t *>*it;
        if( tmp->srcLine < dep->srcLine ){
        
        }
    }
*/
    return;
}

// is this line an End Of Record
bool isEOR(string line){
    int pos = line.find("#########");
    if( pos != string::npos || line.empty() ){
        return true;
    }
    return false;
}

string skipToNext(ifstream &ifs){
    string line;
    while( 1 ){
        if( !getline(ifs,line) ) return string("");
        int pos = line.find("===>");
        if( pos != string::npos ){ break; }
        pos = line.find("#######");
        if( pos != string::npos ){ break; }
    }
    return line;
}
