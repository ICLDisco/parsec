#include "kill.hpp"
#include <stdio.h>

list<dep_t> flow_deps, output_deps, merged_deps;

// Forward Function Declarations
int parse_petit_output(std::ifstream &ifs);
int readNextSource(string line, string &source, std::ifstream &ifs);
int readNextDestination(string line, string source, std::ifstream &ifs);
bool isEOR(string line);
void store_dep(list<dep_t> &depList, dep_t dep);
string skipToNext(std::ifstream &ifs);
//void dumpList(list<dep_t> depList);
void mergeLists(void);
void dumpDep(dep_t dep);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// C++ code

string SetIntersector::itos(int num){
    stringstream out;
    out << num;
    return out.str();
}

void SetIntersector::setFD1(dep_t fd){
    f_dep = fd;
//    o_deps.clear();
    composed_deps.clear();
    cmpsd_counter = 0;
}

void SetIntersector::setOD(dep_t od){
    o_dep = od;
}
  
void SetIntersector::setFD2(dep_t fd){
    f_dep2 = fd;
}

void SetIntersector::compose_FD2_OD(){
    string cmpsd = "cmpsd"+itos(cmpsd_counter);
    cmpsd += " := "+f_dep2.sets+" compose "+o_dep.sets+";";
    composed_deps.push_back(cmpsd);
    ++cmpsd_counter;
}

string SetIntersector::subtract(){
    stringstream ret_val;
    ret_val << "symbolic ";
    set<string>::iterator sym_itr;
    for(sym_itr=symbolic_vars.begin(); sym_itr != symbolic_vars.end(); ++sym_itr) {
        string sym_var = *sym_itr;
        if( sym_itr!=symbolic_vars.begin() )
            ret_val << ", ";
        ret_val << sym_var;
    }
    ret_val << ";" << endl;

    ret_val << "f1 := " << f_dep.sets << ";" << endl;

    list<string>::iterator cd_itr;
    for(cd_itr=composed_deps.begin(); cd_itr != composed_deps.end(); ++cd_itr) {
        string cd = *cd_itr;
        ret_val << cd <<  endl;
    }

    ret_val << "R := f1";
    for(int i=0; i<cmpsd_counter; i++){
        if( i )
            ret_val << " union ";
        else
            ret_val << " - ( ";

        ret_val << "cmpsd" << i;

        if( i == cmpsd_counter-1 )
            ret_val << " )";
    }
    ret_val << ";" << endl;
    ret_val << "R;" << endl;

    return ret_val.str();
}

#if 0
void SetIntersector::addOD(dep_t od){
    o_deps.push_back(od);
}

string SetIntersector::intersect(){
    stringstream ret_val;
    ret_val << "symbolic ";
    set<string>::iterator sym_itr;
    for(sym_itr=symbolic_vars.begin(); sym_itr != symbolic_vars.end(); ++sym_itr) {
        string sym_var = *sym_itr;
        if( sym_itr!=symbolic_vars.begin() )
            ret_val << ", ";
        ret_val << sym_var;
    }
    ret_val << ";" << endl;

    ret_val << "fd := " << rel_to_set(f_dep.sets) << ";" << endl;

    list<dep_t>::iterator od_itr;
    int counter = 0;
    for(od_itr=o_deps.begin(); od_itr != o_deps.end(); ++od_itr) {
        dep_t od = *od_itr;
        ret_val << "od" << counter << " := " << rel_to_set(od.sets) << ";" << endl;
        ++counter;
    }

    ret_val << "R := fd";
    for(int i=0; i<counter; i++){
        if( i )
            ret_val << " union ";
        else
            ret_val << " - ( ";

        ret_val << "od" << i;

        if( i == counter-1 )
            ret_val << " )";
    }
    ret_val << ";" << endl;
    ret_val << "R;" << endl;
    return ret_val.str();

}
#endif




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// C-like code begins
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
    }

    mergeLists();

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
    ss >> type >> srcLine >> dep.srcArray >> junk >> dstLine >> dep.dstArray >> dep.loop ;

    dep.type = type;
    if( dep.loop.find("[") != string::npos ){
        dep.loop.clear();
    }

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

/*
f DSSSSM(k,n,m) -> DSSSSM(k',n,m)
    50 A(m,n) -> 45 A(m,n) {[k,n,m] -> [k',n,m] : 0 <= k < k' < n,m < BB} (+,0,0)

{[k,n,m] -> [k+1,n,m] : 0 <= k <= n-2, m-2 && n < BB && m < BB}
*/

string trim(string in){
    int i;
    for(i=0; i<in.length(); ++i){
        if(in[i] != ' ')
            break;
    }
    return in.substr(i);
}

void dumpDep(dep_t dep, string iv_set){
    stringstream ss;
    string srcParams, dstParams, junk;
    unsigned int posLB, posRB, posCOL;

    if( iv_set.find("FALSE") != string::npos )
        return;

    ss << iv_set;
    ss >> srcParams >> junk >> dstParams;
    posLB = srcParams.find("[");
    posRB = srcParams.find("]");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed set: \"" << iv_set << "\"" << endl; 
       return;
    }
    srcParams = srcParams.substr(posLB+1,posRB-2);

    posLB = dstParams.find("[");
    posRB = dstParams.find("]");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed set: \"" << iv_set << "\"" << endl; 
       return;
    }

    dstParams = dstParams.substr(posLB+1,posRB-1);
    string cond = ss.str();
    posCOL = cond.find(":");
    if( posCOL == string::npos ){
       cerr << "Malformed conditions: \"" << iv_set << "\"" << endl; 
       return;
    }

    cout << dep.source << "(" << srcParams << ") ";
    cout << dep.srcArray << " -> ";
    cout << dep.dstArray << " ";
    cout << dep.sink << "(" << dstParams << ")  ";
    cout << "{" << trim(cond.substr(posCOL+1)) << endl;
/*
    cout << dep.source << "(" << srcParams << ") -> ";
    cout << dep.sink << "(" << dstParams << ")" << endl;
    cout << "    " << dep.srcLine << " ";
    cout << dep.srcArray << " -> ";
    cout << dep.dstLine << " ";
    cout << dep.dstArray << " ";
    cout << dep.sets << " ";
    cout << dep.loop << endl;
*/
}

#if 0
void dumpList(list<dep_t> depList){
    list<dep_t>::iterator itr;

    for(itr=depList.begin(); itr != depList.end(); ++itr) {
        dep_t dep = *itr;
        dumpDep(dep);
    }
}
#endif

void mergeLists(void){
    bool found;
    list<dep_t>::iterator fd_itr; // flow dep iterator
    list<dep_t>::iterator od_itr; // out dep iterator
    list<dep_t>::iterator fd2_itr; // second flow dep iterator
    set<int> srcSet;
    set<int>::iterator src_itr;
    set<string> fake_it;

    fake_it.insert("BB");
    SetIntersector setIntersector(fake_it);

    // Insert every source of flow deps in a set (srcSet).
    for(fd_itr=flow_deps.begin(); fd_itr != flow_deps.end(); ++fd_itr) {
        found = false;
        dep_t f_dep = *fd_itr;
        srcSet.insert(f_dep.srcLine);
    }

    // For every source of a flow dep
    for (src_itr=srcSet.begin(); src_itr!=srcSet.end(); ++src_itr){
        int source = static_cast<int>(*src_itr);
        // Find all flow deps that flow from this source
        for(fd_itr=flow_deps.begin(); fd_itr != flow_deps.end(); ++fd_itr) {
            dep_t f_dep = static_cast<dep_t>(*fd_itr);
            int fd_srcLine = f_dep.srcLine;
            int fd_dstLine = f_dep.dstLine;
            if( fd_srcLine == source ){
                setIntersector.setFD1(f_dep);
                // Find and print every output dep that has the same source as this flow
                // dep and its destination is either a) the same as the source, or
                // b) the source of a second flow dep which has the same destination
                // as "fd_dstLine"
                for(od_itr=output_deps.begin(); od_itr != output_deps.end(); ++od_itr) {
                    dep_t o_dep = *od_itr;
                    int od_srcLine = o_dep.srcLine;
                    int od_dstLine = o_dep.dstLine;

                    if( fd_srcLine != od_srcLine ){
                        continue;
                    }

                    setIntersector.setOD(o_dep);

                    // case (a)
                    if( od_srcLine == od_dstLine ){
                        // Yes, the same as the original fd because the od is (cyclic) on myself.
                        setIntersector.setFD2(f_dep);
                        setIntersector.compose_FD2_OD();
                    }else{ // case (b)
                        for(fd2_itr=flow_deps.begin(); fd2_itr != flow_deps.end(); ++fd2_itr) {
                            dep_t f2_dep = *fd2_itr;
                            int fd2_srcLine = f2_dep.srcLine;
                            int fd2_dstLine = f2_dep.dstLine;
                            if( fd2_srcLine == od_dstLine && fd2_dstLine == fd_dstLine ){
                                setIntersector.setFD2(f2_dep);
                                setIntersector.compose_FD2_OD();
                            }
                        }
                    }
                }
                string sets_for_omega = setIntersector.subtract();
//                cout << sets_for_omega << endl;
                fstream filestr ("/tmp/oc_in.txt", fstream::out);
                filestr << sets_for_omega << endl;
                filestr.close();

                FILE *pfp = popen("/Users/adanalis/Desktop/Research/PLASMA_Distributed/Omega/omega_calc/obj/oc /tmp/oc_in.txt", "r");
                stringstream data;
                char buffer[256];
                while (!feof(pfp)){
                    if (fgets(buffer, 256, pfp) != NULL){
                        data << buffer;
                    }
                }
                pclose(pfp);
                string line;
                while( getline(data, line) ){
                    if( line.find("#") && !line.empty() ){
//                        cout << line << endl;
                        break;
                    }
                }
//                cout << endl;
                dumpDep(f_dep, line);
            }
        }
        cout << "-------------------------------" << endl;
    }

    return;
}
