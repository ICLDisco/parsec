#include "kill.hpp"
#include <stdio.h>
#include <ctype.h>

list<dep_t> flow_deps, output_deps, merged_deps;
map<string,task_t> taskMap;
string omegaHome;

// Forward Function Declarations
int parse_petit_output(std::istream &ifs);
int readNextSource(string line, string &source, std::istream &ifs);
int readNextDestination(string line, string source, std::istream &ifs);
int readNextTaskInfo(string line);
list<string> parseTaskParamSpace(string params);
map<string,string> parseSymbolicVars(string vars);
string skipToNext(std::istream &ifs);
bool isEOR(string line);
void store_dep(list<dep_t> &depList, dep_t dep);
void mergeLists(void);
bool processDep(dep_t dep, string dep_set, task_t &currTask, bool isInversed);
void dumpDep(dep_t dep, string iv_set, bool isInversed);
bool isFakeVariable(string var);
string removeVarFromCond(string var, string condStr);
string removeVarFromSimpleCond(string var, string condStr);
string expressionToRange(string var, string condStr);
string offsetVariables( string vars, string off );
bool isExpressionOrConst(string var);
string makeMinorFormatingChanges(string str);

string trimAll(string str);
string removeWS(string str);

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


string SetIntersector::inverse(string dep_set){
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

    ret_val << "s := " << dep_set << ";" << endl;
    ret_val << "inverse s;" << endl;

    return ret_val.str();
}


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

    // find Omega
    char *oH = getenv("OMEGA_HOME");
    if( !oH ){
        cerr << "environment variable \"OMEGA_HOME\" is not set" << endl;
        exit(-1);
    }
    omegaHome = string(getenv("OMEGA_HOME"));
    ifstream test( (omegaHome+"/omega_calc/obj/oc").c_str() );
    if( !test ){
        cerr << "ERROR: either environment variable \"OMEGA_HOME\" is wrong, ";
        cerr << "or the Omega calculator is not in the expected place: ";
        cerr << omegaHome+"/omega_calc/obj/oc" << endl;
        exit(-1);
    }
  
    fName = argv[1];

    if( !string(fName).compare("-") ){
        parse_petit_output(cin);
    }else{
        ifstream ifs( fName );
        if( !ifs ){
            cerr << "File \""<< fName <<"\" does not exist" << endl;
            exit(-1);
        }
        parse_petit_output(ifs);
    }

    return 0;
}

int parse_petit_output(istream &ifs){
    string line, source;

    flow_deps.clear();
    output_deps.clear();

    // First read the task name and parameter space information
    while( getline(ifs,line) ){
        if( readNextTaskInfo(line) < 0 )
            break;
    }

    // Then read the body of the file with the actual dependencies
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

//DSSSSM(k,n,m) {k=0..BB-1,n=k+1..BB-1,m=k+1..BB-1} A:A(k, n),B:A(m, n),C:L(m, k),D:A(m, k),E:IPIV(m, k)
int readNextTaskInfo(string line){
    static bool in_task_section=false;

    if( !line.compare("TASK SECTION START") ){
        in_task_section=true;
        return 0;
    }

    if( !in_task_section )
        return 0;

    if( !line.compare("TASK SECTION END") ){
        in_task_section=false;
        return -1;
    }

    unsigned int lb_pos, rb_pos;
    lb_pos = line.find(" {");
    rb_pos = line.find("} ");
    if( lb_pos == string::npos || rb_pos == string::npos ){
        cerr << "ERROR: Malformed Task Info entry: \"" << line << "\"" << endl; 
        return -1;
    }
    string taskName = line.substr(0,lb_pos);
    string taskParamSpace = line.substr(lb_pos+1,rb_pos-lb_pos);
    string symVars = line.substr(rb_pos+2);

    task_t task;
    task.name = taskName;
    task.paramSpace = parseTaskParamSpace(taskParamSpace);
    task.symbolicVars = parseSymbolicVars(symVars);
    taskMap[taskName] = task;

    return 1;
}


//A:A(k, n)|B:A(m, n)|C:L(m, k)|D:A(m, k)|E:IPIV(m, k)
map<string,string> parseSymbolicVars(string vars){
    map<string, string> sVars;
    string var;
    unsigned int cm_pos, cl_pos;

    // Tasks IN() and OUT() will not have symbolic variables
    if( vars.empty() ) return sVars;

    cm_pos = vars.find("|");
    while( cm_pos != string::npos ){

        string var = vars.substr(0,cm_pos);
        cl_pos = var.find(":");
        if( cl_pos == string::npos ){
            cerr << "ERROR: Malformed Task symbolic variables: \"" << vars << "\"" << endl; 
            exit(-1);
        }
        string arrayName = removeWS(var.substr(cl_pos+1));
        string symbolic  = var.substr(0,cl_pos);
        sVars[arrayName] = symbolic;

        // skip the part of the string we just processed and start over again.
        vars = vars.substr(cm_pos+1);
        cm_pos = vars.find("|");
    }

    var = vars;
    cl_pos = var.find(":");
    if( cl_pos == string::npos ){
        cerr << "ERROR: Malformed Task symbolic variables: \"" << vars << "\"" << endl; 
        exit(-1);
    }
    string arrayName = removeWS(var.substr(cl_pos+1));
    string symbolic  = var.substr(0,cl_pos);
    sVars[arrayName] = symbolic;

    return sVars;
}

list<string> parseTaskParamSpace(string params){
    list<string> paramSpace;

    // Verify that the string starts with a left bracket and then get rid of it.
    unsigned int pos = params.find("{");
    if( pos == string::npos ){
        cerr << "ERROR: Malformed Task parameter space entry: \"" << params << "\"" << endl; 
        return paramSpace;
    }
    params = params.substr(pos+1);

    // Verify that the string ends with a right bracket and then get rid of it.
    pos = params.find("}");
    if( pos == string::npos ){
        cerr << "ERROR: Malformed Task parameter space entry: \"" << params << "\"" << endl; 
        return paramSpace;
    }
    params = params.substr(0,pos);

    // Split the string at the commas and put the parts into a list
    pos = params.find(",");
    while( pos != string::npos ){
        // Take the string before the comma and put it in the list
        paramSpace.push_back(params.substr(0,pos));
        // Get rid of the part of the string before the comma
        params = params.substr(pos+1);
        // Look for another comma and start all over again
        pos = params.find(",");
    }
    // Take the last part of the string (if any) and put it in the list
    if(!params.empty())
    paramSpace.push_back(params);
    
    return paramSpace;
}


int readNextSource(string line, string &source, istream &ifs){
    unsigned int pos;

    // Keep reading input lines until you hit one that matches the pattern
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
int readNextDestination(string line, string source, istream &ifs){
    stringstream ss;
    string sink, type, srcLine, junk, dstLine;
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

string skipToNext(istream &ifs){
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

string trim(string in){
    int i;
    for(i=0; i<in.length(); ++i){
        if(in[i] != ' ')
            break;
    }
    return in.substr(i);
}

string trimAll(string str){
    unsigned int s,e;
    for(s=0; s<str.length(); ++s){
        if( str[s] != ' ' ) break;
    }
    for(e=s; e<str.length(); ++e){
        if( str[e] == ' ' ) break;
    }
    return str.substr(s,e-s);
}


string removeWS(string str){
    unsigned int s,e;
    string rslt;

    for(s=0; s<str.length(); ++s){
        if( str[s] != ' ' ) break;
    }
    for(e=s; e<str.length(); ++e){
        if( str[e] == ' ' ) continue;
        rslt += str[e];
    }
    return rslt;
}


list<string> stringToVarList( string str ){
    list<string> result;
    stringstream ss;

    ss << str;

    while (!ss.eof()) {
        string token;    
        getline(ss, token, ',');
        result.push_back(token);
    }

    return result;
}


string removeVarFromCond(string var, string condStr){
    string resultCond;
    int loglOpLen;

    if( condStr.find(var) == string::npos )
        return condStr;

    // If the condition is a logical combination of multiple simpler
    // conditions, process each simple condition individually. 
    unsigned int pos = condStr.find_first_of("&|");
    while( pos != string::npos ){
        // See if Omega uses the double symbols "&&" and "||" or the single "&" and "|"
        if( condStr.find("&&") != string::npos || condStr.find("||") != string::npos ){
            loglOpLen = 2;
        }else{
            loglOpLen = 1;
        }
        // Take the first simple condition, clean it and put it in a new string
        resultCond += removeVarFromSimpleCond( var, condStr.substr(0,pos) );
        // Add the logical operator to the new string
        resultCond += " "+condStr.substr(pos,loglOpLen)+" ";

        condStr = condStr.substr(pos+loglOpLen);
        pos = condStr.find_first_of("&|");
    }
    resultCond += removeVarFromSimpleCond( var, condStr );

    // return the new string containing the cleaned condition.
    return resultCond;
}


/*
 * Warning: If after removing the variable from the condition, the condition ends up
 * having no "=" or "<" symbols (i.e. the condition has devolved into a single variable,
 * or list of variables but no comparison operators), then we return a condition
 * that is always true (in particular: "1>0").
 */
string removeVarFromSimpleCond(string var, string condStr){
     unsigned int pos;

     // Remove the white spaces to simplify parsing.
     condStr = removeWS(condStr);

     pos = condStr.find(var);
     // if we found the var, split the string in "left" and "right" around the var
     if( pos != string::npos ){
         string left = condStr.substr(0,pos);
         string right = condStr.substr(pos+var.length());

         // Check if the variable is part of a comma separated list of variables.
         unsigned int beforV = pos-1;
         unsigned int afterV = pos+var.length()+1;

         // if there is a comma before/after the variable remove the variable and the comma
         // and leave the remaining condition untouched.
         if( beforV >= 0 && condStr[beforV] == ',' ){
             left = condStr.substr(0,beforV);
             string result = left+right;
             if( result.find_first_of("=<") == string::npos )
                 return string("1>0");
             return result;
         }
         if( condStr.length() > afterV && condStr[afterV] == ',' ){
             right = condStr.substr(afterV);
             string result = left+right;
             if( result.find_first_of("=<") == string::npos )
                 return string("1>0");
             return result;
         }

         // if the comparison operator to the right of the variable is "<=" then we can just
         // remove the variable and the operator and leave the rest of the condition untouched.
         pos = right.find("<=");
         if( pos != string::npos ){
             return left+right.substr(pos+2);
         }

         pos = right.find("<");
         if( pos != string::npos ){
             // take the part of the string after the "<" or "<=" symbol. This should contain
             // the upper bound, probably followed by other stuff starting with "<".
             right = right.substr(pos+1);
             string offVar, result;

             unsigned int posS = 0;
             unsigned int posE = right.find("<");
             while( posE != string::npos ){
                 offVar = offsetVariables( right.substr(posS,posE), "-1" );
                 result += offVar;
                 if( right.length() > posE+1 && right[posE+1] == '=' ){
                     result += " <= ";
                     posS = posE+2;
                 }else{
                     result += " < ";
                     posS = posE+1;
                 }
                 posE = right.find("<", posS);
             }
             offVar = offsetVariables( right.substr(posS), "-1" );
             result += offVar;

             return left+result;
         }else{
             cerr << "ERROR: condition ends in temp variable: \"" << condStr << "\"" << endl;
         }
     }
     return condStr;
}


// Append to all comma separated variables the string "off"
string offsetVariables( string vars, string off ){
    string result;

    vars = removeWS(vars);

    unsigned int pos = vars.find(",");
    while( pos != string::npos ){
        string tmp = vars.substr(0,pos);
        result += tmp+off+",";
        vars = vars.substr(pos+1);
        pos = vars.find(",");
    }
    result += vars+off;
   
    return result;
}

string expressionToRange(string var, string condStr){
    string lb, ub, off;
    list<string> conditions;

    // split the condition string into a list of strings
    // each of which holding only a simple expression
    while( !condStr.empty() ){
        unsigned int pos = condStr.find_first_of("&|");
        if( pos != string::npos && pos ){
            conditions.push_back( condStr.substr(0,pos) );
            condStr = condStr.substr(pos+1);
        }else{
            conditions.push_back( condStr );
            break;
        }
        conditions.push_back( condStr );
    }

    // For every expression in the list, find the variable and then
    // look for a "<" or "<=" left and/or right of the variable.
    list<string>::iterator cnd_itr=conditions.begin();
    for (; cnd_itr!=conditions.end(); ++cnd_itr){
        unsigned int pos;
        string cond = *cnd_itr;
        pos = cond.find(var);
        // if we found the var, split the string in "left" and "right" around the var
        if( pos != string::npos ){
            string left = cond.substr(0,pos);
            string right = cond.substr(pos+var.length());

            // Process the "left" string looking for the lower bound "lb"
            pos = left.find_last_of("<");
            if( pos != string::npos ){
                // check if the comparison operator is "<=".  If so the expression to the left of the
                // operator is the lower bound, otherwise we need to offset the expression by "+1"
                if( (left.length() > pos+1) && (left[pos+1] == '=') ){
                    off = "";
                }else{
                    off = "+1";
                }

                // take the part of the string up to the "<" symbol. This should contain
                // the lower bound, probably preceeded by other stuff ending in "<" or "="
                string tmp = left.substr(0,pos);
                // find the last occurrence of "<" or "=" (left of the lower bound)
                unsigned int l_pos = tmp.find_last_of("<=");
                if( l_pos != string::npos ){
                    lb = trimAll(tmp.substr(l_pos+1));
                }else{ // if no such symbol, then the whole thing is the lower bound.
                    lb = trimAll(tmp);
                }

                lb = lb.append(off);
            }

            // Process the "right" string looking for the upper bound "ub"
            int op_len=1;
            // assume it's "<". If it ends up being "<=" we will overwrite it.
            off = "-1";
            pos = right.find("<=");
            if( pos != string::npos ){
                    off = "";
                    op_len = 2;
            }else{
                pos = right.find("<");
            }
            if( pos != string::npos ){
                // take the part of the string after the "<" or "<=" symbol. This should contain
                // the upper bound, probably followed by other stuff starting with "<".
                string tmp = right.substr(pos+op_len);

                // find the first occurrence of "<" (right of the upper bound)
                unsigned int r_pos = tmp.find("<");
                if( r_pos != string::npos ){
                    ub = trimAll(tmp.substr(0,r_pos));
                }else{ // if no such symbol, then the whole thing is the upper bound.
                    ub = trimAll(tmp);
                }

                ub = ub.append(off);
            }

        }
    }

    return (lb+".."+ub);
}


void dumpDep(stringstream &ss, dep_t dep, string iv_set, bool isInversed){
    if( isInversed ){
        ss << dep.sink << " " << dep.dstArray;
        ss << " <- " << dep.source << " " << dep.srcArray;
        ss << " " << iv_set << endl;
    }else{
        ss << dep.source << " " << dep.srcArray;
        ss << " -> " << dep.sink << " " << dep.dstArray;
        ss << " " << iv_set << endl;
    }
    return;
}

void dumpMap(stringstream &ss, map<string,string> sV){
    map<string,string>::iterator it;
    for(it=sV.begin(); it!=sV.end(); ++it){
        ss << (*it).first << "==" << (*it).second << "\n";
    }
    ss << endl;
}

bool processDep(dep_t dep, string iv_set, task_t &currTask, bool isInversed){
    stringstream ss, ss0;
    string srcParams, dstParams, junk;
    unsigned int posLB, posRB, posCOL;
    list<string> srcFormals, dstFormals;

    // if it is an impossible dependency, do not print anything.
    if( iv_set.find("FALSE") != string::npos )
        return false;

    // Get the list of formal parameters of the source task (k,m,n,...)
    posLB = dep.source.find("(");
    posRB = dep.source.find(")");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed dependency source: \"" << dep.source << "\"" << endl; 
       return false;
    }
    string srcFrmlStr = dep.source.substr(posLB+1,posRB-posLB-1);
    if( isInversed )
        dstFormals = stringToVarList( srcFrmlStr );
    else
        srcFormals = stringToVarList( srcFrmlStr );

    // Get the list of formal parameters of the destination task (k,m,n,...)
    posLB = dep.sink.find("(");
    posRB = dep.sink.find(")");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed dependency sink: \"" << dep.sink << "\"" << endl; 
       return false;
    }
    string dstFrmlStr = dep.sink.substr(posLB+1,posRB-posLB-1);
    if( isInversed )
        srcFormals = stringToVarList( dstFrmlStr );
    else
        dstFormals = stringToVarList( dstFrmlStr );

    // Process the sets of actual parameters
    ss0 << iv_set;
    ss0 >> srcParams >> junk >> dstParams;

    // Get the list of actual parameters of the source task
    posLB = srcParams.find("[");
    posRB = srcParams.find("]");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed set: \"" << iv_set << "\"" << endl; 
       return false;
    }
    srcParams = srcParams.substr(posLB+1,posRB-posLB-1);
    list<string> srcActuals = stringToVarList(srcParams);

    // Get the list of actual parameters of the destination task
    posLB = dstParams.find("[");
    posRB = dstParams.find("]");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed set: \"" << iv_set << "\"" << endl; 
       return false;
    }
    dstParams = dstParams.substr(posLB+1,posRB-posLB-1);
    list<string> dstActuals = stringToVarList(dstParams);

    // Get the conditions that Omega told us and clean up the string
    string cond = ss0.str();
    posCOL = cond.find(":");
    posRB = cond.find("}");
    if( posCOL == string::npos || posRB == string::npos ){
       cerr << "Malformed conditions: \"" << iv_set << "\"" << endl; 
       return false;
    }
    cond = trim(cond.substr(posCOL+1, posRB-posCOL-1));

    // Remove the formals from the dep.source string
    posLB = dep.source.find("(");
    if( posLB == string::npos ){
       cerr << "Malformed dependency source: \"" << dep.source << "\"" << endl; 
       return false;
    }
    string source = dep.source.substr(0,posLB);

    // Remove the formals from the dep.sink string
    posLB = dep.sink.find("(");
    if( posLB == string::npos ){
       cerr << "Malformed dependency sink: \"" << dep.sink << "\"" << endl; 
       return false;
    }
    string sink = dep.sink.substr(0,posLB);
    
    
    if( srcFormals.size() != srcActuals.size() ){
        cerr << "ERROR: source formal count != source actual count" << endl;
    }

    list<string>::iterator srcF_itr;
    list<string>::iterator srcA_itr;

    // For every source actual that is an "In_1" type variable (i.e. "In_" followed by number)
    // which is what Omega will introduce when we inverse the sets, replace it with the
    // corresponding formal, in the source and destination sets as well as in the conditions.
    // Also do that for the special induction variables "ii" and "jj".
    if(isInversed){
        srcF_itr=srcFormals.begin();
        srcA_itr=srcActuals.begin();
        for (; srcF_itr!=srcFormals.end(); ++srcF_itr, ++srcA_itr){
            string fParam = *srcF_itr;
            string aParam = *srcA_itr;
            if( isFakeVariable(aParam) && fParam.compare(aParam) != 0 ){
                // Replace the variable in the actual parameter with the one from the formal
                *srcA_itr = fParam;
                // Do the same for all the destination actuals
                list<string>::iterator dstA_itr=dstActuals.begin();
                for (; dstA_itr!=dstActuals.end(); ++dstA_itr){
                    string dstaParam = *dstA_itr;
                    unsigned int pos = dstaParam.find(aParam);
                    if( pos != string::npos ){
                        (*dstA_itr).replace(pos,aParam.length(), fParam);
                    }
                }
                // Do the same for all the occurances of the variable in the condition
                unsigned int pos = cond.find(aParam);
                while( pos != string::npos ){
                    cond.replace(pos,aParam.length(), fParam);
                    pos = cond.find(aParam);
                }
            }
        }
    }

    // For every source actual that is not the same variable as the formal do the following:
    // A) If we are looking at an OUT dep, then the actual is an expression and we need to
    // add the condition (formal_variable=actual_expression) into the conditions.
    // B) If we are looking at an IN dep, then if:
    //   i) the actual is a constant, we should do as we do for the OUTs, add the condition
    //      (formal_variable=actual_expression) into the conditions.
    //  ii) otherwise, we should (carefully) replace the actual with the formal in the 
    //      destination actuals and the conditions.  Carefully means that we should replace
    //      the vars in a copy string, while finding them in the original, so we don't have
    //      cascading replaces.  To better understand the problem, consider the operations: 
    //      replace (m) with (k) _AND_ replace (m') with (m) _AND_ replace (k) with (n)

    // create the copies
    string newCond = cond;
    list<string>newDstActuals( dstActuals ); 

    srcF_itr   = srcFormals.begin();
    srcA_itr   = srcActuals.begin();
    for (; srcF_itr!=srcFormals.end(); ++srcF_itr, ++srcA_itr){
        string fParam = *srcF_itr;
        string aParam = *srcA_itr;
        if( fParam.compare(aParam) != 0 ){ // if they are different
            if( isInversed ){ // if an IN dep
                if( isExpressionOrConst(aParam) ){
                    newCond = newCond.append(" && ("+fParam+"="+aParam+") ");
                }else{
                    // Fix the destination actuals. Since we are looking at an IN dep
                    // dstActuals are the actuals of the TASK that sends us this data.
                    list<string>::iterator newDstA_itr, dstA_itr;
                    newDstA_itr= newDstActuals.begin();
                    dstA_itr   = dstActuals.begin();
                    for (; dstA_itr!=dstActuals.end(); ++dstA_itr, ++newDstA_itr){
                        // if the original list has this actual, replace it in the copy
                        if( !(*dstA_itr).compare(aParam) ){
                            *newDstA_itr = fParam;
                        }
                    }

                    // Fix the conditions.
                    unsigned int pos = cond.find(aParam);
                    while( pos != string::npos ){
                        newCond.replace(pos,aParam.length(), fParam);
                        pos = cond.find(aParam, pos+1);
                    }
                }
            }else{
                newCond = newCond.append(" && ("+fParam+"="+aParam+") ");
            }
        }
    }
    // apply the copies onto the originals
    cond = newCond;
    dstActuals = newDstActuals;
#ifdef DEBUG
    ss << cond << endl;
#endif

    list<string> actual_parameter_list;

    // For every destination actual, check if is exists among the source formals.
    // If it doesn't and it is not an expression, replace it with a lb..ub expression
    // and then remove it from the condition.
    // However, if the source task is the special task "IN()", just replace the actuals
    // with the actuals of the destination array.
    list<string>::iterator dstF_itr=dstFormals.begin();
    list<string>::iterator dstA_itr=dstActuals.begin();
    for (; dstF_itr!=dstFormals.end(); ++dstF_itr, ++dstA_itr){
        bool found=false;
        string dstfParam = *dstF_itr;
        string dstaParam = *dstA_itr;
        // look through the source formals to see if it's there
        list<string>::iterator srcF_itr=srcFormals.begin();
        for (; srcF_itr!=srcFormals.end(); ++srcF_itr){
            string srcfParam = *srcF_itr;
            if( !srcfParam.compare(dstaParam) ){
                found = true;
                break;
            }
        }
        if( found ){
            actual_parameter_list.push_back(dstaParam);
            continue;
        }

        if( isExpressionOrConst(dstaParam) ){
            actual_parameter_list.push_back(dstaParam);
        }else{
            string range = expressionToRange(dstaParam, cond);
            cond = removeVarFromCond(dstaParam, cond);
            actual_parameter_list.push_back(range);
        }
    }

    //
    // Manipulation is over, output code follows
    //

    task_t thisTask;
    task_t peerTask;

    if( isInversed ){
        thisTask = taskMap[dep.sink];
        peerTask = taskMap[dep.source];
    }else{
        thisTask = taskMap[dep.source];
        peerTask = taskMap[dep.sink];
    }

    map<string,string> lcl_sV = thisTask.symbolicVars;
    map<string,string> rmt_sV = peerTask.symbolicVars;
    string localVar;

    // Find the alias under which the local variable is refered to in this task
    if( isInversed )
        localVar = lcl_sV[dep.dstArray];
    else
        localVar = lcl_sV[dep.srcArray];

    // see if the task already has a dep involving this local variable
    final_dep_t final_dep;
    list<final_dep_t>::iterator d_itr=currTask.deps.begin();
    for (; d_itr!=currTask.deps.end(); ++d_itr){
        final_dep_t tmp_dep = *d_itr;
        if( !tmp_dep.localVar.empty() && !tmp_dep.localVar.compare(localVar) ){
            final_dep = tmp_dep;
        }
    }

    if( final_dep.localVar.empty() )
        final_dep.localVar = localVar;

    // iterate over the newly created list of actual destination parameters and
    // create a comma separeted list in a string, so we can print it.
    string dstTaskParams;
    list<string>::iterator a_itr=actual_parameter_list.begin();
    for (; a_itr!=actual_parameter_list.end(); ++a_itr){
        string dstActualParam = *a_itr;
        if( a_itr != actual_parameter_list.begin() ){
            dstTaskParams = dstTaskParams.append(",");
        }
        dstTaskParams = dstTaskParams.append(dstActualParam);
    }


    if( isInversed ){

//        ss << "\n  /*" << lcl_sV[dep.dstArray] << " = " << dep.dstArray << "*/\n";
        ss0.str( "" );
        ss0 << "/*" << lcl_sV[dep.dstArray] << " == " << dep.dstArray << "*/";
        if( final_dep.localAlias.empty() )
            final_dep.localAlias = ss0.str();
        if( final_dep.localAlias.compare(ss0.str()) ){
            cerr << "ERROR: expecting \"" << final_dep.localAlias << "\" to match \"" << ss0.str() << "\"" << endl;
            cerr << "ERROR: " << currTask.name << endl;
            cerr << "ERROR: " << final_dep.localVar << endl;
            cerr << "ERROR: " << final_dep.type << "\n" << endl;
        }
  
        if( dep.source.find("IN") == string::npos ){
            //ss << "  /*" << rmt_sV[dep.srcArray] << " = " << dep.srcArray << "*/\n";
            ss0.str( "" );
            ss0 << "/*" << rmt_sV[dep.srcArray] << " == " << dep.srcArray << "*/";
            final_dep.remoteAliases.push_back( ss0.str() );
        }

        // Set of extend the type depending on what it was before
        if( final_dep.type.empty() )
            final_dep.type = "IN";

        if( !final_dep.type.compare("OUT") )
            final_dep.type = "INOUT";

//        ss << "  IN " << lcl_sV[dep.dstArray] << " <- ";
        ss0.str("");
        ss0 << "<- (" << cond << ") ? ";
        // If we are receiving from IN, strip the fake array indices that are in the petit file
        // and put in the actual parameters of the source Task.
        // Alternatively, we could use the fact that the dstArray has a literal meaning and it
        // is an actual array that exists in memory.
        if( dep.source.find("IN") != string::npos ){
            string srcArr = dep.srcArray;
            srcArr = srcArr.substr(0,srcArr.find("("));
            ss0 << srcArr << "(" << dstTaskParams << ") ";
        }else{
            ss0 << rmt_sV[dep.srcArray];
            // Only print the task name if it is NOT "IN"
            ss0 << " " << source << "("<< dstTaskParams <<") ";
        }
        string iE = makeMinorFormatingChanges(ss0.str());
        final_dep.inEdges.push_back(iE);
    }else{
//        map<string,string> lcl_sV = thisTask.symbolicVars;
//        map<string,string> rmt_sV = peerTask.symbolicVars;

//        ss << "\n  /*" << lcl_sV[dep.srcArray] << " == " << dep.srcArray << "*/\n";
        ss0.str("");
        ss0 << "/*" << lcl_sV[dep.srcArray] << " == " << dep.srcArray << "*/";
        if( final_dep.localAlias.empty() )
            final_dep.localAlias = ss0.str();
        if( final_dep.localAlias.compare(ss0.str()) ){
            cerr << "ERROR: expecting \"" << final_dep.localAlias << "\" to match \"" << ss0.str() << "\"" << endl;
            cerr << "ERROR: " << currTask.name << endl;
            cerr << "ERROR: " << final_dep.localVar << endl;
            cerr << "ERROR: " << final_dep.type << "\n" << endl;
        }
        final_dep.localAlias = ss0.str();

        if( dep.sink.find("OUT") == string::npos ){
//            ss << "  /*" << rmt_sV[dep.dstArray] << " == " << dep.dstArray << "*/\n";
            ss0.str("");
            ss0 << "/*" << rmt_sV[dep.dstArray] << " == " << dep.dstArray << "*/";
            final_dep.remoteAliases.push_back( ss0.str() );
        }

        // Set of extend the type depending on what it was before
        if( final_dep.type.empty() )
            final_dep.type = "OUT";

        if( !final_dep.type.compare("IN") )
            final_dep.type = "INOUT";

//        ss << "  OUT " << lcl_sV[dep.srcArray] << " -> ";
        ss0.str("");
        ss0 << "-> (" << cond << ") ? ";
        // If we are sending to OUT, strip the fake array indices that are in the petit file
        // and put in the actual parameters of the destination Task.
        if( dep.sink.find("OUT") != string::npos ){
            string dstArr = dep.dstArray;
            dstArr = dstArr.substr(0,dstArr.find("("));
            ss0 << dstArr << "(" << dstTaskParams << ") ";
        }else{
            ss0 << rmt_sV[dep.dstArray] << " ";
            // Only print the task name if it is NOT "OUT"
            ss0 << sink << "(" << dstTaskParams << ")  ";
        }
        string oE = makeMinorFormatingChanges(ss0.str());
        final_dep.outEdges.push_back(oE);
    }

#ifdef DEBUG
    ss << endl;
    dumpDep(ss, dep, iv_set, isInversed);
#endif

    // see if the task already has this dep.  If so, replace it, otherwise add it
    bool found=false;
    d_itr=currTask.deps.begin();
    for (; d_itr!=currTask.deps.end(); ++d_itr){
        final_dep_t tmp_dep = *d_itr;
        if( !tmp_dep.localVar.compare(localVar) ){
            found = true;
            *d_itr = final_dep;
            break;
        }
    }
    if( !found )
        currTask.deps.push_back( final_dep );

    return true;
}


string makeMinorFormatingChanges(string str){
    // Replace "&&" and "||" with "&" and "|" respectively.
    unsigned int pos = str.find("&&");
    while( pos != string::npos ){
        str.erase(pos,1);
        pos = str.find("&&");
    }

    pos = str.find("||");
    while( pos != string::npos ){
        str.erase(pos,1);
        pos = str.find("||");
    }

    // Replace "=" with "==" except if it's part of "<=" or ">="
    pos = str.find("=");
    if( pos == 0 || pos == str.length()-1){
        cerr << "Malformed dependency starts or ends with symbol \"=\": \"" << str << "\"" << endl; 
        return str;
    }
    while( pos != string::npos ){
        if( str[pos-1] != '<' && str[pos-1] != '>' && str[pos-1] != '=' && str[pos+1] != '=' ){
            str.insert(pos,"=");
            pos = str.find("=",pos+2);
        }else{
            pos = str.find("=",pos+1);
        }
    }

    return str;
}


bool isFakeVariable(string var){
    if( var.find("In_") != string::npos ) return true;
    if( !var.compare("ii") ) return true;
    if( !var.compare("jj") ) return true;

    return false;
}

bool isExpressionOrConst(string var){
    bool isNumber=true;
    if( var.find_first_of("+-*/") != string::npos ){
        return true;
    }

    for(int i=0; i<var.length(); ++i){
        if( !isdigit(var[i]) ){
            isNumber = false;
            break;
        }
    }

    if( isNumber )
        return true;

    // This is the upper bound of the parameter space, by convention
    if( !var.compare("BB") )
        return true;

    return false;
}

void mergeLists(void){
    bool found;
    list<dep_t>::iterator fd_itr; // flow dep iterator
    list<dep_t>::iterator od_itr; // out dep iterator
    list<dep_t>::iterator fd2_itr; // second flow dep iterator
    set<int> srcSet;
    set<int>::iterator src_itr;
    set<string> fake_it;

    fake_it.insert("BB");
    fake_it.insert("step");
    fake_it.insert("NT");
    fake_it.insert("ip");
    fake_it.insert("proot");
    fake_it.insert("P");
    fake_it.insert("B");

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
        list<dep_t> rlvnt_flow_deps;
        // Find all flow deps that flow from this source
        for(fd_itr=flow_deps.begin(); fd_itr != flow_deps.end(); ++fd_itr) {
            dep_t f_dep = static_cast<dep_t>(*fd_itr);
            int fd_srcLine = f_dep.srcLine;
            int fd_dstLine = f_dep.dstLine;
            if( fd_srcLine == source ){
                rlvnt_flow_deps.push_back(f_dep);
            }
        }

        // Iterate over all the relevant flow dependencies and apply the
        // output dependencies to them
        for(fd_itr=rlvnt_flow_deps.begin(); fd_itr != rlvnt_flow_deps.end(); ++fd_itr) {
            dep_t f_dep = static_cast<dep_t>(*fd_itr);
            int fd_srcLine = f_dep.srcLine;
            int fd_dstLine = f_dep.dstLine;
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

            // Subtract the dependencies that kill our current flow dependency, from our current
            // flow dependency and see what's left (if anything).
            string sets_for_omega = setIntersector.subtract();
            fstream filestr ("/tmp/oc_in.txt", fstream::out);
            filestr << sets_for_omega << endl;
            filestr.close();

            FILE *pfp = popen( (omegaHome+"/omega_calc/obj/oc /tmp/oc_in.txt").c_str(), "r");
            stringstream data;
            char buffer[256];
            while (!feof(pfp)){
                if (fgets(buffer, 256, pfp) != NULL){
                    data << buffer;
                }
            }
            pclose(pfp);
            // Read the dependency that comes out of Omega
            string line;
            while( getline(data, line) ){
                if( (line.find("#") == string::npos) && !line.empty() ){
                    break;
                }
            }

            // Find the task in the map where the current dep should go
            task_t task;
            map<string,task_t>::iterator it;
            it = taskMap.find(f_dep.source);
            if ( it == taskMap.end() ){
                cerr << "FATAL ERROR: Task \""<< f_dep.source <<"\" does not exist in the taskMap" << endl;
                exit(-1);
            }
            task = it->second;
            if( task.name.compare(f_dep.source) ){
                cerr << "FATAL ERROR: Task name in taskMap does not match task name in flow dependency: ";
                cerr << "\"" <<task.name << "\" != \"" << f_dep.source << "\"" << endl;
                exit(-1);
            }

            // format the dependency in JDF format and store it in the proper task
            if( processDep(f_dep, line, task, false) ){
                // Update the taskMap, since "task" is just a copy and not a pointer into the map.
                taskMap[task.name] = task;
            }

#if 0
            // Find the task in the map (if it exists) and add the new OUT dep to it's outDeps
            task_t task;
            map<string,task_t>::iterator it;
            it = taskMap.find(f_dep.source);
            if ( it == taskMap.end() ){
                cerr << "FATAL ERROR: Task \""<< f_dep.source <<"\" does not exist in the taskMap" << endl;
                exit(-1);
            }
            task = it->second;
            if( task.name.compare(f_dep.source) ){
                cerr << "FATAL ERROR: Task name in taskMap does not match task name in flow dependency: ";
                cerr << "\"" <<task.name << "\" != \"" << f_dep.source << "\"" << endl;
                exit(-1);
            }
            task.outDeps.push_back(outDep);
            taskMap[f_dep.source] = task;
#endif

            // If this new OUT dep does not go to the exit, invert it to get an IN dep
            // unless it was an impossible dependency (having "FALSE" in the conditions).
            if( f_dep.sink.find("OUT") == string::npos && line.find("FALSE") == string::npos){
 
                // If it was a real dependency, ask Omega to revert it
                string rev_set_for_omega = setIntersector.inverse(line);
                filestr.open("/tmp/oc_in.txt", fstream::out);
                filestr << rev_set_for_omega << endl;
                filestr.close();

                data.clear();
                pfp = popen( (omegaHome+"/omega_calc/obj/oc /tmp/oc_in.txt").c_str(), "r");
                while (!feof(pfp)){
                    if (fgets(buffer, 256, pfp) != NULL){
                        data << buffer;
                    }
                }
                pclose(pfp);
                // Read the reversed dependency
                while( getline(data, line) ){
                    if( (line.find("#") == string::npos) && !line.empty() ){
                        break;
                    }
                }

                // Find the task in the map where the current dep should go
                it = taskMap.find(f_dep.sink);
                if ( it == taskMap.end() ){
                    cerr << "FATAL ERROR: Task \""<< f_dep.sink <<"\" does not exist in the taskMap" << endl;
                    exit(-1);
                }
                task = it->second;
                if( task.name.compare(f_dep.sink) ){
                    cerr << "FATAL ERROR: Task name in taskMap does not match task name in flow dependency: ";
                    cerr << "\"" <<task.name << "\" != \"" << f_dep.sink << "\"" << endl;
                    exit(-1);
                }

                // format the reversed dependency in JDF format and add it to the task
                if( processDep(f_dep, line, task, true) ){
                    // Update the taskMap, since "task" is just a copy and not a pointer into the map.
                    taskMap[task.name] = task;
                }
            }
        }
    }

/*
typedef struct{
    string       localVar;
    string       type; 
    string       localAlias;
    list<string> remoteAliases;
    list<string> inEdges;
    list<string> outEdges;
} final_dep_t;
*/

    // Print all the tasks
    map<string,task_t>::iterator it=taskMap.begin();
    for ( ; it != taskMap.end(); it++ ){
        task_t task = (*it).second;

        // Do not print anything for tasks "IN()" and "OUT()"
        if( task.name.find("IN(") != string::npos || task.name.find("OUT(") != string::npos )
            continue;

        // Print the task name and its parameter space
        if( it != taskMap.begin() )
            cout << "\n\n";
        cout << "TASK: " << task.name << "{" << "\n";

        // Print the parameter space bounds
        list<string>::iterator ps_itr = task.paramSpace.begin();
        for(; ps_itr != task.paramSpace.end(); ++ps_itr)
            cout << "  " << *ps_itr << "\n";

        list<final_dep_t>::iterator fnld_itr;
        for(fnld_itr=task.deps.begin(); fnld_itr != task.deps.end(); ++fnld_itr) {
            final_dep_t fnl_dep = *fnld_itr;

            // Local Alias
            cout << fnl_dep.localAlias << endl;

            // Remote Aliases
            list<string>::iterator a_itr;
            for(a_itr=fnl_dep.remoteAliases.begin(); a_itr != fnl_dep.remoteAliases.end(); ++a_itr) {
                cout << *a_itr << endl;
            }

            // Type & Local Variable
            cout << fnl_dep.type << " " << fnl_dep.localVar << " ";

            // IN Edges (if any)
            list<string>::iterator e_itr;
            for(e_itr=fnl_dep.inEdges.begin(); e_itr != fnl_dep.inEdges.end(); ++e_itr) {
                if( e_itr!=fnl_dep.inEdges.begin() )
                    cout << "        ";
                cout << *e_itr << endl;
            }

            // OUT Edges (if any)
            e_itr;
            for(e_itr=fnl_dep.outEdges.begin(); e_itr != fnl_dep.outEdges.end(); ++e_itr) {
                if( e_itr!=fnl_dep.outEdges.begin() || !fnl_dep.inEdges.empty() )
                    cout << "        ";
                cout << *e_itr << endl;
            }

            // Newline
            cout << endl;

        }

#if 0
        // Print the IN dependencies
        list<string>::iterator id_itr = task.inDeps.begin();
        for(; id_itr != task.inDeps.end(); ++id_itr)
            cout << *id_itr << "\n";

        cout << "\n";

        // Print the OUT dependencies
        list<string>::iterator od_itr = task.outDeps.begin();
        for(; od_itr != task.outDeps.end(); ++od_itr)
            cout << *od_itr << "\n";
#endif

        cout << "}" << endl;
    }



    return;
}
