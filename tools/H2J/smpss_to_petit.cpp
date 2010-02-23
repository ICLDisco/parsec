#include <string>
#include <list>
#include <set>
#include <map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctype.h>
#include <boost/regex.hpp> 

using namespace std;

int var_counter=1;
void parse_SMPSS(std::ifstream &ifs);
set<string> arrays;

int main(int argc, char **argv){
    char *fName;

    if( argc < 2 ){
        cerr << "Usage: "<< argv[0] << " SMPSS_File" << endl;
        exit(-1);
    }

    fName = argv[1];
    ifstream ifs( fName );
    if( !ifs ){
        cerr << "File \""<< fName <<"\" does not exist" << endl;
        exit(-1);
    }

    parse_SMPSS(ifs);

    return 0;
}


class Task{
    private:
        map<string,string> vars;
        list<string> argument_types, actual_args;
// #pragma css task inout(U[200*200], L[200*200]) output(dL[200*40], IPIV[200])

        void parse_type(string type, string str){
            boost::regex re("\\w*(\\[[^]]*\\])*");
            boost::sregex_token_iterator i(str.begin(), str.end(), re, 0);
            boost::sregex_token_iterator j;
            while(i != j){
               string str = *i++;
               if( str.empty() ) continue;
               unsigned int pos = str.find("[");
               if( pos != string::npos )
                   str = str.substr(0,pos);
               vars[str] = type;
            }

        }

        void parse_pragma(string str){
            // skip over the beginning
            str = str.substr(str.find("task")+5);

            boost::regex re("(input|inout|output)\\([^)]*\\)");
            boost::sregex_token_iterator i(str.begin(), str.end(), re, 0);
            boost::sregex_token_iterator j;
            while(i != j){
               string str = *i++;
               string name = str.substr(0,str.find("("));
               str = str.substr( str.find("(")+1 );
               parse_type(name, str);
            }
        }

        void parse_func_decl(string str){
            // skip over the beginning
            str = str.substr(str.find("("));

            boost::regex re("(int|double) \\*\\w*");
            boost::sregex_token_iterator i(str.begin(), str.end(), re, 0);
            boost::sregex_token_iterator j;
            while(i != j){
               string str = *i++;
               unsigned int pos = str.find("*");
               if( pos == string::npos || str.empty() )
                   continue;
               str = str.substr(pos+1);
               argument_types.push_back(vars[str]);
            }
        }

        void parse_func_call(string str){
            // skip over the beginning
            str = str.substr(str.find("("));

            boost::regex re("\\w*\\((\\w|,| )*\\)");
            boost::sregex_token_iterator i(str.begin(), str.end(), re, 0);
            boost::sregex_token_iterator j;
            while(i != j){
               string str = *i++;
               if( str.empty() )
                   continue;
               actual_args.push_back(str);
               unsigned int pos = str.find("(");
               if( pos != string::npos )
                   arrays.insert(str.substr(0,pos));
            }
        }

        string convert_func_call(int offset){
            stringstream result;
            list<string> in_args, out_args;

            if( actual_args.size() != argument_types.size() ){
                cerr << "ERROR: Arguments have not been parsed correctly." << endl;
                return "";
            }

            while( !actual_args.empty() ){
                string arg = (string)actual_args.front();
                actual_args.pop_front();
                string type = (string)argument_types.front();
                argument_types.pop_front();

                if( !type.compare("input") ){
                    in_args.push_back(arg);
                }else if( !type.compare("output") ){
                    out_args.push_back(arg);
                }else if( !type.compare("inout") ){
                    in_args.push_back(arg);
                    out_args.push_back(arg);
                }
            }

            while( !in_args.empty() ){
                string arg = (string)in_args.front();
                in_args.pop_front();
                result << string(offset,' ') << "v" << var_counter << " = " << arg << endl;
                ++var_counter;
            }

            while( !out_args.empty() ){
                string arg = (string)out_args.front();
                out_args.pop_front();
                result << string(offset,' ') << arg << " = v" << var_counter << endl;
                ++var_counter;
            }

            return result.str();
        }

    public:
        string pragma;
        string func_decl;
        string func_to_assignments(string line, int offset){
            stringstream result;
            vars.clear();
            argument_types.clear();
            result << "!" << string(offset,' ') << pragma << endl;
            result << "!" << string(offset,' ') << func_decl << endl;
            result << "!!" << line << endl;

            parse_pragma(pragma);
            parse_func_decl(func_decl);
            parse_func_call(line);

            result << convert_func_call(offset);
            return result.str();

        }

};


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


void parse_SMPSS(ifstream &ifs){
    int i;
    stringstream result;
    string line, source;
    list<string> ind_vars;
    map<string,Task> task_info;
    Task task;

    i=0;
    // parse the pragmas first
    while( getline(ifs,line) ){
        if( line.empty() ){
            continue;
        }

        if( line.find("#pragma") != string::npos &&
            line.find("css") != string::npos &&
            line.find("start") != string::npos ){
            break;
        }

        if( line.find("#pragma") != string::npos &&
            line.find("css") != string::npos &&
            line.find("task") != string::npos ){
            task.pragma = line;
        }else{
            unsigned int pos_task = line.find("task_");
            if( pos_task != string::npos ){
                unsigned int pos_par = line.find("(");
                if( pos_par == string::npos ){
                    cerr << "Malformed line \""+line+"\"" << endl;
                    return;
                }
                string name = line.substr(pos_task,pos_par-pos_task);
                name = line.substr(pos_task,pos_par-pos_task);
                task.func_decl = line;
                task_info[name] = task;
            }
        }
    }

    // parse the loops next (for() HAS to use brackets "{","}")
    int offset = 0;
    while( getline(ifs,line) ){
        unsigned int pos_par, pos_eq;

        if( line.find("for") != string::npos ){
            offset += 4;
            pos_par = line.find("(");
            pos_eq = line.find("=");
            string var = line.substr(pos_par, pos_eq-pos_par);
            ind_vars.push_back(trimAll(var));
            result << line << endl;
        }
        if( line.find("}") != string::npos ){
            offset -= 4;
            ind_vars.pop_back();
            result << line << endl;
        }
        if( line.find("task_") != string::npos ){
            pos_par = line.find("(");
            string name = trimAll(line.substr(0,pos_par));
            Task t = (Task)task_info[name];
            result << t.func_to_assignments(line, offset) << endl;
        }
    }

    cout << "real ";
    for(int i=1; i<var_counter; ++i){
        if( i>1 )
            cout << ", ";
        cout << "v" <<i;
    }
    set<string>::iterator ar_itr;
    for(ar_itr = arrays.begin(); ar_itr!= arrays.end(); ++ar_itr){
        cout << ", " << (string)*ar_itr << "[200][200]";
    }
    cout << endl << result.str();
}
