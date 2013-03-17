%{
/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "jdf.h"
#include "node_struct.h"
#include "utility.h"
#include "omega_interface.h"
#include "symtab.h"
#include "parse_utility.h"


extern int yyget_lineno();
void yyerror(const char *s);
extern int yylex (void);
extern int _q2j_add_phony_tasks;
extern node_t *_q2j_func_list_head;
extern char *q2j_input_file_name;

type_list_t *type_hash[HASH_TAB_SIZE] = {0};

%}

%union {
 node_t node;
 type_node_t type_node;
 char *string;
}


%type <node> ADD_ASSIGN
%type <node> AND_ASSIGN
%type <node> BREAK
%type <node> CASE
%type <node> CONTINUE
%type <node> DEC_OP
%type <node> DEFAULT
%type <node> DIV_ASSIGN
%type <node> DO
%type <node> ELLIPSIS
%type <node> ELSE
%type <node> EQ_OP
%type <node> FLOATCONSTANT
%type <node> FOR
%type <node> GE_OP
%type <node> GOTO
%type <node> IDENTIFIER
%type <node> IF
%type <node> INC_OP
%type <node> INTCONSTANT
%type <node> BIN_MASK
%type <node> L_AND
%type <node> L_OR
%type <node> LEFT_ASSIGN
%type <node> LEFT_OP
%type <node> LE_OP
%type <node> MOD_ASSIGN
%type <node> MUL_ASSIGN
%type <node> NE_OP
%type <node> OR_ASSIGN
%type <node> PTR_OP
%type <node> RETURN
%type <node> RIGHT_ASSIGN
%type <node> RIGHT_OP
%type <node> STRING_LITERAL
%type <node> SUB_ASSIGN
%type <node> SWITCH
%type <node> WHILE
%type <node> XOR_ASSIGN
%type <node> additive_expression
%type <node> and_expression
%type <node> argument_expression_list
%type <node> assignment_expression
%type <node> assignment_operator
%type <node> cast_expression
%type <node> compound_statement
%type <node> conditional_expression
%type <node> constant_expression
%type <node> enumerator
%type <node> enumerator_list
%type <node> equality_expression
%type <node> exclusive_or_expression
%type <node> expression
%type <node> expression_statement
%type <node> external_declaration
%type <node> function_definition
%type <node> inclusive_or_expression
%type <node> iteration_statement
%type <node> jump_statement
%type <node> labeled_statement
%type <node> logical_and_expression
%type <node> logical_or_expression
%type <node> multiplicative_expression
%type <node> postfix_expression
%type <node> primary_expression
%type <node> relational_expression
%type <node> selection_statement
%type <node> shift_expression
%type <node> statement
%type <node> statement_list
%type <node> translation_unit
%type <node> unary_expression
%type <node> unary_operator

%type <node> init_declarator
%type <node> init_declarator_list
%type <node> declarator
%type <node> direct_declarator
%type <node> declaration
%type <node> declaration_list
%type <node> initializer
%type <node> pragma_parameters
%type <node> pragma_specifier
//%type <node> pragma_options
//%type <node> task_arguments

%type <string> abstract_declarator
%type <type_node> parameter_declaration

%type <string> identifier_list
%type <string> initializer_list

%type <string> AUTO
%type <string> CHAR
%type <string> CONST
%type <string> DOUBLE
%type <string> ENUM
%type <string> EXTERN
%type <string> FLOAT
%type <string> INT
%type <string> LONG

%type <string> INT8
%type <string> INT16
%type <string> INT32
%type <string> INT64
%type <string> UINT8
%type <string> UINT16
%type <string> UINT32
%type <string> UINT64
%type <string> INTPTR
%type <string> UINTPTR
%type <string> INTMAX
%type <string> UINTMAX

%type <string> REGISTER
%type <string> SHORT
%type <string> SIGNED
%type <string> SIZEOF
%type <string> STATIC
%type <string> STRUCT
%type <string> TYPEDEF
%type <string> PRAGMA
%type <string> DIR_PARSEC_DATA_COLOCATED
%type <string> DIR_PARSEC_INVARIANT
%type <string> DIR_PARSEC_TASK_START
%type <string> TYPE_NAME
%type <string> UNION
%type <string> UNSIGNED
%type <string> VOID
%type <string> VOLATILE
%type <string> PLASMA_COMPLEX32_T
%type <string> PLASMA_COMPLEX64_T
%type <string> PLASMA_ENUM
%type <string> PLASMA_REQUEST
%type <string> PLASMA_DESC
%type <string> PLASMA_SEQUENCE

%type <node> parameter_type_list
%type <node> parameter_list

%type <string> declaration_specifiers
%type <string> typedef_specifier
%type <string> direct_abstract_declarator
%type <string> enum_specifier
%type <string> pointer
%type <string> specifier_qualifier_list
%type <string> storage_class_specifier
%type <string> struct_declaration
%type <string> struct_declaration_list
%type <string> struct_declarator
%type <string> struct_declarator_list
%type <string> struct_or_union
%type <string> struct_or_union_specifier
%type <string> type_qualifier
%type <string> type_qualifier_list
%type <string> type_specifier
%type <string> type_name


%token IDENTIFIER INTCONSTANT BIN_MASK FLOATCONSTANT STRING_LITERAL SIZEOF
%token PTR_OP INC_OP DEC_OP LEFT_OP RIGHT_OP LE_OP GE_OP EQ_OP NE_OP
%token L_AND L_OR MUL_ASSIGN DIV_ASSIGN MOD_ASSIGN ADD_ASSIGN
%token SUB_ASSIGN LEFT_ASSIGN RIGHT_ASSIGN AND_ASSIGN
%token XOR_ASSIGN OR_ASSIGN TYPE_NAME

%token TYPEDEF PRAGMA EXTERN STATIC AUTO REGISTER
%token DIR_PARSEC_DATA_COLOCATED DIR_PARSEC_INVARIANT DIR_PARSEC_TASK_START
%token CHAR SHORT INT LONG SIGNED UNSIGNED FLOAT DOUBLE CONST VOLATILE VOID
%token INT8 INT16 INT32 INT64 UINT8 UINT16 UINT32 UINT64 INTPTR UINTPTR INTMAX UINTMAX
%token PLASMA_COMPLEX32_T PLASMA_COMPLEX64_T PLASMA_ENUM PLASMA_REQUEST PLASMA_DESC PLASMA_SEQUENCE

%token STRUCT UNION ENUM ELLIPSIS

%token CASE DEFAULT IF ELSE SWITCH WHILE DO FOR GOTO CONTINUE BREAK RETURN

%start translation_unit
%%

primary_expression
	: IDENTIFIER
          { 
//              char *name=$1.u.var_name;
//              char *var_type = st_type_of_variable(name, $1.symtab);
//              if( NULL == var_type ){
//                  printf("No entry for \"%s\" in symbol table\n",name);
//              }else{
//                  printf("\"%s\" is of type \"%s\"\n",name, var_type);
//              }
          } 
	| INTCONSTANT
	| FLOATCONSTANT
	| STRING_LITERAL
	| '(' expression ')' {$$ = $2;}
	;

postfix_expression
	: primary_expression
          { 
          $$ = $1;
          }
	| postfix_expression '[' expression ']'
          {

              if( ARRAY == $1.type ){
                  int count;
                  $$ = $1;
                  count = ++($$.u.kids.kid_count);
                  $$.u.kids.kids = (node_t **)realloc( $$.u.kids.kids, count*sizeof(node_t *) );
                  $$.u.kids.kids[count-1] = node_to_ptr($3);
              }else{
                  $$.type = ARRAY;
                  $$.u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
                  $$.u.kids.kid_count = 2;
                  $$.u.kids.kids[0] = node_to_ptr($1);
                  $$.u.kids.kids[1] = node_to_ptr($3);
              }

          }
	| postfix_expression '(' ')'
          {
              $$.type = FCALL;
              $$.lineno = yyget_lineno();

              $$.u.kids.kids = (node_t **)calloc(1, sizeof(node_t *));
              $$.u.kids.kid_count = 1;
              $$.u.kids.kids[0] = node_to_ptr($1);
          }
	| postfix_expression '(' argument_expression_list ')'
          {
              node_t *tmp, *flwr;
              int i, count = 0;

              $$.type = FCALL;
              $$.lineno = yyget_lineno();

              for(tmp=$3.next; NULL != tmp ; flwr=tmp, tmp=tmp->prev){
                  count++;
              }
              $$.u.kids.kids = (node_t **)calloc(count+1, sizeof(node_t *));
              $$.u.kids.kid_count = count+1;
              $$.u.kids.kids[0] = node_to_ptr($1);

              /* Unchain the temporary list of arguments and make them the */
              /* kids of this FCALL */
              for(i=1; i<count+1; ++i){
                  assert(flwr != NULL);
                  $$.u.kids.kids[i] = flwr;
                  flwr = flwr->next;
              }
          }
	| postfix_expression '.' IDENTIFIER
          {
              $$.type = S_U_MEMBER;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| postfix_expression PTR_OP IDENTIFIER
          {
              $$.type = PTR_OP;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| postfix_expression INC_OP
          {
              $$.type = EXPR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($2);
          }
	| postfix_expression DEC_OP
          {
              $$.type = EXPR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($2);
          }
	;

/*
 * Create a fake chain (mimicking the behavior of statement_list) that we will
 * then take apart and put all the elements as kids of an FCALL in the
 * postfix_expression rule.
 */
argument_expression_list
	: assignment_expression
          { 
              node_t *tmp;
              tmp = node_to_ptr($1);
              tmp->prev = NULL;
              tmp->next = NULL;
              $$.next = tmp;
          } 
	| argument_expression_list ',' assignment_expression
          { 
              node_t *tmp;
              tmp = node_to_ptr($3);
              tmp->next = NULL;
              tmp->prev = $1.next;
              tmp->prev->next = tmp;
              $$.next = tmp;
          } 
	;


unary_expression
	: postfix_expression { $$ = $1; }
	| INC_OP unary_expression
          {
              $$.type = EXPR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($2);
          }
	| DEC_OP unary_expression
          {
              $$.type = EXPR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($2);
          }
	| unary_operator cast_expression
          {
              $$.type = EXPR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($2);
          }
	| SIZEOF unary_expression
          {
              $$.type = SIZEOF;
              $$.u.kids.kid_count = 1;
              $$.u.kids.kids = (node_t **)calloc(1,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($2);
          }
	| SIZEOF '(' type_name ')'
          {
              $$.type = SIZEOF;
              $$.u.kids.kid_count = 0;
              $$.u.var_name = strdup($3);
          }
	;

unary_operator
	: '&' {$$.type = ADDR_OF;}
	| '*' {$$.type = STAR;}
	| '+' {$$.type = PLUS;}
	| '-' {$$.type = MINUS;}
	| '~' {$$.type = TILDA;}
	| '!' {$$.type = BANG;}
	;

cast_expression
	: unary_expression 
	| '(' type_name ')' cast_expression
          {
            $$ = $4;
            $$.var_type = strdup($2);
          }
	;

multiplicative_expression
	: cast_expression
	| multiplicative_expression '*' cast_expression
          {
              $$.type = MUL;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| multiplicative_expression '/' cast_expression
          {
              $$.type = DIV;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| multiplicative_expression '%' cast_expression
          {
              $$.type = MOD;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

additive_expression
	: multiplicative_expression
	| additive_expression '+' multiplicative_expression
          {
              $$.type = ADD;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| additive_expression '-' multiplicative_expression
          {
              $$.type = SUB;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

shift_expression
	: additive_expression
	| shift_expression LEFT_OP additive_expression
          {
              $$.type = LSHIFT;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| shift_expression RIGHT_OP additive_expression
          {
              $$.type = RSHIFT;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

relational_expression
	: shift_expression
	| relational_expression '<' shift_expression
          {
              $$.type = LT;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| relational_expression '>' shift_expression
          {
              $$.type = GT;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| relational_expression LE_OP shift_expression
          {
              $$.type = LE;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| relational_expression GE_OP shift_expression
          {
              $$.type = GE;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

equality_expression
	: relational_expression
	| equality_expression EQ_OP relational_expression
          {
              $$.type = EQ_OP;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	| equality_expression NE_OP relational_expression
          {
              $$.type = NE_OP;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

and_expression
	: equality_expression
	| and_expression '&' equality_expression
          {
              $$.type = B_AND;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

exclusive_or_expression
	: and_expression
	| exclusive_or_expression '^' and_expression
          {
              $$.type = B_XOR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

inclusive_or_expression
	: exclusive_or_expression
	| inclusive_or_expression '|' exclusive_or_expression
          {
              $$.type = B_OR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

logical_and_expression
	: inclusive_or_expression
	| logical_and_expression L_AND inclusive_or_expression
          {
              $$.type = $2.type;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

logical_or_expression
	: logical_and_expression
	| logical_or_expression L_OR logical_and_expression
          {
              $$.type = $2.type;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

conditional_expression
	: logical_or_expression { $$ = $1; }
	| logical_or_expression '?' expression ':' conditional_expression
          {
            $$.type = COND;
            $$.u.kids.kid_count = 3;
            $$.u.kids.kids = (node_t **)calloc(3, sizeof(node_t *));
            $$.u.kids.kids[0] = node_to_ptr($1);
            $$.u.kids.kids[1] = node_to_ptr($3);
            $$.u.kids.kids[2] = node_to_ptr($5);
          }
	;

assignment_expression
	: conditional_expression
          { 
            $$ = $1;
          }
	| unary_expression assignment_operator assignment_expression 
          {
            $$.type = $2.type;
            $$.u.kids.kid_count = 2;
            $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
            $$.u.kids.kids[0] = node_to_ptr($1);
            $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

assignment_operator
	: '=' {$$.type = ASSIGN; }
	|MUL_ASSIGN {$$.type = MUL_ASSIGN;}
	|DIV_ASSIGN {$$.type = DIV_ASSIGN;}
	|MOD_ASSIGN {$$.type = MOD_ASSIGN;}
	|ADD_ASSIGN {$$.type = ADD_ASSIGN;}
	|SUB_ASSIGN {$$.type = SUB_ASSIGN;}
	|LEFT_ASSIGN {$$.type = LEFT_ASSIGN;}
	|RIGHT_ASSIGN {$$.type = RIGHT_ASSIGN;}
	|AND_ASSIGN {$$.type = AND_ASSIGN;}
	|XOR_ASSIGN {$$.type = XOR_ASSIGN;}
	|OR_ASSIGN {$$.type = OR_ASSIGN;}
	;

expression
	: assignment_expression
	| expression ',' assignment_expression
          {
              $$.type = COMMA_EXPR;
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              $$.u.kids.kids[0] = node_to_ptr($1);
              $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

constant_expression
	: conditional_expression
	  {
	      $$ = $1;
	  }
	;

declaration
	: declaration_specifiers ';'
          {
//              fprintf(stderr,"DEBUG: Is this correct C?: \"%s;\"\n",(char *)$1);
          }
	| declaration_specifiers init_declarator_list ';'
          {

              node_t *tmp;
              // rewind the pointer to the beginning of the list
              for(tmp=$2.next; NULL != tmp->prev ; tmp=tmp->prev);
              // traverse the list
              for(; NULL != tmp ; tmp=tmp->next){
                  node_t *variable = tmp;
                  if( ASSIGN == tmp->type )
                      variable = tmp->u.kids.kids[0];

                  if( IDENTIFIER == variable->type ){
                      st_insert_new_variable(variable->u.var_name, (char *)$1);
#if 0 // debug
                      printf("st_insert(%s, %s)\n",variable->u.var_name, (char *)$1);
#endif
                  }

              }
#if 0 // debug
              printf("%s ",(char *)$1);
              // rewind the pointer to the beginning of the list
              for(tmp=$2.next; NULL != tmp->prev; tmp=tmp->prev);
              // traverse the list
              for(; NULL != tmp; tmp=tmp->next){
                  if(NULL != tmp->prev){
                      printf(", ");
                  }
                  printf("%s",tree_to_str(tmp));
              }
              printf("\n");
#endif // debug
              $$ = $2;
          }
	;

typedef_specifier
	: TYPEDEF declaration_specifiers IDENTIFIER ';'
	  {
              add_type(tree_to_str(&($3)), $2);
	  }
	;

/*
pragma_options
	: IDENTIFIER { }
	| IDENTIFIER ':' pragma_options { }
	;
*/

/*
task_arguments
	: pragma_options { }
	| pragma_options ',' task_arguments { }
	;
*/

pragma_parameters
	: IDENTIFIER
      { 
          node_t *tmp;
          tmp = node_to_ptr($1);
          tmp->prev = NULL;
          tmp->next = NULL;
          $$.next = tmp;
      } 
	| IDENTIFIER pragma_parameters
	  {
          node_t *tmp;
          tmp = node_to_ptr($1);
          tmp->next = NULL;
          tmp->prev = $2.next;
          tmp->prev->next = tmp;
          $$.next = tmp;
	  }
	;

pragma_specifier
	: PRAGMA IDENTIFIER pragma_parameters 
      {
          /* #pragma ztstrf U A L IPIV */
          node_t *tmp;
          int i=0;
          add_variable_naming_convention($2.u.var_name, $3.next);
      }
	| PRAGMA DIR_PARSEC_INVARIANT expression
	  {
	      add_pending_invariant(node_to_ptr($3));
	  }
	| PRAGMA DIR_PARSEC_DATA_COLOCATED pragma_parameters
	  {
              //#pragma PARSEC_DATA_COLOCATED T A
              //int i=0;
              node_t *tmp, *reference;

              // find the reference matrix in the pragma
              for(reference=$3.next; NULL != reference->prev; reference=reference->prev)
                  /* nothing */ ;

              // traverse the list backwards
              //printf("(");
              for(tmp=$3.next; NULL != tmp->prev; tmp=tmp->prev){
                  //if(i++)
                  //    printf(" and ");
                  //printf("%s",tmp->u.var_name);
                  add_colocated_data_info(tmp->u.var_name, reference->u.var_name);
              }
              // add a tautologic relation from the reference element to itself
              add_colocated_data_info(reference->u.var_name, reference->u.var_name);
              //printf(") is co-located with %s\n",tmp->u.var_name);
	  }
/*
	| PRAGMA DIR_PARSEC_TASK_START IDENTIFIER task_arguments
	  {
              //#pragma PARSEC_TASK_START  TASK_NAME  PARAM[:PSEUDONAME]:(IN|OUT|INOUT|SCRATCH)[:TYPE_NAME] [, ...]
	  }
*/
	;

declaration_specifiers
	: storage_class_specifier
	| storage_class_specifier declaration_specifiers
          {
              char *str = strdup($1);
              $$ = append_to_string(str, $2, " %s", 1+strlen($2) );
          }
	| type_specifier
          {
              $$ = $1;
          }
	| type_specifier declaration_specifiers
          {
              char *str = strdup($1);
              $$ = append_to_string(str, $2, " %s", 1+strlen($2) );
          }
	| type_qualifier
          {
              $$ = $1;
          }
	| type_qualifier declaration_specifiers
          {
              char *str = strdup($1);
              $$ = append_to_string(str, $2, " %s", 1+strlen($2) );
          }
	;

init_declarator_list
	: init_declarator
          { 
              node_t *tmp;
              tmp = node_to_ptr($1);
              tmp->prev = NULL;
              tmp->next = NULL;
              $$.next = tmp;
          } 
	| init_declarator_list ',' init_declarator
          {
              node_t *tmp;
              tmp = node_to_ptr($3);
              tmp->next = NULL;
              tmp->prev = $1.next;
              tmp->prev->next = tmp;
              $$.next = tmp;
          }
	;

init_declarator
	: declarator
	| declarator '=' initializer
          {
            $$.type = ASSIGN;
            $$.u.kids.kid_count = 2;
            $$.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
            $$.u.kids.kids[0] = node_to_ptr($1);
            $$.u.kids.kids[1] = node_to_ptr($3);
          }
	;

storage_class_specifier
	: EXTERN
	| STATIC
	| AUTO
	| REGISTER
	;

type_specifier
	: VOID
    | CHAR
    | SHORT
    | INT
    | LONG
    | INT8
    | INT16
    | INT32
    | INT64
    | UINT8
    | UINT16
    | UINT32
    | UINT64
    | INTPTR
    | UINTPTR
    | INTMAX
    | UINTMAX
    | FLOAT
    | DOUBLE
    | SIGNED
    | UNSIGNED
    | struct_or_union_specifier
    | enum_specifier
    | PLASMA_COMPLEX32_T
    | PLASMA_COMPLEX64_T
    | PLASMA_ENUM
    | PLASMA_REQUEST
    | PLASMA_DESC
    | PLASMA_SEQUENCE
    | TYPE_NAME
    
	;

struct_or_union_specifier
	: struct_or_union IDENTIFIER '{' struct_declaration_list '}' {}
	| struct_or_union '{' struct_declaration_list '}' {}
	| struct_or_union IDENTIFIER {}
	;

struct_or_union
	: STRUCT
	| UNION
	;

struct_declaration_list
	: struct_declaration
	| struct_declaration_list struct_declaration
	;

struct_declaration
	: specifier_qualifier_list struct_declarator_list ';'
	;

specifier_qualifier_list
	: type_specifier specifier_qualifier_list
          {
              char *str = strdup($1);
              $$ = append_to_string(str, $2, " %s", 1+strlen($2) );
          }
	| type_specifier
	| type_qualifier specifier_qualifier_list
          {
              char *str = strdup($1);
              $$ = append_to_string(str, $2, " %s", 1+strlen($2) );
          }
	| type_qualifier
	;

struct_declarator_list
	: struct_declarator
	| struct_declarator_list ',' struct_declarator
	;

struct_declarator
	: declarator {}
/*	| ':' constant_expression */
/*	| declarator ':' constant_expression */
	;

enum_specifier
	: ENUM '{' enumerator_list '}'
	| ENUM IDENTIFIER '{' enumerator_list '}'
	| ENUM IDENTIFIER
	;

enumerator_list
	: enumerator
	| enumerator_list ',' enumerator
	;

enumerator
	: IDENTIFIER
	| IDENTIFIER '=' constant_expression
	;

type_qualifier
	: CONST
	| VOLATILE
	;

declarator
	: pointer direct_declarator
      {
          // TODO: Maybe make changes in the symbol table?
          $$ = $2;
      }
	| direct_declarator
	;

direct_declarator
	: IDENTIFIER
	| '(' declarator ')'
          {
              $$ = $2;
          }
	| direct_declarator '[' constant_expression ']'
          {
              if( ARRAY == $1.type ){
                  int count;
                  $$ = $1;
                  count = ++($$.u.kids.kid_count);
                  $$.u.kids.kids = (node_t **)realloc( $$.u.kids.kids, count*sizeof(node_t *) );
                  $$.u.kids.kids[count-1] = node_to_ptr($3);
              }else{
                  $$.type = ARRAY;
                  $$.u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
                  $$.u.kids.kid_count = 2;
                  $$.u.kids.kids[0] = node_to_ptr($1);
                  $$.u.kids.kids[1] = node_to_ptr($3);
              }
          }
	| direct_declarator '[' ']'
          {
              if( ARRAY == $1.type ){
                  int count;
                  $$ = $1;
                  count = ++($$.u.kids.kid_count);
                  $$.u.kids.kids = (node_t **)realloc( $$.u.kids.kids, count*sizeof(node_t *) );
                  node_t tmp;
                  tmp.type=EMPTY;
                  $$.u.kids.kids[count-1] = node_to_ptr(tmp);
              }else{
                  $$.type = ARRAY;
                  $$.u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
                  $$.u.kids.kid_count = 2;
                  $$.u.kids.kids[0] = node_to_ptr($1);
                  node_t tmp;
                  tmp.type=EMPTY;
                  $$.u.kids.kids[1] = node_to_ptr(tmp);
              }
          }
	| direct_declarator '(' parameter_type_list ')'
          {
              /**
               ** This is the rule for the declaration of the body of a function
               **/
              node_t *ptr, *head;

              for(head=$3.next; NULL != head && NULL != head->prev; head=head->prev)
                  /* just walk to the beginning */;

              /* Type "TMP" should not survive in the final tree */
              ptr = DA_create_B_expr(TMP, node_to_ptr($1), head);
              $$ = *ptr;
              $$.symtab = st_get_current_st();
          }
	| direct_declarator '(' identifier_list ')'
	| direct_declarator '(' ')'
	;

pointer
	: '*' {}
	| '*' type_qualifier_list {}
	| '*' pointer {}
	| '*' type_qualifier_list pointer {}
	;

type_qualifier_list
	: type_qualifier
	| type_qualifier_list type_qualifier
	;


parameter_type_list
	: parameter_list
	| parameter_list ',' ELLIPSIS
	;

parameter_list
	: parameter_declaration
          {
              node_t *tmp;
              (void)st_enter_new_scope();
              st_insert_new_variable(tree_to_str($1.var), $1.type);

              $1.var->symtab = st_get_current_st();
              tmp = $1.var;
              tmp->prev = NULL;
              tmp->next = NULL;
              $$.next = tmp;
          }
	| parameter_list ',' parameter_declaration
          {
              node_t *tmp;
              st_insert_new_variable(tree_to_str($3.var), $3.type);
              tmp = $3.var;
              tmp->next = NULL;
              tmp->prev = $1.next;
              tmp->prev->next = tmp;
              $$.next = tmp;
          }
	;


parameter_declaration
	: declaration_specifiers declarator
          {
              $$.type = strdup($1);
              $$.var = node_to_ptr($2);
          }
	| declaration_specifiers abstract_declarator
          {
              char *str = strdup($1);
              str = append_to_string(str, $2, " %s", 1+strlen($2) );
              printf("WARNING: the following parameter declaration is not inserted into the symbol table:\n%s\n",str);
          }
	| declaration_specifiers { }
	;

identifier_list
	: IDENTIFIER { $$ = $1.u.var_name; }
	| identifier_list ',' IDENTIFIER
          {
              char *str = strdup($1);
              str = append_to_string(str, ", ", NULL, 0 );
              $$ = append_to_string(str, $3.u.var_name, NULL, 0 );
          }
	;

type_name
	: specifier_qualifier_list
	| specifier_qualifier_list abstract_declarator
	;

abstract_declarator
	: pointer
	| direct_abstract_declarator
	| pointer direct_abstract_declarator
	;

direct_abstract_declarator
	: '(' abstract_declarator ')' {}
	| '[' ']' {}
	| '[' constant_expression ']' {}
	| direct_abstract_declarator '[' ']' {}
	| direct_abstract_declarator '[' constant_expression ']' {}
	| '(' ')' {}
	| '(' parameter_type_list ')' {}
	| direct_abstract_declarator '(' ')' {}
	| direct_abstract_declarator '(' parameter_type_list ')' {}
	;

initializer
	: assignment_expression
	| '{' initializer_list '}' { } 
	| '{' initializer_list ',' '}' {}
	;

initializer_list
	: initializer {}
	| initializer_list ',' initializer {}
	;

statement
	: labeled_statement
	| compound_statement
	| expression_statement 
	| selection_statement
	| iteration_statement
	| jump_statement
/*
 If we are to support pragma directives inside the body of a function
 we have to create a pragma scope type of hierarchy.
    | pragma_specifier 
*/
	;

labeled_statement
	: IDENTIFIER ':' statement
	| CASE constant_expression ':' statement
	| DEFAULT ':' statement
	;

compound_statement
	: '{' '}'
          { 
              $$.type = BLOCK;
              $$.u.block.first = NULL;
              $$.u.block.last = NULL;
          }
	| '{' statement_list '}'
          {
              node_t *tmp;

              $$.type = BLOCK;
              $$.u.block.last = $2.next;
              for(tmp=$2.next->prev; NULL != tmp && NULL != tmp->prev; tmp=tmp->prev) ;
              if( NULL == tmp )
                  $$.u.block.first = $$.u.block.last;
              else
                  $$.u.block.first = tmp;
          }
	| '{' declaration_list '}'
          {
              node_t *tmp;

              $$.type = BLOCK;
              $$.u.block.last = $2.next;
              for(tmp=$2.next->prev; NULL != tmp && NULL != tmp->prev; tmp=tmp->prev) ;
              if( NULL == tmp )
                  $$.u.block.first = $$.u.block.last;
              else
                  $$.u.block.first = tmp;
          }
	| '{' declaration_list statement_list '}'
          {
              // We've got two separate lists and we need to chain them together.
              node_t *tmp;

              $$.type = BLOCK;
              // take as last the last of the second list
              $$.u.block.last = $3.next;

              // then walk back the first list to find the beginning.
              for(tmp=$2.next->prev; NULL != tmp && NULL != tmp->prev; tmp=tmp->prev) ;

              if( NULL == tmp )
                  $$.u.block.first = $2.next;
              else
                  $$.u.block.first = tmp;

              // then walk back the second list to find its beginning.
              for(tmp=$3.next; NULL != tmp && NULL != tmp->prev; tmp=tmp->prev) ;

              // Now connect the end of the first list to the beginning of the second.
              $2.next->next = tmp;
              if( NULL != tmp )
                  tmp->prev = $2.next;

//for(; NULL != tmp; tmp=tmp->next)
//    dump_tree(*tmp,0);
          }
	;

declaration_list
	: declaration
          { 
              node_t *tmp;
              tmp = node_to_ptr($1);
              tmp->prev = NULL;
              tmp->next = NULL;
              $$.next = tmp;
          } 
	| declaration_list declaration
          { 
              node_t *tmp;
              tmp = node_to_ptr($2);
              tmp->next = NULL;
              tmp->prev = $1.next;
              tmp->prev->next = tmp;
              $$.next = tmp;
          } 
	;

statement_list
	: statement 
          { 
              node_t *tmp;
              tmp = node_to_ptr($1);
              tmp->prev = NULL;
              tmp->next = NULL;
              $$.next = tmp;
          } 
	;
	| statement_list statement
          { 
              node_t *tmp;
              tmp = node_to_ptr($2);
              tmp->next = NULL;
              tmp->prev = $1.next;
              tmp->prev->next = tmp;
              $$.next = tmp;
          } 
	;

expression_statement
	: ';' { $$.type = EMPTY; }
	| expression ';' {$$=$1;}
	;

selection_statement
	: IF '(' expression ')' statement
          {
              $$.type = IF;
              $$.u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids[0] = node_to_ptr($3);
              $$.u.kids.kids[1] = node_to_ptr($5);
          }
	| IF '(' expression ')' statement ELSE statement
          {
              $$.type = IF;
              $$.u.kids.kids = (node_t **)calloc(3, sizeof(node_t *));
              $$.u.kids.kid_count = 3;
              $$.u.kids.kids[0] = node_to_ptr($3);
              $$.u.kids.kids[1] = node_to_ptr($5);
              $$.u.kids.kids[2] = node_to_ptr($7);
          }
	| SWITCH '(' expression ')' statement
	;

iteration_statement
	: WHILE '(' expression ')' statement
          {
              $$.type = WHILE;
              $$.u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids[0] = node_to_ptr($3);
              $$.u.kids.kids[1] = node_to_ptr($5);
              $$.trip_count = -1;
              $$.loop_depth = -1;
          }
	| DO statement WHILE '(' expression ')' ';'
          {
              $$.type = DO;
              $$.u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
              $$.u.kids.kid_count = 2;
              $$.u.kids.kids[0] = node_to_ptr($5);
              $$.u.kids.kids[1] = node_to_ptr($3);
              $$.trip_count = -1;
              $$.loop_depth = -1;
          }
	| FOR '(' expression_statement expression_statement ')' statement
          {
              $$.type = FOR;
              $$.u.kids.kids = (node_t **)calloc(4, sizeof(node_t *));
              $$.u.kids.kid_count = 4;

              $$.u.kids.kids[0] = node_to_ptr($3);
              $$.u.kids.kids[1] = node_to_ptr($4);
              node_t tmp;
              tmp.type=EMPTY;
              $$.u.kids.kids[2]  = node_to_ptr(tmp);
              $$.u.kids.kids[3]  = node_to_ptr($6);
              DA_canonicalize_for(&($$));

              $$.trip_count = -1;
              $$.loop_depth = -1;
          }
      
	| FOR '(' expression_statement expression_statement expression ')' statement
          {
              $$.type = FOR;

              $$.u.kids.kids = (node_t **)calloc(4, sizeof(node_t *));
              $$.u.kids.kid_count = 4;
              $$.u.kids.kids[0] = node_to_ptr($3);
              $$.u.kids.kids[1] = node_to_ptr($4);
              $$.u.kids.kids[2] = node_to_ptr($5);
              $$.u.kids.kids[3] = node_to_ptr($7);
              DA_canonicalize_for(&($$));

              $$.trip_count = -1;
              $$.loop_depth = -1;
          }
	;

jump_statement
	: GOTO IDENTIFIER ';'
	| CONTINUE ';'
	| BREAK ';'
	| RETURN ';'
	| RETURN expression ';'
	;

translation_unit
	: external_declaration
	| translation_unit external_declaration
	;

external_declaration
	: function_definition
      {
          node_t *tmp;
          (void)st_exit_scope();
          associate_pending_pragmas_with_function(&($1));
          if( NULL == _q2j_func_list_head ){
              _q2j_func_list_head = node_to_ptr($1);
          }else{
              for(tmp=_q2j_func_list_head; NULL != tmp->next; tmp = tmp->next){
                  /* just walk down the list */;
              }
              tmp->next = node_to_ptr($1);
          }
      }
	| declaration
      {
          // Here is where the global scope variables were declared
          static node_t tmp;
          tmp.type=EMPTY;
          $$=tmp;
      }
    | typedef_specifier
      {
          /* do nothing, we added the type when we reduced the typedef */
      }
	| pragma_specifier
      {
          /* do nothing, we processed the pragma when we reduced it to pragma_specifier */
      }
	;

function_definition
	: declaration_specifiers declarator declaration_list compound_statement
      {
          /* This is the K&R style function declaration */
          node_t *ptr;
          ptr = DA_create_Func(DA_kid(&($2),0), DA_kid(&($2),1), node_to_ptr($4));
          ptr->symtab = $4.symtab;
          DA_parentize(ptr);
          $$ = *ptr;
      }
	| declaration_specifiers declarator compound_statement
      {
          node_t *ptr;
          ptr = DA_create_Func(DA_kid(&($2),0), DA_kid(&($2),1), node_to_ptr($3));
          ptr->symtab = $3.symtab;
          DA_parentize(ptr);
          $$ = *ptr;
      }
	| declarator declaration_list compound_statement
      {
          /* This is the K&R style function declaration, for a function with an undefined type */
          node_t *ptr;
          ptr = DA_create_Func(DA_kid(&($1),0), DA_kid(&($1),1), node_to_ptr($3));
          ptr->symtab = $3.symtab;
          DA_parentize(ptr);
          $$ = *ptr;
      }
	| declarator compound_statement
      {
          /* This is a function declaration, for a function with an undefined type */
          node_t *ptr;
          ptr = DA_create_Func(DA_kid(&($1),0), DA_kid(&($1),1), node_to_ptr($2));
          ptr->symtab = $2.symtab;
          DA_parentize(ptr);
          $$ = *ptr;
      }
	;

%%

extern char yytext[];
extern int column;

void yyerror(const char *s){
	fflush(stdout);
	fprintf(stderr,"Syntax error near line %d in file %s. %s\n", yyget_lineno(), q2j_input_file_name, s);
	exit(-1);
}



void add_type(char *new_type, char *old_type){
    unsigned long int h;
    type_list_t *t, *tmp;

    if( NULL != lookup_type(new_type) ){
        fprintf(stderr,"type defined aready\n");
        exit(-1);
    }

    tmp = (type_list_t *)calloc(1, sizeof(type_list_t));
    tmp->new_type = strdup(new_type);
    tmp->old_type = strdup(old_type);

    h = hash(new_type);
    t=type_hash[h];
    if( NULL == t ){
        type_hash[h] = tmp;
        return;
    }

    for(; NULL != t->next; t=t->next);
    t->next = tmp;
 
    return;
}



char *lookup_type(char *new_type){
    unsigned long int h;
    type_list_t *t;

    h = hash(new_type);
    for(t=type_hash[h]; NULL != t; t=t->next){
        if( !strcmp(t->new_type, new_type) )
            return t->old_type;
    }

    return NULL;
}


/* Modified djb2 hash from: http://www.cse.yorku.ca/~oz/hash.html */
unsigned long hash(char *str) {
    unsigned long result = 5381;
    int c;

    while( 0 != (c = *str++) ){
        result = ((result << 5) + result) + c; /* result * 33 + c */
    }

    unsigned long tmp = result;
    while( tmp > (HASH_TAB_SIZE/4) ){
        tmp /= 2;
    }
    result = tmp + result%(HASH_TAB_SIZE-(HASH_TAB_SIZE/4));

    return result;
}

