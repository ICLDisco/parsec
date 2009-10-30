#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assignment.h"
#include "dplasma.h"
#include "expr.h"
#include "symbol.h"

/*
 * Forward Declarations
 */

void external_hook(void);
void processTask(dplasma_t *currTask, int localsCount, int whichLocal, assignment_t *assgn, unsigned int nbassgn);
void generateTaskInstances(dplasma_t *currTask, assignment_t *assgn, unsigned int nbassgn);
void generateEdges(dplasma_t *currTask, assignment_t *assgn, unsigned int nbassgn);
void generateEdge(dplasma_t *currTask, char *fromNodeStr, assignment_t *assgn, unsigned int nbassgn, dep_t *dep);
void printNodeColor(dplasma_t *currTask, char *taskInstanceStr,assignment_t *assgn, unsigned int nbassgn);
void generatePeerNode(dep_t *peerNode, char *fromNodeStr, unsigned int whichCallParam, int callParamCount, assignment_t *assgn, unsigned int nbassgn, int *callParamsV);


/*
 * Actual Functions
 */

/**************************************************************************/
int nameToColor(const char *name){
    int i, len, tmp, rslt;

    rslt = 1234;
    len = strlen(name);

    for(i=0; i<len; ++i){
        tmp = (int)name[i];
        tmp += (tmp<<16) + (tmp<<8);
        rslt = rslt + tmp;
        rslt = rslt*3;
        tmp |= tmp*11;
        rslt = rslt ^ tmp;
        rslt += 0x101010;
        rslt %= 0xffffff;
    }

    return rslt;
}

/**************************************************************************/
void generatePeerNode(dep_t *peerNode, char *fromNodeStr, unsigned int whichCallParam, int callParamCount, assignment_t *assgn, unsigned int nbassgn, int *callParamsV){
    int success = 0;
    int i, ret_val, res;

    /* Do not generate the OUT nodes, it makes comparison harder */
    if( !strcmp(peerNode->dplasma->name,"OUT") )
        return;

    if( whichCallParam == callParamCount ){
        printf("  %s -> %s",fromNodeStr, peerNode->dplasma->name);
        for(i=0; i<callParamCount; ++i){
            printf("_%d",callParamsV[i]);
        }
        printf(";\n");
        return;
    }

    if( peerNode->call_params[whichCallParam] == NULL ){
        fprintf(stderr,"Please plant a tree\n");
        exit(-1);
    }

    ret_val = expr_eval((expr_t *)peerNode->call_params[whichCallParam], assgn, nbassgn, &res);
    if( EXPR_SUCCESS == ret_val ){
        callParamsV[whichCallParam] = res;
        generatePeerNode(peerNode, fromNodeStr, whichCallParam+1, callParamCount, assgn, nbassgn, callParamsV);
        success = 1;
    }else if( EXPR_FAILURE_CANNOT_EVALUATE_RANGE == ret_val ){
        int j, min, max;
        
        ret_val = expr_range_to_min_max((expr_t *)peerNode->call_params[whichCallParam], assgn, nbassgn, &min, &max);
        if( EXPR_SUCCESS == ret_val ){
            success = 1;
            for(j=min; j<=max; ++j){
                callParamsV[whichCallParam] = j;
                generatePeerNode(peerNode, fromNodeStr, whichCallParam+1, callParamCount, assgn, nbassgn, callParamsV);
            }
        }
    }

    if( !success ){
        printf("Can't evaluate expression for dep: %s ",peerNode->dplasma->name);
        printf("call_params[%d] ",i);
        expr_dump( peerNode->call_params[whichCallParam] );
        printf("\n");
        exit(-1);
    }

    return;
}

/**************************************************************************/
void generateEdge(dplasma_t *currTask, char *fromNodeStr, assignment_t *assgn, unsigned int nbassgn, dep_t *dep){
    int i, res, ret_val, callParamsV[MAX_CALL_PARAM_COUNT], callParamCount;

    if( dep->cond != NULL ){
        ret_val = expr_eval(dep->cond, assgn, nbassgn, &res);
        if( EXPR_SUCCESS != ret_val ){
            printf("Can't evaluate expression for dep:\n  ");
            expr_dump( dep->cond );
            printf("\n");
            exit(-1);
        }

        /* If there is a dependency and it's not true, do not generate an edge */
        if( res == 0 ) return;
    }
    for(i=0;  i < MAX_CALL_PARAM_COUNT && dep->call_params[i] != NULL; ++i );
    callParamCount=i;
    generatePeerNode(dep, fromNodeStr, 0, callParamCount, assgn, nbassgn, callParamsV);

    return;
}

/**************************************************************************/
void printNodeColor(dplasma_t *currTask, char *taskInstanceStr,assignment_t *assgn, unsigned int nbassgn){
    int i, color, val = 0;
    expr_t *e;

    for(i=0; i<MAX_PRED_COUNT; ++i){
        int res, max;
        if( currTask->preds[i] == NULL ) break;
        e = currTask->preds[i];
        if( !EXPR_IS_BINARY(e->op) || !EXPR_IS_BINARY(e->bop1->op) ){
            fprintf(stderr,"Predicate %d does not conform to the expected format \"actual %% global1 == global2\": ",i);
            expr_dump(e);
            fprintf(stderr,"\n");
            exit(-1);
        }
        e = (expr_t *)currTask->preds[i]->bop1;

        /* Get the value of the LHS of the predicate, i.e. "k % GRIDrows" */
        if( EXPR_SUCCESS != expr_eval(e, assgn, nbassgn, &res) ){
            fprintf(stderr,"Cannot evaluate LHS of predicate %d: ",i);
            expr_dump(e);
            fprintf(stderr,"\n");
            exit(-1);
        }

        /* Get the value of the divisor, i.e. GRIDrows, GRIDcols, etc */
        if( EXPR_SUCCESS != expr_eval(e->bop2, assgn, nbassgn, &max) ){
            if( EXPR_OP_SYMB != e->bop2->op ){
                fprintf(stderr,"Expecting to find symbol instead of expression: ");
                expr_dump(e->bop2);
                fprintf(stderr,"\n");
                exit(-1);
            }
            fprintf(stderr,"Cannot evaluate the value of global variable: %s",e->bop2->var->name);
            expr_dump(e);
            fprintf(stderr,"\n");
            exit(-1);
        }
        if( i >= 3 ){
            fprintf(stderr,"Error: parameter spaces have to be limited to 3D\n");
            break;
        }
        val = val*max + res;
    }
    val += 4;
    color = nameToColor(currTask->name);
    printf("  %s [shape=\"polygon\" sides=\"%d\" color=\"#000000\" fillcolor=\"#%x\"];\n",taskInstanceStr, val, color);

    return;
}

/**************************************************************************/
void generateEdges(dplasma_t *currTask, assignment_t *assgn, unsigned int nbassgn){
    int i, j, k, off, len;
    param_t *currParam;
    dep_t *currOutDep;
    char *taskInstanceStr;

    /* if the locals take values that exceed 2^32 this string might overflow */
    len = strlen(currTask->name) + nbassgn*10;
    taskInstanceStr = (char *)malloc( len * sizeof(char) );
    off = sprintf(taskInstanceStr,"%s",currTask->name);

    for(i=0; i<nbassgn; ++i){
        off += sprintf(taskInstanceStr+off,"_%d",assgn[i].value);
    }

    printNodeColor(currTask, taskInstanceStr, assgn, nbassgn);

    for(j=0; j<MAX_PARAM_COUNT; ++j){
        if( (currParam=currTask->params[j]) == NULL ) break;
        for(k=0; k<MAX_DEP_OUT_COUNT; ++k){
            if( (currOutDep=currParam->dep_out[k]) == NULL ) break;
            generateEdge(currTask, taskInstanceStr, assgn, nbassgn, currOutDep);
        }
    }

    free(taskInstanceStr);

}

/**************************************************************************/
void processTask(dplasma_t *currTask, int localsCount, int whichLocal, assignment_t *assgn, unsigned int nbassgn) {
    int i, lb, ub;
    assignment_t *assgnNew;

    /* Evaluate the lower bound for the local 'whichLocal' */
    if( EXPR_SUCCESS != expr_eval((expr_t *)currTask->locals[whichLocal]->min, assgn, nbassgn, &lb) ){
        printf("Can't evaluate expression for min:\n  ");
        expr_dump( currTask->locals[whichLocal]->min );
        printf("\n");
        exit(-1);
    }

    /* Evaluate the upper bound for the local 'whichLocal' */
    if( EXPR_SUCCESS != expr_eval((expr_t *)currTask->locals[whichLocal]->max, assgn, nbassgn, &ub) ){
        printf("Can't evaluate expression for max:\n  ");
        expr_dump( currTask->locals[whichLocal]->max );
        printf("\n");
        exit(-1);
    }

    assgnNew = (assignment_t *)malloc( (nbassgn+1)*sizeof(assignment_t) );
    memcpy( assgnNew, assgn, nbassgn*sizeof(assignment_t) );
    for(i=lb; i<=ub; ++i){
        assgnNew[nbassgn].sym = currTask->locals[whichLocal];
        assgnNew[nbassgn].value = i;

        /* if 'whichLocal' is the last local, then 'assgn' holds a vector of values for all local symbols */
        if( whichLocal == localsCount-1 ){
            generateEdges(currTask, assgnNew, nbassgn+1);
        }else{ /* else recursively process the next local */
            processTask(currTask, localsCount, whichLocal+1, assgnNew, nbassgn+1);
        }
    }
    free(assgnNew);

    return;
}

/**************************************************************************/
void external_hook(void){
    int i, j;

    printf("digraph DAG {\n");
    printf("  node [shape = oval, style = filled];\n");

    for( i = 0; ;i++ ) {
        const dplasma_t *currTask=dplasma_element_at(i);
        int localsCount;
        if( currTask == NULL ) break;
        for(j=0; currTask->locals[j] != NULL && j<MAX_LOCAL_COUNT; j++);
        localsCount = j;
        if( localsCount == 0 ){ continue; }
        processTask((dplasma_t *)currTask, localsCount, 0, NULL, 0);
    }
    printf("}\n");

    return;
}

