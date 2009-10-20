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
void generatePeerNode(dep_t *peerNode, char *fromNodeStr, unsigned int whichCallParam, int callParamCount, assignment_t *assgn, unsigned int nbassgn, int *callParamsV);


/*
 * Actual Functions
 */

/**************************************************************************/
void generatePeerNode(dep_t *peerNode, char *fromNodeStr, unsigned int whichCallParam, int callParamCount, assignment_t *assgn, unsigned int nbassgn, int *callParamsV){
    int success = 0;
    int i, ret_val, res;

    if( whichCallParam == callParamCount ){
        printf("  %s -> %s",fromNodeStr, peerNode->dplasma_name);
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
        printf("Can't evaluate expression for dep: %s ",peerNode->dplasma_name);
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
    printf("  node [shape = circle];\n");

    for( i = 0; ;i++ ) {
        const dplasma_t *currTask=dplasma_element_at(i);
        int localsCount;
        if( currTask == NULL ) break;
        for(j=0; currTask->locals[j] != NULL && j<MAX_LOCAL_COUNT; j++);
        localsCount = j;
        processTask((dplasma_t *)currTask, localsCount, 0, NULL, 0);
    }
    printf("}\n");

    return;
}

