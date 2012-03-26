/** Freely inspired from http://xmlsoft.org/examples/ */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#include <GTG.h>
#include <GTGPaje.h>

#ifdef LIBXML_TREE_ENABLED

/*
 *To compile this file using gcc you can type
 *gcc $(xml2-config --cflags --libs) -o xml2traces xml2traces.c
 */
FILE *pajeGetProcFile();

trace_return_t pajeSetState2(varPrec time, const char* type,
                             const char *cont, const char* val, 
                             const int x, const int y) 
{
    FILE *procFile = pajeGetProcFile();
    if (procFile){
        fprintf (procFile, "10 %.13e \"%s\" \"%s\" \"%s\" %d %d\n", 
                 time, type, cont, val, x, y);
        return TRACE_SUCCESS;
    }
    return TRACE_ERR_WRITE;
}

trace_return_t pajePushState2(varPrec time, const char* type,
                              const char*  cont, const char* val, 
                              const int x, const int y)
{
    FILE *procFile = pajeGetProcFile();
    if (procFile){
        fprintf (procFile, "11 %.13e \"%s\" \"%s\" \"%s\" %d %d\n", 
                 time, type, cont, val, x, y);
        return TRACE_SUCCESS;
    }
    return TRACE_ERR_WRITE;
}

trace_return_t pajePopState2(varPrec time, const char* type,
                             const char*  cont, 
                             const int x, const int y)
{
    FILE *procFile = pajeGetProcFile();
    if (procFile){
        fprintf (procFile, "12 %.13e \"%s\" \"%s\" %d %d\n", 
                 time, type, cont, x, y);
        return TRACE_SUCCESS;
    }
    return TRACE_ERR_WRITE;
}


static xmlNodePtr xmlGetFirstNodeWithName(xmlNodePtr parent, const xmlChar *name)
{
    xmlNodePtr c, p;
    for(c = parent->children; c; c = c->next) {
        if( xmlStrEqual(c->name, name) )
            return c;
        if( NULL != (p = xmlGetFirstNodeWithName(c, name)) )
            return p;
    }
    return NULL;
}

static xmlChar *xmlGetFirstNodeChildContentWithName(xmlNodePtr parent, const xmlChar *name)
{
    xmlNodePtr p = xmlGetFirstNodeWithName(parent, name);
    if( p == NULL ) return NULL;
    if( p->children == NULL ) return NULL;
    return p->children->content;
}

static char *getMPIContainerIdentifier( const char *fileid ) {
    char *t = strdup(fileid);
    char *r = t + strlen(t) - 8;
    char *ret;
    *r = '\0';
    while ( *r != '.' )
        r--;

    asprintf( &ret, "P%s", r+1 );
    free( t );
    return ret;
}

static char *getThreadContainerIdentifier( const char *prefix, const char *identifier ) {
    const char *r = identifier + strlen(identifier) - 1;
    char *ret;

    while ( *r != ' ' )
        r--;

    asprintf( &ret, "%sT%s", prefix, r+1);
    return ret;
}

int main(int argc, char **argv)
{
    xmlDoc *doc = NULL;
    xmlXPathContextPtr xpathCtx; 
    xmlXPathObjectPtr xpathObj_INFOS; 
    xmlXPathObjectPtr xpathObj_DICO;
    xmlXPathObjectPtr xpathObj_CONT;
    xmlNodePtr tmp;
    xmlChar *dico_id, *dico_name, *dico_attr;
    xmlChar *app_name;
    char *cont_mpi_name, *cont_thread_name, *name;
    int x, y;
    int i, nbkeys;
    xmlNodePtr *key_heads;
    gtg_color_t color;
    unsigned long int color_code;
    traceType_t fmt = PAJE;

    if (argc != 2)
        return(1);

    /* Init GTG */
    setTraceType(fmt);
    initTrace ("out", 0, GTG_FLAG_NONE);
    if( fmt == PAJE || fmt == VITE ) {
        pajeEventDefAddParam( GTG_PAJE_EVTDEF_SetState,  "CoordX", GTG_PAJE_FIELDTYPE_Int );
        pajeEventDefAddParam( GTG_PAJE_EVTDEF_SetState,  "CoordY", GTG_PAJE_FIELDTYPE_Int );
        pajeEventDefAddParam( GTG_PAJE_EVTDEF_PushState, "CoordX", GTG_PAJE_FIELDTYPE_Int );
        pajeEventDefAddParam( GTG_PAJE_EVTDEF_PushState, "CoordY", GTG_PAJE_FIELDTYPE_Int );
        pajeEventDefAddParam( GTG_PAJE_EVTDEF_PopState,  "CoordX", GTG_PAJE_FIELDTYPE_Int );
        pajeEventDefAddParam( GTG_PAJE_EVTDEF_PopState,  "CoordY", GTG_PAJE_FIELDTYPE_Int );
    }
    addContType ("CT_Appli", "0", "Application");
    addContType ("CT_P", "CT_Appli", "Process");
    addContType ("CT_T", "CT_P", "Thread");
    addStateType("ST_TS", "CT_T", "Thread State");
    addEntityValue ("Wait", "ST_TS", "Waiting", GTG_LIGHTGREY);

    /*
     * this initialize the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    { LIBXML_TEST_VERSION }
    
    /*parse the file and get the DOM */
    doc = xmlReadFile(argv[1], NULL, XML_PARSE_NOBLANKS | XML_PARSE_NOCDATA | XML_PARSE_NONET);

    if (doc == NULL) {
        printf("error: could not parse file %s\n", argv[1]);
    }

    xpathCtx = xmlXPathNewContext(doc);
    if(xpathCtx == NULL) {
        fprintf(stderr,"Error: unable to create new XPath context\n");
        xmlFreeDoc(doc); 
        return(-1);
    }

    app_name = xmlGetFirstNodeChildContentWithName( xmlDocGetRootElement( doc ), (xmlChar*)"IDENTIFIER" );
    addContainer (0.00000, "Appli", "CT_Appli", "0", (char*)app_name, "");

    xpathObj_INFOS = xmlXPathEvalExpression((xmlChar*)"//INFOS/INFO", xpathCtx);
    if(xpathObj_INFOS == NULL) {
        fprintf(stderr,"Error: unable to find any //INFOS/INFO in this profile\n");
        xmlXPathFreeContext(xpathCtx); 
        xmlFreeDoc(doc); 
        return(-1);
    }
        
    /* Iterate on each key */
    xpathObj_DICO = xmlXPathEvalExpression((xmlChar*)"//DICTIONARY/KEY", xpathCtx);
    if(xpathObj_DICO == NULL || xpathObj_DICO->nodesetval == NULL) {
        fprintf(stderr,"Error: unable to find any //DICTIONARY/KEY in this profile\n");
        xmlXPathFreeContext(xpathCtx);
        xmlXPathFreeObject(xpathObj_INFOS);
        xmlFreeDoc(doc); 
        return(-1);
    }

    for(i = 0; i < xpathObj_DICO->nodesetval->nodeNr; i++) {
        /* addEntityValue */
        dico_id = xmlGetProp( xpathObj_DICO->nodesetval->nodeTab[i], (xmlChar*)"ID" );
        dico_name = xmlGetFirstNodeChildContentWithName( xpathObj_DICO->nodesetval->nodeTab[i], (xmlChar*)"NAME" );
        dico_attr = xmlGetFirstNodeChildContentWithName( xpathObj_DICO->nodesetval->nodeTab[i], (xmlChar*)"ATTRIBUTES" );
        
        if( NULL == dico_attr ||
            NULL == dico_name ||
            NULL == dico_id ) {
            fprintf(stderr, "Malformed profiles file for reason 1.\n");
            return -143;
        }

        color_code = strtoul( (char*)dico_attr + strlen((char*)dico_attr) - 6, NULL, 0);
        color = gtg_color_create((char*)dico_name, 
                                 GTG_COLOR_GET_RED(color_code),
                                 GTG_COLOR_GET_GREEN(color_code),
                                 GTG_COLOR_GET_BLUE(color_code));
        addEntityValue ((char*)dico_id, "ST_TS", (char*)dico_name, color);
        gtg_color_free(color);
        /*printf("KEY ID %s: NAME=%s, ATTRIBUTES=%s\n", dico_id, dico_name, dico_attr);*/
    }

    /* Iterate on each container */
    xpathObj_CONT = xmlXPathEvalExpression((xmlChar*)"//DISTRIBUTED_PROFILE/NODE", xpathCtx);
    if(xpathObj_CONT == NULL || xpathObj_CONT->nodesetval == NULL) {
        fprintf(stderr,"Error: unable to find any //DISTRIBUTED_PROFILE/NODE in this profiles\n");
        xmlXPathFreeContext(xpathCtx);
        xmlXPathFreeObject(xpathObj_INFOS);
        xmlXPathFreeObject(xpathObj_DICO);
        xmlFreeDoc(doc); 
        return(-1);
    }

    for(i = 0; i < xpathObj_CONT->nodesetval->nodeNr; i++) {
        xmlNodePtr cn, ct;
        xmlChar *fileid;
         
        cn = xpathObj_CONT->nodesetval->nodeTab[i];

        fileid = xmlGetProp( cn, (xmlChar*)"FILEID");
        if( NULL == fileid ) {
            fprintf(stderr, "Malformed profiles file for reason 2a\n");
            return -144;
        }

        /*printf("MPI container: %s\n", cont_mpi_name);*/
        cont_mpi_name = getMPIContainerIdentifier( (char*)fileid );
        asprintf( &name, "MPI-%s", cont_mpi_name+1 );
        addContainer (0.00000, cont_mpi_name, "CT_P", "Appli", name, cont_mpi_name);
        free(name);

        for(ct = xmlGetFirstNodeWithName( cn, (xmlChar*)"THREAD" );
            ct;
            ct = ct->next) {
            xmlChar *identifier;

            if( !xmlStrEqual(ct->name, (xmlChar*)"THREAD") ) continue;
            identifier = xmlGetFirstNodeChildContentWithName(ct, (xmlChar*)"IDENTIFIER");

            if( NULL == identifier ) {
                fprintf(stderr, "Malformed profiles file for reason 2b\n");
                return -144;
            }
            /*printf("Thread container: %s\n", cont_thread_name);*/
            cont_thread_name = getThreadContainerIdentifier( cont_mpi_name, (char*)identifier );
            addContainer (0.00000, cont_thread_name, "CT_T", cont_mpi_name, (char*)(identifier)+6, cont_thread_name);

            nbkeys = 0;
            for(tmp = xmlGetFirstNodeWithName( ct, (xmlChar*)"KEY" );
                tmp;
                tmp = tmp->next) {

                if( !xmlStrEqual(tmp->name, (xmlChar*)"KEY") ) continue;
                nbkeys++;
            }
            key_heads = (xmlNodePtr*)malloc(nbkeys * sizeof(xmlNodePtr));
            nbkeys = 0;
            for(tmp = xmlGetFirstNodeWithName( ct, (xmlChar*)"KEY" );
                tmp;
                tmp = tmp->next) {

                if( !xmlStrEqual(tmp->name, (xmlChar*)"KEY") ) continue;
                key_heads[nbkeys] = xmlGetFirstNodeWithName( tmp, (xmlChar*)"EVENT" );
                assert( key_heads[nbkeys] != NULL);

                nbkeys++;
            }

            do {
                xmlNodePtr e;
                xmlChar *id, *start, *end, *info;
                xmlChar *keyid;
                long long int sd, best_sd;
                int best;

                /* Find the key with the smallest start date.
                 * Assumes that for a given key, events are ordered. */
                best = -1;
                for(i = 0; i < nbkeys; i++) {
                    if( NULL == (e = key_heads[i]) ) {
                        continue;
                    }
                    start = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"START" );
                    if( start == NULL ) {
                        fprintf(stderr, "Malformed profiles file for reason 3a\n");
                        return -145;
                    }
                    sd = strtoll((char*)start, NULL, 0);
                    if( (best==-1) || (sd < best_sd) ) {
                        best = i;
                        best_sd = sd;
                    }
                }
                /* best is the index of the best key head */
                if( best == -1 )
                    break;
                /* store in into tmp and consume this head */
                e = key_heads[best];
                key_heads[best] = e->next;
                assert( key_heads[best] == NULL || xmlStrEqual( key_heads[best]->name, (xmlChar*)"EVENT") );

                keyid = xmlGetProp(e->parent, (xmlChar*)"ID");
                 
                id    = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"ID"    );
                start = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"START" );
                end   = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"END"   );
                info  = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"INFO"  );

                if( NULL == id ||
                    NULL == start ||
                    NULL == end ) {
                    fprintf(stderr, "Malformed profiles file for reason 3\n");
                    return -145;
                }

                if( NULL != info ) {
                    sscanf((const char *)info, "(null)(%d, %d)", &x, &y);
                } else {
                    x = -1;
                    y = -1;
                }

                if( fmt == PAJE || fmt == VITE ) {
                    pajeSetState2( strtoll((char*)start, NULL, 0) * 1e-3, "ST_TS", cont_thread_name, (char*)keyid, x, y);
                    pajeSetState2( strtoll((char*)end,   NULL, 0) * 1e-3, "ST_TS", cont_thread_name, "Wait", x, y );
                } else {
                    setState( strtoll((char*)start, NULL, 0) * 1e-3, "ST_TS", cont_thread_name, (char*)keyid);
                    setState( strtoll((char*)end,   NULL, 0) * 1e-3, "ST_TS", cont_thread_name, "Wait" );
                }

                printf("  %s %s %s %s %s. best = %d\n", keyid, id, start, end, info, best);
            } while(1);

            nbkeys = 0;
            free(key_heads);
            key_heads = NULL;
        }
    }     

    endTrace();

    /*free the document */
    xmlXPathFreeContext(xpathCtx);
    xmlXPathFreeObject(xpathObj_CONT);
    xmlXPathFreeObject(xpathObj_INFOS);
    xmlXPathFreeObject(xpathObj_DICO);
    xmlFreeDoc(doc);

    /*
     *Free the global variables that may
     *have been allocated by the parser.
     */
    xmlCleanupParser();

    return 0;
}
#else
int main(void) {
    fprintf(stderr, "Tree support not compiled in\n");
    exit(1);
}
#endif
