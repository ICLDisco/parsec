/** Freely inspired from http://xmlsoft.org/examples/ */

#include <stdio.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#ifdef LIBXML_TREE_ENABLED

/*
 *To compile this file using gcc you can type
 *gcc $(xml2-config --cflags --libs) -o xml2traces xml2traces.c
 */

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

int main(int argc, char **argv)
{
    xmlDoc *doc = NULL;
    xmlXPathContextPtr xpathCtx; 
    xmlXPathObjectPtr xpathObj_INFOS; 
    xmlXPathObjectPtr xpathObj_DICO;
    xmlXPathObjectPtr xpathObj_CONT;
    xmlNodePtr tmp;
    xmlChar *dico_id, *dico_name, *dico_attr;
    xmlChar *cont_mpi_name, *cont_thread_name;
    int i;

    if (argc != 2)
        return(1);

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

        printf("KEY ID %s: NAME=%s, ATTRIBUTES=%s\n", dico_id, dico_name, dico_attr);
    }

    /* Iterate on each container */
    xpathObj_CONT = xmlXPathEvalExpression((xmlChar*)"//DISTRIBUTED_PROFILE/NODE/PROFILES/THREAD", xpathCtx);
     if(xpathObj_CONT == NULL || xpathObj_CONT->nodesetval == NULL) {
        fprintf(stderr,"Error: unable to find any //DISTRIBUTED_PROFILE/NODE/PROFILES/THREAD in this profiles\n");
        xmlXPathFreeContext(xpathCtx);
        xmlXPathFreeObject(xpathObj_INFOS);
        xmlXPathFreeObject(xpathObj_DICO);
        xmlFreeDoc(doc); 
        return(-1);
     }
     for(i = 0; i < xpathObj_CONT->nodesetval->nodeNr; i++) {
         xmlNodePtr ct;
         
         ct = xpathObj_CONT->nodesetval->nodeTab[i];

         cont_mpi_name = xmlGetProp( ct->parent->parent, (xmlChar*)"FILEID");
         cont_thread_name = xmlGetFirstNodeChildContentWithName(ct, (xmlChar*)"IDENTIFIER");

         if( NULL == cont_mpi_name ||
             NULL == cont_thread_name ) {
             fprintf(stderr, "Malformed profiles file for reason 2: %s is NULL\n",
                     NULL == cont_mpi_name ? "cont_mpi_name" : "cont_thread_name");
             return -144;
         }
         printf("MPI container: %s\n", cont_mpi_name);
         printf("Thread container: %s\n", cont_thread_name);

         for(tmp = xmlGetFirstNodeWithName( ct, (xmlChar*)"KEY" );
             tmp;
             tmp = tmp->next) {
             xmlNodePtr e;
             xmlChar *id, *start, *end, *info;
             xmlChar *keyid;

             if( !xmlStrEqual(tmp->name, (xmlChar*)"KEY") ) continue;

             keyid = xmlGetProp(tmp, (xmlChar*)"ID");

             for(e = xmlGetFirstNodeWithName( tmp, (xmlChar*)"EVENT" );
                 e;
                 e = e->next) {
                 if( !xmlStrEqual(e->name, (xmlChar*)"EVENT") ) continue;
                 
                 id = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"ID" );
                 start = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"START" );
                 end = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"END" );
                 info = xmlGetFirstNodeChildContentWithName( e, (xmlChar*)"INFO" );

                 if( NULL == id ||
                     NULL == start ||
                     NULL == end ||
                     NULL == info ) {
                     fprintf(stderr, "Malformed profiles file for reason 3\n");
                     return -145;
                 }

                 printf("  %s %s %s %s %s\n", keyid, id, start, end, info);
             }
         }
     }
     

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
