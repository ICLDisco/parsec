/*
 * Copyright (c)      2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */
#include <string.h>
#include <complex.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <assert.h>

typedef struct property_s        property_t;
typedef struct function_s        function_t;
typedef struct namespace_s       namespace_t;
typedef struct category_s        category_t;
typedef struct profiling_shmem_s profiling_shmem_t;

const int PAGE_SIZE = 4096;

/* UTF8 encoding is of variable size. However, it accepts ASCII,
 * and only ASCII, on one byte per char, so if the buffer size is
 * equal to the string length, the buffer can safely be cast to
 * char* */
#if !defined(NDEBUG)
static const char *xml_safe_cast(const xmlChar *b)
{
    int len, size;
    assert(xmlCheckUTF8(b));
    len = xmlUTF8Strlen(b);
    size = xmlUTF8Size(b);
    assert(len == size);
    return (const char*)b;
}
#else
#define xml_safe_cast(xmlbuffer) ((char*)xmlbuffer)
#endif

xmlNode *
findNode(xmlNode *node, const char *target)
{
    if (!xmlStrcmp(node->name, (const xmlChar *)target))
        return node;
    xmlNode *cur_node = xmlFirstElementChild(node);
    xmlNode *F = NULL;
    while (cur_node != NULL) {
        if (NULL != (F = findNode(cur_node, target))) break;
        cur_node = xmlNextElementSibling(cur_node);
    }
    return F;
}

static int parse_int(xmlNode *node)
{
    xmlChar *buf = xmlNodeGetContent(node);
    int value = atoi(xml_safe_cast(buf));
    xmlFree(buf);
    return value;
}

struct property_s {
    char        *name;
    size_t       offset;
    char         type;
    function_t  *function;
};

struct function_s {
    char        *name;
    int          nb_properties;
    property_t  *properties;
    namespace_t *namespace;
};

struct namespace_s {
    char        *name;
    int          nb_functions;
    function_t  *functions;
    category_t  *category;
};

struct category_s {
    char        *name;
    namespace_t *namespaces;
    int          nb_namespaces;
};

struct profiling_shmem_s {
    int                                 version;
    int                                 running;
    int                                 prank;
    int                                 psize;
    int                                 nb_xml_pages;
    int                                 nb_node_pages;
    int                                 nb_vp_pages;
    int                                 nb_eu_pages;
    int                                 nb_vp;
    int                                 nb_eu;
    int                                 first_node;
    int                                 first_vp;
    int                                 first_eu;
    size_t                              xml_sz;
    char                               *header;
    int                                 shm_fd;
    char                               *shmem_name;
    void                               *buffer;
    int                                 nb_pages;
    category_t                         *per_nd;
    category_t                         *per_vp;
    category_t                         *per_eu;
};


static void print_value(char *value, property_t *prop, void *buff)
{
    switch (prop->type) {
    case 'i': sprintf(value, "%d", *(int32_t*)buff); break;
    case 'q': sprintf(value, "%"PRId64"", *(int64_t*)buff); break;
    case 'f': sprintf(value, "%f", *(float*)buff); break;
    case 'd': sprintf(value, "%lf", *(double*)buff); break;
    }
    return;        
}

static void pretty_print_property(property_t *property, char *buffer)
{
    char value[256];
    print_value(value, property, buffer+property->offset);
    
    fprintf(stdout, "    %s:%s:%s \t %s\n",
            property->function->namespace->name,
            property->function->name,
            property->name,
            value);
}

static void pretty_print_function(function_t *function, char *buffer)
{
    int p;
    for (p = 0; p < function->nb_properties; ++p)
        pretty_print_property(function->properties+p, buffer);
}

static void pretty_print_namespace(namespace_t *namespace, char *buffer)
{
    int f;
    for (f = 0; f < namespace->nb_functions; ++f)
        pretty_print_function(namespace->functions+f, buffer);
}

static void pretty_print_category(category_t *category, char *buffer)
{
    int n;
    for (n = 0; n < category->nb_namespaces; ++n)
        pretty_print_namespace(category->namespaces+n, buffer);
}

static void pretty_print(profiling_shmem_t *shmem)
{
    fprintf(stdout, "Node %d/%d {\n", shmem->prank, shmem->psize);
    pretty_print_category(shmem->per_nd, shmem->buffer+(shmem->nb_xml_pages*PAGE_SIZE));

    int vp;
    for (vp = 0; vp < shmem->nb_vp; ++vp) {
        fprintf(stdout, "  VP %d {\n", vp);
        pretty_print_category(shmem->per_vp, shmem->buffer+(shmem->first_vp+vp)*PAGE_SIZE);
        fprintf(stdout, "  }\n");
    }

    int eu;
    for (eu = 0; eu < shmem->nb_eu; ++eu) {
        fprintf(stdout, "  EU %d {\n", eu);
        pretty_print_category(shmem->per_eu, shmem->buffer+(shmem->first_eu+eu)*PAGE_SIZE);
        fprintf(stdout, "  }\n");
    }
    fprintf(stdout, "}\n");
}


static void init_property(property_t *property, xmlNode *property_node, function_t *function)
{
    xmlNode *type_node, *offset_node;
    xmlChar *buf;
    type_node   = findNode(property_node, "t");
    offset_node = findNode(property_node, "o");
    property->function = function;
    property->name = strdup( xml_safe_cast(property_node->name) );
    property->offset = parse_int(offset_node);
    buf = xmlNodeGetContent(type_node);
    property->type = xml_safe_cast(buf)[0];
    xmlFree(buf);
    return;
}

static void init_function(function_t *function, xmlNode *function_node, namespace_t *namespace)
{
    int nb_property = xmlChildElementCount(function_node);
    if (nb_property > 0) {
        function->name = strdup( xml_safe_cast(function_node->name) );
        function->nb_properties = nb_property;
        function->properties = (property_t*)calloc(nb_property, sizeof(property_t));
        function->namespace = namespace;
        xmlNode *cur_node = xmlFirstElementChild(function_node);
        int i = 0;
        fprintf(stdout, "    {%s}\n", function->name);
        while (cur_node != NULL) {
            init_property(function->properties+i, cur_node, function);
            i++;
            cur_node = xmlNextElementSibling(cur_node);
        }
    }
    return;
}

static void init_namespace(namespace_t *namespace, xmlNode *namespace_node, category_t *category)
{
    int nb_function = xmlChildElementCount(namespace_node);
    if (nb_function > 0) {
        namespace->name = strdup( xml_safe_cast(namespace_node->name) );
        namespace->nb_functions = nb_function;
        namespace->functions = (function_t*)calloc(nb_function, sizeof(function_t));
        namespace->category = category;
        xmlNode *cur_node = xmlFirstElementChild(namespace_node);
        int i = 0;
        fprintf(stdout, "  {%s}\n", namespace->name);
        while (cur_node != NULL) {
            init_function(namespace->functions+i, cur_node, namespace);
            i++;
            cur_node = xmlNextElementSibling(cur_node);
        }
    }
    return;
}

static category_t *init_category(xmlNode *category_node)
{
    category_t *category = (category_t*)calloc(1, sizeof(category_t));
    int nb_namespace = xmlChildElementCount(category_node);
    if (nb_namespace > 0) {
        category->name = strdup( xml_safe_cast(category_node->name) );
        category->nb_namespaces = nb_namespace;
        category->namespaces = (namespace_t*)calloc(nb_namespace, sizeof(namespace_t));        
        xmlNode *cur_node = xmlFirstElementChild(category_node);
        int i = 0;
        fprintf(stdout, "{%s}\n", category->name);
        while (cur_node != NULL) {
            init_namespace(category->namespaces+i, cur_node, category);
            i++;
            cur_node = xmlNextElementSibling(cur_node);
        }
    }
    return category;
}


int
main(int argc, char *argv[])
{
    int shm_fd;
    xmlNode *root, *application_node, *prank_node, *psize_node, *nb_vp_node, *nb_eu_node,
        *per_nd_node, *per_vp_node, *per_eu_node,
        *pages_per_nd_node, *pages_per_vp_node, *pages_per_eu_node;
    profiling_shmem_t *shmem = NULL;

    if ((2 > argc) ||
        ((argc == 2) && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-help")))) {
        fprintf(stdout, "Usage: %s <shmem_name> [data|header]\n", argv[0]);
        fprintf(stdout, "data: client will loop and dump data from the shared memory area.\n");
        fprintf(stdout, "header: client displays header of shared memory and quits.\n");
        return 0;
    }

    if (2 <= argc)
        shm_fd = shm_open(argv[1], O_RDONLY, 0666);
    else
        shm_fd = shm_open("parsec_shmem", O_RDONLY, 0666);

    void *ptr;
    if (MAP_FAILED == (ptr = mmap(0, PAGE_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0))) {
        fprintf(stderr, "Map 1 PAGE failed\n");
        return -1;
    }

    if (NULL == (shmem = (profiling_shmem_t*)calloc(1, sizeof(profiling_shmem_t)))) {
        fprintf(stderr, "Failed allocating profiling_shmem.\n");
        return -1;
    }
    shmem->shm_fd = shm_fd;
    shmem->nb_pages = *((int*)ptr);

    munmap(ptr, PAGE_SIZE);
    if (MAP_FAILED == (ptr = mmap(0, shmem->nb_pages*PAGE_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0))) {
        fprintf(stderr, "Map %d PAGES failed\n", shmem->nb_pages);
        return -1;
    }

    shmem->buffer = ptr;
    shmem->header = ptr;
    shmem->header += 3*sizeof(int);
    shmem->nb_xml_pages = 1+(strlen(shmem->header)-1)/PAGE_SIZE;
    shmem->running = ((int*)shmem->buffer)[1];
    shmem->version = ((int*)shmem->buffer)[2];

    xmlDocPtr header;
    if (NULL == (header = xmlReadMemory(shmem->header, strlen(shmem->header), "noname.xml", NULL, 0))) {
        fprintf(stderr, "Failed to parse document\n");
        return -2;
    }

    /* <?xml version="1.0"?>
         <root>
         <version>1</version>
         <running>1</running>
         <application>
           <prank>0</prank>
           <psize>1</psize>
           <nb_vp>1</nb_vp>
           <nb_eu>2</nb_eu>
           <pages_per_nd>0</pages_per_nd>
           <pages_per_vp>0</pages_per_vp>
           <per_eu_properties>
             <dgemm_NN_summa>
               <GEMM>
                 <flops><t>i</t><o>0</o></flops>
                 <kflops><t>d</t><o>4</o></kflops>
               </GEMM>
             </dgemm_NN_summa>
           </per_eu_properties>
           <pages_per_eu>1</pages_per_eu>
         </application>
         </root> */

    root = xmlDocGetRootElement(header); /* <?xml ... > */

    application_node = findNode(root, "application");

    prank_node         = findNode(application_node, "prank");
    psize_node         = findNode(application_node, "psize");
    nb_vp_node         = findNode(application_node, "nb_vp");
    nb_eu_node         = findNode(application_node, "nb_eu");
    pages_per_nd_node  = findNode(application_node, "pages_per_nd");
    pages_per_vp_node  = findNode(application_node, "pages_per_vp");
    pages_per_eu_node  = findNode(application_node, "pages_per_eu");

    shmem->prank            = parse_int(prank_node);
    shmem->psize            = parse_int(psize_node);
    shmem->nb_vp            = parse_int(nb_vp_node);
    shmem->nb_eu            = parse_int(nb_eu_node);
    shmem->nb_node_pages    = parse_int(pages_per_nd_node);
    shmem->nb_vp_pages      = parse_int(pages_per_vp_node);
    shmem->nb_eu_pages      = parse_int(pages_per_eu_node);
    
    per_nd_node = findNode(application_node, "per_nd_properties");
    per_vp_node = findNode(application_node, "per_vp_properties");
    per_eu_node = findNode(application_node, "per_eu_properties");

    shmem->per_eu = init_category(per_eu_node);
    shmem->per_vp = init_category(per_vp_node);
    shmem->per_nd = init_category(per_nd_node);

    shmem->first_vp = shmem->nb_xml_pages + shmem->nb_node_pages;
    shmem->first_eu = shmem->first_vp + shmem->nb_vp * shmem->nb_vp_pages;

    if (argc >= 3) {
        if (!strcmp(argv[2], "data")) {
            do {
                pretty_print(shmem);

                sleep(1);                
                shmem->running = ((int*)shmem->buffer)[1];
            } while(shmem->running);
        } else fprintf(stdout, "%s", shmem->header);
    } else fprintf(stdout, "%s", shmem->header);


    xmlFreeDoc(header);
    return 0;
}
