#ifndef graph_h
#define graph_h

typedef struct {
    char              *task_id;
    char              *task_name;
    char              *task_parameters;
    char              *node;
    char              *vp;
    char              *thread;
    unsigned long long taskpool_id;
    int                priority;
} node_info_t;

#define STATUS_RUNNING 1
#define STATUS_ENABLED 2
#define STATUS_READY   4
#define STATUS_DONE    8

#define NID ((unsigned int)-1)

/** Underlying Graph Description */
unsigned int add_node(node_info_t *info);
unsigned int find_node_by_task_id(const char *task_id);
unsigned int find_node_by_task_name_and_parameters(const char *name, const char *parameters);
unsigned int find_node_by_task_name_and_taskpool_id(const char *name, unsigned long long oid);
void get_node_info(unsigned int node, node_info_t *info);
unsigned int add_edge(unsigned int src, unsigned int dst);

int add_nodes_from_dotfile(const char *filename, int fileidx, char **types, int nbtypes);
int add_edges_from_dotfile(const char *filename);
void add_key_nodes(void);

void set_node_status(unsigned int node, int status_bits);
void clear_node_status(unsigned int node, int status_bits);
void update_neighbors_status(unsigned int node);

/** Rendering interface */
void graphInit(void);
int graphFini(void);
void graphRenderStatusAtDistance(int status_bits, int distance,
                                 char **result, unsigned int *length);

void persistentGraphLayoutEntireGraph(void);
void persistentGraphRender(char **result, unsigned int *length);
void persistentGraphClose(void);

#endif
