#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <mpi.h>
#include <curl/curl.h>
#include <parsec.h>
#include <parsec/interfaces/dtd/insert_function.h>
#include <parsec/mca/device/device.h>

#define MAX_STEPS 2

typedef enum { AGENT_IDLE, AGENT_THINKING, AGENT_READY } agent_status_t;

typedef struct {
    char thought[512];
    int step_count;
    int finished;
    agent_status_t status;
} agent_state_t;

// Background I/O
pthread_t io_thread;
pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  queue_cond  = PTHREAD_COND_INITIALIZER;
agent_state_t *request_queue[64];
int queue_size = 0;
int io_thread_running = 1;

// Task Prototypes
static int think_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int poll_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int tool_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int finish_task_body(parsec_execution_stream_t *es, parsec_task_t *task);

void* io_thread_loop(void* arg) {
    while(1) {
        pthread_mutex_lock(&queue_mutex);
        while(queue_size == 0 && io_thread_running) pthread_cond_wait(&queue_cond, &queue_mutex);
        if (!io_thread_running && queue_size == 0) { pthread_mutex_unlock(&queue_mutex); break; }
        agent_state_t *state = request_queue[--queue_size];
        pthread_mutex_unlock(&queue_mutex);

        // Blocking I/O
        CURL *curl = curl_easy_init();
        char post_data[2048];
        snprintf(post_data, sizeof(post_data), "{\"model\":\"tinyllama\",\"prompt\":\"Done\",\"stream\":false}");
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        // Signal completion
        state->status = AGENT_READY;
    }
    return NULL;
}

static int think_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    printf("[Think] Offloading...\n");
    
    state->status = AGENT_THINKING;
    pthread_mutex_lock(&queue_mutex);
    request_queue[queue_size++] = state;
    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);

    parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), poll_task_body, 0, PARSEC_DEV_CPU, "poll_task", 
                           sizeof(agent_state_t*), state, PARSEC_REF, PARSEC_DTD_ARG_END);
    return PARSEC_HOOK_RETURN_DONE;
}

static int poll_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);

    if (state->status == AGENT_READY) {
        printf("[Poll] Result Ready!\n");
        if (state->step_count >= MAX_STEPS) {
            parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), finish_task_body, 0, PARSEC_DEV_CPU, "finish_task", 
                                   sizeof(agent_state_t*), state, PARSEC_REF, PARSEC_DTD_ARG_END);
        } else {
            parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), tool_task_body, 0, PARSEC_DEV_CPU, "tool_task", 
                                   sizeof(agent_state_t*), state, PARSEC_REF, PARSEC_DTD_ARG_END);
        }
    } else {
        // Yield and poll again
        parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), poll_task_body, 0, PARSEC_DEV_CPU, "poll_task", 
                               sizeof(agent_state_t*), state, PARSEC_REF, PARSEC_DTD_ARG_END);
    }
    return PARSEC_HOOK_RETURN_DONE;
}

static int tool_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    printf("[Tool] Step %d\n", state->step_count++);
    parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), think_task_body, 0, PARSEC_DEV_CPU, "think_task", 
                           sizeof(agent_state_t*), state, PARSEC_REF, PARSEC_DTD_ARG_END);
    return PARSEC_HOOK_RETURN_DONE;
}

static int finish_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    printf("[Finish] Done.\n");
    state->finished = 1;
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    curl_global_init(CURL_GLOBAL_ALL);
    pthread_create(&io_thread, NULL, io_thread_loop, NULL);

    parsec_context_t *ctx = parsec_init(-1, NULL, NULL);
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_dtd_create_task_class(tp, "think_task",  sizeof(agent_state_t*), PARSEC_REF, PARSEC_DTD_ARG_END, think_task_body, 0);
    parsec_dtd_create_task_class(tp, "poll_task",   sizeof(agent_state_t*), PARSEC_REF, PARSEC_DTD_ARG_END, poll_task_body, 0);
    parsec_dtd_create_task_class(tp, "tool_task",   sizeof(agent_state_t*), PARSEC_REF, PARSEC_DTD_ARG_END, tool_task_body, 0);
    parsec_dtd_create_task_class(tp, "finish_task", sizeof(agent_state_t*), PARSEC_REF, PARSEC_DTD_ARG_END, finish_task_body, 0);

    parsec_context_add_taskpool(ctx, tp);
    parsec_context_start(ctx);

    agent_state_t* state = calloc(1, sizeof(agent_state_t));
    parsec_dtd_insert_task(tp, think_task_body, 0, PARSEC_DEV_CPU, "think_task", sizeof(agent_state_t*), state, PARSEC_REF, PARSEC_DTD_ARG_END);

    parsec_taskpool_wait(tp);

    pthread_mutex_lock(&queue_mutex);
    io_thread_running = 0;
    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);
    pthread_join(io_thread, NULL);

    parsec_taskpool_free(tp);
    parsec_fini(&ctx);
    curl_global_cleanup();
    MPI_Finalize();
    free(state);
    printf("Step 6 Successful.\n");
    return 0;
}
