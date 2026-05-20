#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <curl/curl.h>
#include <parsec.h>
#include <parsec/interfaces/dtd/insert_function.h>
#include <parsec/mca/device/device.h>

#define MAX_STEPS 5

typedef struct {
    char thought[256];
    char action[64];
    char action_input[64];
    int step_count;
    int finished;
} agent_state_t;

size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t realsize = size * nmemb;
    char *buffer = (char *)userdata;
    strncat(buffer, ptr, 4095 - strlen(buffer));
    return realsize;
}

void call_ollama(agent_state_t* state) {
    if (state->step_count >= MAX_STEPS) {
        printf("Safety limit reached!\n");
        state->finished = 1;
        strcpy(state->action, "Final Answer");
        strcpy(state->action_input, "Limit reached");
        return;
    }

    CURL *curl = curl_easy_init();
    if (!curl) return;

    char response_buffer[4096] = {0};
    char prompt[512];
    snprintf(prompt, sizeof(prompt), 
             "Answer with either 'Action:ToolName' or 'Final Answer:Value'. Current thought: %s.", 
             state->thought);

    char post_data[2048];
    snprintf(post_data, sizeof(post_data), 
             "{\"model\":\"tinyllama\",\"prompt\":\"%s\",\"stream\":false}", prompt);

    curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/generate");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response_buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L); // Increase timeout to 5 minutes

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        printf("Curl failed: %s. (Did 5 minute timeout expire?)\n", curl_easy_strerror(res));
        strcpy(state->action, "Final Answer");
        state->finished = 1;
    } else {
        printf("DEBUG: Raw Response received (length: %zu)\n", strlen(response_buffer));
        // Simple manual parsing for the 'response' field
        char *ptr = strstr(response_buffer, "\"response\":\"");
        if (ptr) {
            ptr += 12; // Skip "response":"
            char *end = strchr(ptr, '\"');
            if (end) {
                int len = (int)(end - ptr);
                if (len > 255) len = 255;
                strncpy(state->thought, ptr, len);
                state->thought[len] = '\0';
                printf("LLM said: %s\n", state->thought);
            }
        }
        
        // Very basic parsing for ReAct actions
        if (strstr(state->thought, "Final Answer")) {
             strcpy(state->action, "Final Answer");
             strcpy(state->action_input, "Success");
             state->finished = 1;
        } else {
             // Default to next step
             strcpy(state->action, "Tool");
             strcpy(state->action_input, "Input");
        }
    }
    curl_easy_cleanup(curl);
}

// Forward declarations
static int think_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int tool_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int finish_task_body(parsec_execution_stream_t *es, parsec_task_t *task);

static int think_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    
    printf("[Think] Step %d\n", state->step_count);
    call_ollama(state);
    
    if (state->finished) {
        parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), finish_task_body, 0, PARSEC_DEV_CPU, "finish_task", sizeof(agent_state_t*), &state, PARSEC_VALUE, PARSEC_DTD_ARG_END);
    } else {
        parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), tool_task_body, 0, PARSEC_DEV_CPU, "tool_task", sizeof(agent_state_t*), &state, PARSEC_VALUE, PARSEC_DTD_ARG_END);
    }
    return PARSEC_HOOK_RETURN_DONE;
}

static int tool_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    printf("[Tool] Action: %s, Input: %s\n", state->action, state->action_input);
    state->step_count++;
    parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), think_task_body, 0, PARSEC_DEV_CPU, "think_task", sizeof(agent_state_t*), &state, PARSEC_VALUE, PARSEC_DTD_ARG_END);
    return PARSEC_HOOK_RETURN_DONE;
}

static int finish_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    printf("[Finish] Final Answer: %s\n", state->action_input);
    free(state);
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    curl_global_init(CURL_GLOBAL_ALL);
    parsec_context_t *ctx = parsec_init(-1, NULL, NULL);
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_dtd_create_task_class(tp, "think_task", sizeof(agent_state_t*), PARSEC_VALUE, PARSEC_DTD_ARG_END, think_task_body, 0);
    parsec_dtd_create_task_class(tp, "tool_task", sizeof(agent_state_t*), PARSEC_VALUE, PARSEC_DTD_ARG_END, tool_task_body, 0);
    parsec_dtd_create_task_class(tp, "finish_task", sizeof(agent_state_t*), PARSEC_VALUE, PARSEC_DTD_ARG_END, finish_task_body, 0);

    parsec_context_add_taskpool(ctx, tp);
    parsec_context_start(ctx);

    agent_state_t* initial_state = calloc(1, sizeof(agent_state_t));
    parsec_dtd_insert_task(tp, think_task_body, 0, PARSEC_DEV_CPU, "think_task", sizeof(agent_state_t*), &initial_state, PARSEC_VALUE, PARSEC_DTD_ARG_END);

    parsec_taskpool_wait(tp);
    parsec_taskpool_free(tp);
    parsec_fini(&ctx);
    curl_global_cleanup();
    MPI_Finalize();
    return 0;
}
