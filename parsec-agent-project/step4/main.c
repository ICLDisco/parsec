#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <parsec.h>
#include <parsec/interfaces/dtd/insert_function.h>
#include <parsec/mca/device/device.h>

typedef struct {
    char thought[256];
    char action[64];
    char action_input[64];
    int step_count;
    int finished;
} agent_state_t;

// Stub LLM function
void stub_llm(agent_state_t* state) {
    if (state->step_count < 2) {
        strcpy(state->thought, "I need to check the temperature.");
        strcpy(state->action, "TemperatureTool");
        strcpy(state->action_input, "London");
    } else {
        strcpy(state->thought, "The temperature is 20C.");
        strcpy(state->action, "Final Answer");
        strcpy(state->action_input, "20C");
        state->finished = 1;
    }
}

// Forward declarations
static int think_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int tool_task_body(parsec_execution_stream_t *es, parsec_task_t *task);
static int finish_task_body(parsec_execution_stream_t *es, parsec_task_t *task);

static int think_task_body(parsec_execution_stream_t *es, parsec_task_t *task) {
    agent_state_t* state;
    parsec_dtd_unpack_args(task, &state);
    
    printf("[Think] Step %d\n", state->step_count);
    stub_llm(state);
    
    if (strcmp(state->action, "Final Answer") == 0) {
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
    printf("[Finish] Final Answer: %s (Steps: %d)\n", state->action_input, state->step_count);
    free(state);
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
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
    MPI_Finalize();
    return 0;
}
