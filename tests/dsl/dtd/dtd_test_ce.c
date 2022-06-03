#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "parsec/parsec_comm_engine.h"

#include "parsec/runtime.h"

#define ACTIVE_MESSAGE_FROM_0_TAG 2
#define ACTIVE_MESSAGE_FROM_1_TAG 3
#define NOTIFY_ABOUT_GET_FROM_0_TAG 4
#define NOTIFY_ABOUT_PUT_FROM_0_TAG 5
#define NOTIFY_ABOUT_MEM_HANDLE_FROM_1_TAG 6

int
get_end(parsec_comm_engine_t *ce,
        parsec_ce_mem_reg_handle_t lreg,
        ptrdiff_t ldispl,
        parsec_ce_mem_reg_handle_t rreg,
        ptrdiff_t rdispl,
        size_t size,
        int remote,
        void *cb_data);

int
put_end(parsec_comm_engine_t *ce,
        parsec_ce_mem_reg_handle_t lreg,
        ptrdiff_t ldispl,
        parsec_ce_mem_reg_handle_t rreg,
        ptrdiff_t rdispl,
        size_t size,
        int remote,
        void *cb_data);

int
put_end_ack(parsec_comm_engine_t *ce,
            parsec_ce_tag_t tag,
            void *msg,
            size_t msg_size,
            int src,
            void *cb_data);

int counter = 0;
int my_rank;

/* Tag 0 for float message */
int
callback_tag_2(parsec_comm_engine_t *ce,
               parsec_ce_tag_t tag,
               void *msg,
               size_t msg_size,
               int src,
               void *cb_data)
{
    (void) ce; (void) cb_data;
    printf("[%d] In callback for tag %"PRIu64", message sent from %d size: %zu message: ", my_rank, tag, src, msg_size);

    int i, total = msg_size/sizeof(int);

    int *buffer = (int *)msg;
    printf("[");
    for(i = 0; i < total; i++) {
        if(i == total - 1) {
            printf("%d", buffer[i]);
        } else {
            printf("%d,", buffer[i]);
        }
    }
    printf("]\n");
    counter++;

    return 1;
}

/* Tag 1 for int message */
int
callback_tag_3(parsec_comm_engine_t *ce,
               parsec_ce_tag_t tag,
               void *msg,
               size_t msg_size,
               int src,
               void *cb_data)
{
    (void) ce; (void) cb_data;
    printf("[%d] In callback for tag %"PRIu64", message sent from %d size: %zu message: ", my_rank, tag, src, msg_size);

    int i, total = msg_size/sizeof(float);

    float *buffer = (float *)msg;
    printf("[");
    for(i = 0; i < total; i++) {
        if(i == total - 1) {
            printf("%f", buffer[i]);
        } else {
            printf("%f,", buffer[i]);
        }
    }
    printf("]\n");
    counter++;

    return 1;
}

typedef struct handshake_info_s {
    int       buf_size;
    uintptr_t cb_fn;
} handshake_info_t;


/* Active Message for GET notification.
 * This function will be triggered in rank 1 after rank 0
 * sends an active message to 1 informing about a GET that
 * 1 needs to complete.
 */
int
notify_about_get(parsec_comm_engine_t *ce,
                 parsec_ce_tag_t tag,
                 void *msg,
                 size_t msg_size,
                 int src,
                 void *cb_data)
{
    (void) tag; (void) cb_data; (void) msg_size;
    assert(my_rank == 1);

    handshake_info_t *GET_activation_message = (handshake_info_t *) msg;

    /* We have the remote memory_handle.
     * Let's allocate the local memory_handle
     * and let's start the GET.
     */
    parsec_ce_mem_reg_handle_t rank_1_memory_handle;
    size_t rank_1_memory_handle_size;
    /* GET operation will store the actual data in the following buffer */
    int *receive_buf = malloc(sizeof(int) * GET_activation_message->buf_size);

    if(ce->capabilites.supports_noncontiguous_datatype) {
        parsec_datatype_t *datatype = malloc(sizeof(parsec_datatype_t));
        parsec_type_create_contiguous(GET_activation_message->buf_size, parsec_datatype_int_t, datatype);
        ce->mem_register(receive_buf, PARSEC_MEM_TYPE_NONCONTIGUOUS,
                         1, *datatype,
                         -1,
                         &rank_1_memory_handle, &rank_1_memory_handle_size);
    } else {
         ce->mem_register(receive_buf, PARSEC_MEM_TYPE_CONTIGUOUS,
                          -1, PARSEC_DATATYPE_NULL,
                          sizeof(int) * GET_activation_message->buf_size,
                          &rank_1_memory_handle, &rank_1_memory_handle_size);
    }

    parsec_ce_mem_reg_handle_t rank_0_memory_handle = (parsec_ce_mem_reg_handle_t)(((char *)GET_activation_message) + sizeof(handshake_info_t));

    /* Let's start the GET */
    ce->get(ce, rank_1_memory_handle, 0, rank_0_memory_handle, 0,
            0, src,
            get_end, (void *) ce,
            GET_activation_message->cb_fn, rank_0_memory_handle, ce->get_mem_handle_size());

    counter++;

    return 1;
}

/* This will be called in rank 1 when GET is done */
int
get_end(parsec_comm_engine_t *ce,
        parsec_ce_mem_reg_handle_t lreg,
        ptrdiff_t ldispl,
        parsec_ce_mem_reg_handle_t rreg,
        ptrdiff_t rdispl,
        size_t size,
        int remote,
        void *cb_data)
{
    (void) ldispl; (void) rdispl; (void) size; (void) remote; (void) cb_data; (void) rreg;

    void *mem;
    int mem_size;
    int count;
    parsec_datatype_t datatype;

    ce->mem_retrieve(lreg, &mem, &datatype, &count);

    parsec_type_size(datatype, &mem_size);

    printf("[%d] GET is over, message:\n[", my_rank);
    int *receive_buf = (int *)mem;
    int i;
    int total = (int)(mem_size/(sizeof(int)));
    for(i = 0; i < total; i++) {
        if(i == total-1) {
            printf("%d", receive_buf[i]);
        } else {
            printf("%d,", receive_buf[i]);
        }
    }
    printf("]\n");

    parsec_ce_mem_reg_handle_t rank_1_memory_handle = lreg;
    ce->mem_unregister(&rank_1_memory_handle);

    free(receive_buf);

    counter++;

    return 1;
}

/* This funciton was passed from rank 0 to rank 1 as the notification
 * function to be called when the GET is over for clean up on rank 0.
 * The memory_handle of rank 0 was also sent to rank 1 to be sent back
 * as the callback data.
 */
int
get_end_ack(parsec_comm_engine_t *ce,
            parsec_ce_tag_t tag,
            void *msg,
            size_t msg_size,
            int src,
            void *cb_data)
{
    (void) tag; (void) msg_size; (void) src; (void) cb_data;
    parsec_ce_mem_reg_handle_t rank_0_memory_handle = (parsec_ce_mem_reg_handle_t) msg;

    /* cb_data is the data passed while this function was registered with the lower
     * level comm. engine. */
    void *mem;
    int count;
    parsec_datatype_t datatype;

    ce->mem_retrieve(rank_0_memory_handle, &mem, &datatype, &count);

    printf("[%d] Notification of GET over received\n", my_rank);
    int *send_buf = (int *)mem;

    free(send_buf);

    ce->mem_unregister(&rank_0_memory_handle);

    counter++;

    return 1;
}

/* Rank 0 send an active message to rank 1
 * notifying about a PUT
 */
int
notify_about_put(parsec_comm_engine_t *ce,
                 parsec_ce_tag_t tag,
                 void *msg,
                 size_t msg_size,
                 int src,
                 void *cb_data)
{
    (void) tag; (void) src; (void) cb_data; (void) msg_size;
    assert(my_rank == 1);

    handshake_info_t *PUT_activation_message = (handshake_info_t *) msg;

    parsec_ce_mem_reg_handle_t rank_1_memory_handle;
    size_t rank_1_memory_handle_size;
    int *receive_buf = malloc(sizeof(int) * PUT_activation_message->buf_size);

    /* We have the remote mem_reg_handle.
     * Let's allocate the local mem_reg_handle
     * and send both to other side to start a PUT.
     */

    if(ce->capabilites.supports_noncontiguous_datatype) {
        parsec_datatype_t *datatype = malloc(sizeof(parsec_datatype_t));
        parsec_type_create_contiguous(PUT_activation_message->buf_size, parsec_datatype_int_t, datatype);
        ce->mem_register(receive_buf, PARSEC_MEM_TYPE_NONCONTIGUOUS,
                         1, *datatype,
                         -1,
                         &rank_1_memory_handle, &rank_1_memory_handle_size);
    } else {
         ce->mem_register(receive_buf, PARSEC_MEM_TYPE_CONTIGUOUS,
                          -1, PARSEC_DATATYPE_NULL,
                          sizeof(int) * PUT_activation_message->buf_size,
                          &rank_1_memory_handle, &rank_1_memory_handle_size);
    }

    handshake_info_t handshake_info;
    handshake_info.buf_size = 0;
    handshake_info.cb_fn = (uintptr_t) put_end_ack;

    /* Rank 1 has rank 0's memory_handle, it will pack both 0's and it's own
     * memory_handle and send it to rank 0. After receiving this message, 0 will
     * be able to perform a PUT.
     */
    int PUT_forward_mem_handle_message_size = sizeof(handshake_info_t) + (2 * ce->get_mem_handle_size());
    void *PUT_forward_mem_handle_message = malloc(PUT_forward_mem_handle_message_size);

    /* pack the handshake_info_t */
    memcpy( PUT_forward_mem_handle_message,
            &handshake_info,
            sizeof(handshake_info_t) );
    /* pack rank 0's memory_handle */
    memcpy( ((char *) PUT_forward_mem_handle_message) + sizeof(handshake_info_t),
            ((char *) PUT_activation_message) + sizeof(handshake_info_t),
            ce->get_mem_handle_size() );
    /* pack rank 1's memory_handle */
    memcpy( ((char *) PUT_forward_mem_handle_message) + sizeof(handshake_info_t) + ce->get_mem_handle_size(),
            rank_1_memory_handle,
            ce->get_mem_handle_size() );

    ce->send_am(ce, NOTIFY_ABOUT_MEM_HANDLE_FROM_1_TAG, 0, PUT_forward_mem_handle_message, PUT_forward_mem_handle_message_size);

    free(PUT_forward_mem_handle_message);

    counter++;

    return 1;
}

/* This function is called in rank 0 when rank 1 has received the activation message for
 * a PUT and is prepared the buffers for rank 0 to complete the PUT.
 */
int
put_ack_am(parsec_comm_engine_t *ce,
           parsec_ce_tag_t tag,
           void *msg,
           size_t msg_size,
           int src,
           void *cb_data)
{
    (void) tag; (void) cb_data; (void) msg_size;
    assert(my_rank == 0);

    handshake_info_t *PUT_forward_mem_handle_message = (handshake_info_t *) msg;

    parsec_ce_mem_reg_handle_t rank_0_memory_handle = ((char *)PUT_forward_mem_handle_message) +
                                                      sizeof(handshake_info_t);
    parsec_ce_mem_reg_handle_t rank_1_memory_handle = ((char *)PUT_forward_mem_handle_message) +
                                                      sizeof(handshake_info_t) +
                                                      ce->get_mem_handle_size();

    printf("[%d] Received the remote mem_reg_handle and now can start the PUT\n", my_rank);

    ce->put(ce, rank_0_memory_handle, 0,
            rank_1_memory_handle, 0,
            0, src,
            put_end, NULL,
            (uintptr_t) PUT_forward_mem_handle_message->cb_fn, (void *)rank_1_memory_handle, ce->get_mem_handle_size());

    counter++;

    return 1;
}

/* This function will be called once the PUT is over in rank 0 */
int
put_end(parsec_comm_engine_t *ce,
        parsec_ce_mem_reg_handle_t lreg,
        ptrdiff_t ldispl,
        parsec_ce_mem_reg_handle_t rreg,
        ptrdiff_t rdispl,
        size_t size,
        int remote,
        void *cb_data)
{
    (void) ldispl; (void) rdispl; (void) size; (void) remote; (void) cb_data; (void) rreg;
    printf("[%d] PUT is finished\n", my_rank);

    parsec_ce_mem_reg_handle_t rank_0_memory_handle = lreg;

    void *send_buf;

    int count;
    parsec_datatype_t datatype;

    ce->mem_retrieve(rank_0_memory_handle, &send_buf, &datatype, &count);

    free(send_buf);

    ce->mem_unregister(&rank_0_memory_handle);

    counter++;

    return 1;
}

/* This function is called to notify rank 1 that PUT by 0 is over */
int
put_end_ack(parsec_comm_engine_t *ce,
            parsec_ce_tag_t tag,
            void *msg,
            size_t msg_size,
            int src,
            void *cb_data)
{
    (void) tag; (void) msg_size; (void) src; (void)cb_data;
    parsec_ce_mem_reg_handle_t rank_1_memory_handle = (parsec_ce_mem_reg_handle_t) msg;

    void *mem;
    int mem_size;

    int count;
    parsec_datatype_t datatype;
    ce->mem_retrieve(rank_1_memory_handle, &mem, &datatype, &count);

    parsec_type_size(datatype, &mem_size);

    printf("[%d] PUT is over, message:\n[", my_rank);

    int *receive_buf = (int *)mem;
    int i;
    int total = (int)(mem_size/(sizeof(int)));
    for(i = 0; i < total; i++) {
        if(i == total-1) {
            printf("%d", receive_buf[i]);
        } else {
            printf("%d,", receive_buf[i]);
        }
    }
    printf("]\n");

    free(receive_buf);

    ce->mem_unregister(&rank_1_memory_handle);

    counter++;

    return 1;
}

int main(int argc, char **argv)
{
    int rank, world;
    int i;
    int test_GET = 1;
    int test_PUT = 1;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    my_rank = rank;

    parsec_comm_engine_t *ce = parsec_comm_engine_init(NULL);

    if( world != 2 ) {
        printf("World is too small, too bad! Buh-bye");
        return 0;
    }

    ce->tag_register(ACTIVE_MESSAGE_FROM_0_TAG, callback_tag_2, ce, 4096);
    ce->tag_register(ACTIVE_MESSAGE_FROM_1_TAG, callback_tag_3, ce, 4096);

    /* Active message for GET notification */
    ce->tag_register(NOTIFY_ABOUT_GET_FROM_0_TAG, notify_about_get, ce, 4096);

    /* Active message for PUT notification */
    ce->tag_register(NOTIFY_ABOUT_PUT_FROM_0_TAG, notify_about_put, ce, 4096);
    ce->tag_register(NOTIFY_ABOUT_MEM_HANDLE_FROM_1_TAG, put_ack_am, ce, 4096);

    if(ce->capabilites.sided == 1) {
        /* This is true onesided and we need to register more tags for notifications */
        ce->tag_register((parsec_ce_tag_t)(uintptr_t)get_end_ack, get_end_ack, ce, 4096);
        ce->tag_register((parsec_ce_tag_t)(uintptr_t)put_end_ack, put_end_ack, ce, 4096);
    }

    /* To make sure all the ranks have the tags registered */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Testing active message */
    if(rank == 0) {
        int   *intbuffer = NULL;
        intbuffer = malloc(3*sizeof(int));
        intbuffer[0] = 10;
        intbuffer[1] = 11;
        intbuffer[2] = 12;

        printf("[%d] Sending same active message twice to 1, message: [%d,%d,%d]\n",
                my_rank, intbuffer[0], intbuffer[1], intbuffer[2]);

        ce->send_am(ce, ACTIVE_MESSAGE_FROM_0_TAG, 1, intbuffer, 3*sizeof(int));
        ce->send_am(ce, ACTIVE_MESSAGE_FROM_0_TAG, 1, intbuffer, 3*sizeof(int));

        free(intbuffer);
    }
    if(rank == 1) {
        while(counter != 2) {
            ce->progress(ce);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    counter = 0;
    printf("-------------------------------------\n");

    if(rank == 1) {
        float *floatbuffer = NULL;
        floatbuffer = malloc(2*sizeof(float));
        floatbuffer[0] = 9.5;
        floatbuffer[1] = 19.5;

        printf("[%d] Sending same active message twice to 0, message: [%f,%f]\n",
                my_rank, floatbuffer[0], floatbuffer[1]);

        ce->send_am(ce, ACTIVE_MESSAGE_FROM_1_TAG, 0, floatbuffer, 2*sizeof(float));
        ce->send_am(ce, ACTIVE_MESSAGE_FROM_1_TAG, 0, floatbuffer, 2*sizeof(float));

        free(floatbuffer);
    }
    if(rank == 0) {
        while(counter != 2) {
            ce->progress(ce);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    counter = 0;
    printf("-------------------------------------\n");

    if(test_GET) {
        /* Let's test Get from 1 -> 0 (1 gets from 0) */
        if(rank == 0) {
            parsec_ce_mem_reg_handle_t rank_0_memory_handle;
            size_t rank_0_memory_handle_size;
            int buf_size = 9;
            int *send_buf = malloc(sizeof(int) * buf_size);

            for(i = 0; i < buf_size; i++) {
                send_buf[i] = i;
            }

            /* Registering a memory with a mem_reg_handle */
            if(ce->capabilites.supports_noncontiguous_datatype) {
                parsec_datatype_t *datatype = malloc(sizeof(parsec_datatype_t));
                parsec_type_create_contiguous(buf_size, parsec_datatype_int_t, datatype);
                ce->mem_register(send_buf, PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                 1, *datatype,
                                 -1,
                                 &rank_0_memory_handle, &rank_0_memory_handle_size);
            } else {
                 ce->mem_register(send_buf, PARSEC_MEM_TYPE_CONTIGUOUS,
                                  -1, PARSEC_DATATYPE_NULL,
                                  buf_size,
                                  &rank_0_memory_handle, &rank_0_memory_handle_size);
            }

            printf("[%d] Starting a GET (1 will GET from 0), message:\n[", my_rank);
            for(i = 0; i < buf_size; i++) {
                if(i == buf_size - 1) {
                    printf("%d", send_buf[i]);
                } else {
                    printf("%d,", send_buf[i]);
                }
            }
            printf("]\n");

            handshake_info_t handshake_info;
            handshake_info.buf_size = buf_size;
            handshake_info.cb_fn    = (uintptr_t) get_end_ack;

            /* Actual message sent from 0 to 1 will contain handshake_info_t and
             * a copy of the local_memory_handle of rank 0.
             */
            int GET_activation_message_size = sizeof(handshake_info_t) + rank_0_memory_handle_size;
            void *GET_activation_message = malloc(GET_activation_message_size);
            memcpy( GET_activation_message,
                    &handshake_info,
                    sizeof(handshake_info_t) );
            memcpy( ((char *)GET_activation_message) + sizeof(handshake_info_t),
                    rank_0_memory_handle,
                    rank_0_memory_handle_size );

            /* 0 lets 1 know that it has some data for 1 to get */
            ce->send_am(ce, NOTIFY_ABOUT_GET_FROM_0_TAG, 1, GET_activation_message, GET_activation_message_size);

            free(GET_activation_message);

            while(counter != 1) {
                ce->progress(ce);
            }
        }

        if(rank == 1) {
            while(counter != 2) {
                ce->progress(ce);
            }
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    counter = 0;

    if(test_PUT) {
        printf("-------------------------------------\n");
        /* Let's test PUT from 0 -> 1 (0 puts in 1) */
        if(rank == 0) {
            parsec_ce_mem_reg_handle_t rank_0_memory_handle;
            size_t rank_0_memory_handle_size;

            int buf_size = 9;
            int *send_buf = malloc(sizeof(int) * buf_size);
            for(i = 0; i < buf_size; i++) {
                send_buf[i] = i * 2;
            }

            if(ce->capabilites.supports_noncontiguous_datatype) {
                parsec_datatype_t *datatype = malloc(sizeof(parsec_datatype_t));
                parsec_type_create_contiguous(buf_size, parsec_datatype_int_t, datatype);
                ce->mem_register(send_buf, PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                 1, *datatype,
                                 -1,
                                 &rank_0_memory_handle, &rank_0_memory_handle_size);
            } else {
                 ce->mem_register(send_buf, PARSEC_MEM_TYPE_CONTIGUOUS,
                                  -1, PARSEC_DATATYPE_NULL,
                                  buf_size,
                                  &rank_0_memory_handle, &rank_0_memory_handle_size);
            }

            printf("[%d] Starting a PUT (0 will PUT in 1), message:\n[", my_rank);
            for(i = 0; i < buf_size; i++) {
                if(i == buf_size -1) {
                    printf("%d", send_buf[i]);
                } else {
                    printf("%d,", send_buf[i]);
                }
            }
            printf("]\n");

            handshake_info_t handshake_info;
            handshake_info.buf_size = buf_size;
            handshake_info.cb_fn    = 0;

            int PUT_activation_message_size = sizeof(handshake_info_t) + rank_0_memory_handle_size;
            void *PUT_activation_message = malloc(PUT_activation_message_size);

            memcpy( PUT_activation_message,
                    &handshake_info,
                    sizeof(handshake_info_t) );
            memcpy( ((char *)PUT_activation_message) + sizeof(handshake_info_t),
                    rank_0_memory_handle,
                    rank_0_memory_handle_size );

            /* 0 lets 1 know that it has the data for 1 */
            ce->send_am(ce, NOTIFY_ABOUT_PUT_FROM_0_TAG, 1, PUT_activation_message, PUT_activation_message_size);

            free(PUT_activation_message);

            while(counter != 2) {
                ce->progress(ce);
            }
        }

        if(rank == 1) {
            while(counter != 2)
                ce->progress(ce);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ce->tag_unregister(ACTIVE_MESSAGE_FROM_0_TAG);
    ce->tag_unregister(ACTIVE_MESSAGE_FROM_1_TAG);
    ce->tag_unregister(NOTIFY_ABOUT_GET_FROM_0_TAG);
    ce->tag_unregister(NOTIFY_ABOUT_PUT_FROM_0_TAG);
    ce->tag_unregister(NOTIFY_ABOUT_MEM_HANDLE_FROM_1_TAG);

    parsec_comm_engine_fini(ce);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
