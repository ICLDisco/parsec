#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

int main (void)
{
    uint64_t bobi = 64;
    uint64_t bibi = 12;
    int tx = 1;
    tx ++;
    bibi = tx;
    bobi = bibi;

    printf("test uint64_t : %" PRIu64 "\n", bobi);   
    return 0;
}
