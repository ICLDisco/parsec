/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

extern int cuda_arch_probe_run(void);

int main(int argc, char *argv[])
{
    (void)argc; (void)argv;
    return cuda_arch_probe_run();
}
