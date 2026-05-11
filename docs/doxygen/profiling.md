PaRSEC Profiling How-To {#parsec_profiling_howto}
=======================

PaRSEC can record execution information through the @ref
parsec_public_profiling API and through the PaRSEC INStrumentation
system (PINS). The profiling API provides the storage and event model;
PINS decides which runtime events, such as task execution, are emitted.

Profiling is a build-time feature. A build can provide one file-based
profiling substrate, either the PaRSEC binary trace format or OTF2, and
can also mirror emitted events to NVTX ranges for NVIDIA Nsight Systems.
At runtime, file-based substrates require a profile filename. NVTX does
not write a PaRSEC profile file and is enabled with its own MCA parameter.

## Build-Time Support

The common option for all profiling substrates is:

```
cmake -S <src> -B <build> -DPARSEC_PROF_TRACE=ON
```

The `configure` wrapper can also enable the common profiling support:

```
./configure --enable-prof-trace
```

By default, `PARSEC_PROF_TRACE=ON` selects the PaRSEC binary trace
format. To select OTF2 instead, configure with:

```
cmake -S <src> -B <build> \
  -DPARSEC_PROF_TRACE=ON \
  -DPARSEC_PROF_TRACE_SYSTEM=OTF2 \
  -DOTF2_DIR=<otf2-prefix>
```

OTF2 support requires a distributed MPI build and an OTF2 installation
that provides `otf2-config`.

To add NVTX support, enable both the common profiling layer and the NVTX
mirror:

```
cmake -S <src> -B <build> \
  -DPARSEC_PROF_TRACE=ON \
  -DPARSEC_PROF_TRACE_NVTX=ON \
  -DPARSEC_NVTX_ROOT=<cuda-or-nvtx-prefix>
```

or, through the `configure` wrapper:

```
./configure --enable-prof-nvtx --with-nvtx=<cuda-or-nvtx-prefix>
```

`PARSEC_NVTX_ROOT` and `--with-nvtx` are optional when the NVTX headers
can be found from the CUDA Toolkit or the system include paths.

## Runtime Instrumentation

The profiling layer only records events that are emitted by the
application or by selected runtime instrumentation modules. To collect
task events from the PaRSEC runtime, enable the PINS task profiler:

```
--mca mca_pins task_profiler
```

The task profiler enables `release_deps` and `exec_begin` events by
default. The selected task-profiler event families can be changed with:

```
--mca pins_task_profiler_event exec,release_deps,complete_exec
```

Accepted task-profiler event family names include `select`,
`prepare_input`, `release_deps`, `activate_cb`, `data_flush`, `exec`,
`complete_exec`, and `schedule`.

If the application has its own command line parser, PaRSEC MCA arguments
are commonly passed after the application's `--` separator:

```
./app <app-arguments> -- --mca mca_pins task_profiler
```

Some applications pass MCA arguments directly to PaRSEC and do not need
the separator.

## PaRSEC Binary Trace Format

The PaRSEC binary trace format is the default file-based substrate when
profiling is compiled in:

```
cmake -S <src> -B <build> -DPARSEC_PROF_TRACE=ON
```

Collect a trace by providing a base profile filename and enabling the
task profiler:

```
./app <app-arguments> -- \
  --mca profile_filename run \
  --mca mca_pins task_profiler
```

The binary substrate creates one `.prof` file per process, using the
profile filename as the base name, for example `run-0.prof`,
`run-1.prof`, and so on.

The binary files can be converted to the HDF5/PTT format with the
profiling tools:

```
<build>/tools/profiling/python/profile2h5.py \
  --output run.h5 \
  run-*.prof
```

The resulting HDF5 file can be inspected with the PaRSEC profiling
Python tools and viewers.

## OTF2

OTF2 is selected at configure time instead of the PaRSEC binary trace
format:

```
cmake -S <src> -B <build> \
  -DPARSEC_PROF_TRACE=ON \
  -DPARSEC_PROF_TRACE_SYSTEM=OTF2 \
  -DOTF2_DIR=<otf2-prefix>
```

Collect an OTF2 trace by providing a profile filename and enabling the
task profiler:

```
./app <app-arguments> -- \
  --mca profile_filename traces/run \
  --mca mca_pins task_profiler
```

For OTF2, `profile_filename` names the archive path used by the OTF2
writer. The parent directory must already exist and must be writable by
the application. The generated archive can be opened with standard OTF2
trace analysis tools.

## NVTX and NVIDIA Nsight Systems

NVTX support mirrors PaRSEC profiling events into an NVTX domain named
`PaRSEC`. It is intended for use with NVIDIA Nsight Systems and can be
used without creating a PaRSEC profile file.

Build with NVTX support:

```
cmake -S <src> -B <build> \
  -DPARSEC_PROF_TRACE=ON \
  -DPARSEC_PROF_TRACE_NVTX=ON
```

or:

```
./configure --enable-prof-nvtx
```

Collect NVTX events with Nsight Systems by enabling the NVTX runtime
mirror and the PINS task profiler:

```
nsys profile --trace=nvtx,cuda -o parsec_report \
  ./app <app-arguments> -- \
  --mca profile_nvtx 1 \
  --mca mca_pins task_profiler
```

Environment variables are often convenient when the profiled command
line is already complex:

```
PARSEC_MCA_profile_nvtx=1 \
PARSEC_MCA_mca_pins=task_profiler \
nsys profile --trace=nvtx,cuda -o parsec_report ./app <app-arguments>
```

No `profile_filename` is required for NVTX-only collection. To collect
both NVTX ranges and the selected file-based substrate in the same run,
enable `profile_nvtx` and also provide `profile_filename`:

```
nsys profile --trace=nvtx,cuda -o parsec_report \
  ./app <app-arguments> -- \
  --mca profile_nvtx 1 \
  --mca profile_filename run \
  --mca mca_pins task_profiler
```

## Troubleshooting Empty Traces

If a profiling run completes but the trace is empty, check the following
items:

- The build was configured with `PARSEC_PROF_TRACE=ON`.
- NVTX runs were also configured with `PARSEC_PROF_TRACE_NVTX=ON`.
- File-based runs provided `--mca profile_filename <name>`.
- Task event collection enabled the PINS task profiler with
  `--mca mca_pins task_profiler`.
- The selected application path actually executed PaRSEC tasks.
- The output directory for binary or OTF2 traces already exists and is
  writable.
- Nsight Systems was instructed to collect NVTX events with
  `--trace=nvtx` or a trace list that includes `nvtx`.

When using Nsight Systems, `nsys status --environment` can also confirm
whether the current system supports the requested CUDA and tracing
features.
