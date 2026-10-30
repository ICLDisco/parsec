PaRSEC Profiling How-To {#parsec_profiling_howto}
=======================

PaRSEC can record execution information through the @ref
parsec_public_profiling API and through the PaRSEC INStrumentation
system (PINS). The profiling API provides the storage and event model;
PINS decides which runtime events, such as task execution, are emitted.

Profiling is a build-time feature. The emitted events are forwarded to
independent substrates, any number of which can be active in the same
run:

- file substrates write a trace file: the PaRSEC binary trace format
  (always built) and OTF2 (when found at configure time). They are
  selected with the `profile_format` MCA parameter and require a profile
  filename;
- mirror substrates replay the events into an external tool: NVTX ranges
  for NVIDIA Nsight Systems and Score-P user regions. They do not write a
  PaRSEC profile file and are enabled with their own MCA parameters.

## Build-Time Support

The common option for all profiling substrates is:

```
cmake -S <src> -B <build> -DPARSEC_PROF_TRACE=ON
```

The `configure` wrapper can also enable the common profiling support:

```
./configure --enable-prof-trace
```

`PARSEC_PROF_TRACE=ON` always builds the PaRSEC binary trace format. To
also build OTF2 support, and make OTF2 the default format, configure
with:

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

To add Score-P support, enable both the common profiling layer and the
Score-P mirror:

```
cmake -S <src> -B <build> \
  -DPARSEC_PROF_TRACE=ON \
  -DPARSEC_PROF_TRACE_SCOREP=ON \
  -DPARSEC_SCOREP_ROOT=<scorep-prefix>
```

or, through the `configure` wrapper:

```
./configure --enable-prof-scorep --with-scorep=<scorep-prefix>
```

`PARSEC_SCOREP_ROOT` and `--with-scorep` are optional when `scorep-config`
is in the `PATH` or `SCOREP_ROOT` is set.

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

The PaRSEC binary trace format is always available when profiling is
compiled in, and is the default file format unless OTF2 support is built:

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

OTF2 support is built in addition to the PaRSEC binary trace format, and
becomes the default format, with:

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

## Selecting the File Formats

The `profile_format` MCA parameter is a comma-separated list of the file
formats written when a `profile_filename` is provided. Its default is
`otf2` when OTF2 support is built, and `binary` otherwise. Both formats
can be written in the same run, using the same base name:

```
./app <app-arguments> -- \
  --mca profile_filename traces/run \
  --mca profile_format binary,otf2 \
  --mca mca_pins task_profiler
```

A format that is not available in the build is ignored with a warning.

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

## Score-P

Score-P support mirrors PaRSEC profiling events into Score-P user
regions, through the Score-P user adapter. Each profiling dictionary
entry (for example, each task class) becomes a Score-P region in the
`PaRSEC` group, so PaRSEC tasks appear in Score-P profiles (CUBE) and
traces (OTF2) next to the MPI, CUDA, or other events recorded by
Score-P. No PaRSEC profile file is created.

Build with Score-P support:

```
cmake -S <src> -B <build> \
  -DPARSEC_PROF_TRACE=ON \
  -DPARSEC_PROF_TRACE_SCOREP=ON
```

PaRSEC only needs the Score-P headers at build time: the user adapter
functions are weak symbols in libparsec, so libparsec and applications
that are not linked with Score-P build and run as usual. The Score-P
measurement system is provided by the application, which must be linked
with the Score-P user adapter and, for multi-threaded runs, with pthread
support:

```
scorep --user --thread=pthread --mpp=mpi mpicc -o app app.o -lparsec
```

If Score-P was built as static libraries, the linker has no reason to
pull the user adapter functions into the application, add
`-Wl,-u,SCOREP_User_RegionInit -Wl,-u,SCOREP_User_RegionEnter
-Wl,-u,SCOREP_User_RegionEnd -Wl,-u,SCOREP_User_RegionSetGroup` (with
a leading `_` on the symbol names on macOS) to the link line.

Collect Score-P measurements by enabling the Score-P mirror and the PINS
task profiler:

```
SCOREP_ENABLE_TRACING=true \
./app <app-arguments> -- \
  --mca profile_scorep 1 \
  --mca mca_pins task_profiler
```

If `profile_scorep` is set but the application is not linked with the
Score-P user adapter, PaRSEC prints a warning and disables the Score-P
mirror. Score-P filter files apply to the PaRSEC regions as to any other
user region and can be used to select the recorded task classes.

Score-P requires the regions to be properly nested on each thread. By
default, only the events traced by a thread into its own profiling stream
(worker threads and the communication thread) are mirrored. Events traced
on behalf of another stream, such as the events of the GPU streams emitted
by the GPU manager thread, are skipped. They can be included with
`--mca profile_scorep_all_streams 1`. When an event ends while events
started after it are still open on the same thread, those enclosed
regions are closed early to keep the Score-P measurement consistent, and
a warning is printed once.

## Troubleshooting Empty Traces

If a profiling run completes but the trace is empty, check the following
items:

- The build was configured with `PARSEC_PROF_TRACE=ON`.
- NVTX runs were also configured with `PARSEC_PROF_TRACE_NVTX=ON`.
- Score-P runs were also configured with `PARSEC_PROF_TRACE_SCOREP=ON`,
  and the application was linked with `scorep --user`.
- File-based runs provided `--mca profile_filename <name>`, and
  `profile_format` names a format available in the build.
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

## Adding a Substrate

The profiling core (`parsec/profiling.c`) implements the public profiling
API. It owns the dictionary, the streams, the information key/values and
the time base, and forwards every event to the active substrates. A
substrate implements the `parsec_profiling_substrate_t` interface defined
in `parsec/profiling_internal.h` (`profiling_binary.c`,
`profiling_otf2.c`, `profiling_nvtx.c` and `profiling_scorep.c` are the
existing implementations) and is added to the list of substrates at the
top of `parsec/profiling.c`. A substrate activated after keys or streams
were created receives them when it is activated, so substrates do not
depend on the order in which the profiling calls are made.
