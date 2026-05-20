<!--
Copyright (c) 2026 NVIDIA Corporation.  All rights reserved.
-->

PaRSEC Binding How-To {#parsec_binding_howto}
=====================

PaRSEC can bind its runtime threads to the CPU resources made available
to the process. Binding is controlled with MCA parameters, and, when
PaRSEC is built with HWLOC support, it adapts to bindings inherited from
MPI launchers, batch schedulers, and cgroup-based resource managers.

This page describes the binding choices available to users, how to
display the resulting binding, and how PaRSEC interprets resource indexes
when the launcher restricts the process to a subset of the node.

## Basic Terms

**Processing resource** means the object PaRSEC is allowed to bind to. By
default this is a physical core. With `runtime_allow_pu` enabled, the
resource is a hardware PU (processing unit), such as a hardware thread.

**Allowed cpuset** is the set of resources visible to the PaRSEC process
after applying the process binding inherited from the launcher. For
example, `mpirun --map-by socket --bind-to socket` may give each MPI rank
a different socket. Each rank then has a different allowed cpuset.

**Relative resource index** is the index used by most PaRSEC binding
options. Non-negative indexes are relative to the allowed cpuset, not to
the whole node. If a rank is allowed to use physical cores 8 through 15,
then resource `0` means the first resource in that allowed set.

**Absolute resource index** is used only where explicitly documented. For
`bind_comm`, negative values force an absolute physical core selection.

**Virtual process map (VP map)** groups worker threads into virtual
processes and describes the candidate binding resources for each thread.
Unless the user provides `bind_map`, the VP map supplies the default
worker-thread placement.

## Displaying Bindings

The main MCA parameters for inspecting binding are:

| MCA parameter | Purpose |
| --- | --- |
| `runtime_report_bindings` | Print the allowed, used, and free CPU masks for each PaRSEC process, plus per-thread binding messages. |
| `runtime_vpmap` with `display:` | Print the VP map, including each VP thread candidate mask. For example, `display:flat` or `display:hwloc`. |
| `runtime_warn_slow_binding` | Enable warnings for binding configurations that may perform poorly. A value of `0` disables these warnings; otherwise the value limits distributed overlap checks to jobs with no more than that many PaRSEC nodes. |

Examples:

```
./app -- --mca runtime_report_bindings 1
./app -- --mca runtime_report_bindings 1 --mca runtime_vpmap display:flat
./app -- --mca runtime_report_bindings 1 --mca runtime_vpmap display:hwloc
```

Applications that pass PaRSEC MCA options directly do not need the extra
`--` separator:

```
./app --mca runtime_report_bindings 1
```

The same parameters can be set through environment variables:

```
PARSEC_MCA_runtime_report_bindings=1 \
PARSEC_MCA_runtime_vpmap=display:flat \
./app
```

They can also be set persistently in the user MCA parameter file
`${HOME}/.parsec/mca-params.conf`:

```
# ${HOME}/.parsec/mca-params.conf
runtime_report_bindings = 1
runtime_vpmap = display:flat
```

Command-line MCA values override values from the user MCA parameter file.

The `runtime_report_bindings` output contains three process masks:

| Mask | Meaning |
| --- | --- |
| `ALLOWED` | Resources inherited from the launcher or selected by PaRSEC when inherited bindings are ignored. |
| `USED` | Resources selected for PaRSEC worker threads. |
| `FREE` | Allowed resources not used by worker threads; the communication thread prefers this mask when no explicit `bind_comm` is set. |

The VP map display reports logical candidate masks and their physical
translation. When ranks inherit different allowed masks from the launcher,
two ranks may both report logical resource `0`, while the physical
translation is different for each rank.

## Scheduler and Launcher Integration

When HWLOC support is available, PaRSEC initializes binding from the
current process binding:

1. The launcher or batch scheduler starts each process with a CPU binding.
2. PaRSEC records that binding as the allowed cpuset.
3. Non-negative PaRSEC resource indexes are translated through the
   allowed cpuset.
4. Worker-thread binding, VP maps, and communication-thread fallback
   binding are restricted to the allowed cpuset.

This means PaRSEC should naturally follow common launch policies:

```
mpirun -np 2 --map-by socket --bind-to socket ./app -- \
  --mca runtime_report_bindings 1
```

In this example, each rank sees only the resources allowed by its socket
binding. A user binding such as `bind_map 0,1,2,3` binds to the first
four resources in each rank's allowed cpuset. The physical cores can
differ by rank.

Some resource managers use cgroups or restricted topology views. In that
case, "all available resources" means all resources visible through the
restricted process view.

To ignore the inherited process binding and use all resources visible on
the node, set:

```
--mca runtime_ignore_bindings 1
```

Use this carefully. In distributed runs it can cause multiple ranks on
the same node to use the same cores, even if the launcher originally
gave them disjoint bindings.

## Global Binding Controls

| MCA parameter | Default behavior |
| --- | --- |
| `runtime_num_cores` | Maximum number of processing resources PaRSEC may use. `-1` means all available resources. Applications often expose this through a `-c` or similar command-line option passed to `parsec_init()`. |
| `runtime_allow_pu` | Use hardware PUs instead of physical cores as binding resources. |
| `runtime_ignore_bindings` | Ignore inherited process bindings and use all resources visible on the node. |
| `runtime_singlify_bindings` | Restrict each thread binding mask to one resource. Negative values singlify before building the VP map, `0` disables singlification, and positive values singlify after parsing the VP map. |
| `runtime_bind_main_thread` | Bind the thread that called `parsec_init()` when thread binding is enabled. |
| `bind_threads` | Enable or disable binding of PaRSEC main and worker threads. |

## Worker Thread Binding

Worker-thread binding is enabled by `bind_threads`. With no explicit
`bind_map`, PaRSEC uses `runtime_vpmap` to build candidate resources for
each VP thread, then picks one allowed, preferably unused, resource from
each candidate mask.

The supported VP map choices are:

| `runtime_vpmap` value | Meaning |
| --- | --- |
| `flat` | Put all selected resources under one virtual process. This is the default. |
| `hwloc` | Build virtual processes from hardware locality, grouping threads under the same socket or NUMA ancestor. |
| `file:<filename>` | Load a VP map from a rank file. |
| `rr:n:p:c` | Historical round-robin VP-map form. The parser recognizes this syntax, but the implementation is not complete in this branch; prefer `flat`, `hwloc`, or `file:<filename>`. |
| `display:<map>` | Build `<map>` and print the resulting VP map. For example, `display:flat` or `display:hwloc`. |

The rank-file form uses one VP description per line:

```
[mpi_rank]:nb_threads:binding
```

The `mpi_rank` field may be empty when the description applies to all
ranks. The `binding` field accepts:

| Binding form | Example |
| --- | --- |
| Core list | `1,3,5-6` |
| Hexadecimal mask | `0xff012` |
| Range expression | `start;end;step` |

The range expression in `runtime_vpmap file:` uses semicolons.

## Explicit Worker Binding with `bind_map`

`bind_map` overrides the default VP-map-derived worker placement. It
defines the concrete worker-thread binding order for the process.

Accepted forms are:

| Binding form | Example |
| --- | --- |
| Comma-separated resource list | `0,1,2,3` |
| Hexadecimal mask | `0xff` |
| Range expression | `0:7:1` |
| Per-rank file | `file:bindings.txt` |

Unlike `runtime_vpmap file:`, the `bind_map` range expression uses
colons. Non-negative indexes in `bind_map` are relative to the allowed
cpuset. If the list is shorter than the number of worker threads,
remaining worker threads are left unbound.

When `bind_map` starts with `+`, and `bind_comm` has not been set, the
communication thread is included in the binding policy rather than being
treated as a separately requested core.

Examples:

```
./app -- --mca bind_map 0,1,2,3
./app -- --mca bind_map 0:7:1
./app -- --mca bind_map file:bindings.txt
```

## Communication Thread Binding

The communication thread has its own binding policy.

If `bind_comm` is not set, the communication thread first tries to bind
to a resource in the `FREE` mask. If all allowed resources are already
used by worker threads, it falls back to the full `ALLOWED` mask and may
share resources with workers.

If `bind_comm` is set to a non-negative value, the value is interpreted
relative to the allowed cpuset:

```
./app -- --mca bind_comm 0
```

If `bind_comm` is set to a negative value, PaRSEC bypasses the allowed
cpuset translation and selects an absolute physical core index:

```
./app -- --mca bind_comm -4
```

Use absolute communication-thread binding carefully in managed
environments, because it can defeat scheduler-provided process binding.

## Practical Recipes

Display what the launcher gave PaRSEC:

```
mpirun -np 2 --map-by socket --bind-to socket ./app -- \
  --mca runtime_report_bindings 1
```

Display both process binding masks and the default flat VP map:

```
./app -- \
  --mca runtime_report_bindings 1 \
  --mca runtime_vpmap display:flat
```

Use hardware locality to create VPs and display the result:

```
./app -- \
  --mca runtime_vpmap display:hwloc \
  --mca runtime_report_bindings 1
```

Disable PaRSEC worker-thread binding:

```
./app -- --mca bind_threads 0
```

Keep inherited scheduler bindings but choose a specific relative worker
layout:

```
./app -- --mca bind_map 0,1,2,3
```

Ignore inherited launcher bindings and use all resources visible on the
node:

```
./app -- --mca runtime_ignore_bindings 1
```

## Reading the Output

When `runtime_report_bindings` is enabled, always interpret `USED` and
`FREE` relative to `ALLOWED`. In multi-rank jobs, comparing only logical
indexes can be misleading. Use the physical translation printed by
`runtime_vpmap display:<map>` or external tools such as `hwloc-ls
--restrict binding` to confirm how each rank maps its relative resources
to node resources.

If PaRSEC reports that multiple processes may share physical cores, the
job may be oversubscribed, or the resource manager may be hiding the real
binding through cgroups or another topology restriction. The check can be
silenced with:

```
--mca runtime_warn_slow_binding 0
```

Prefer fixing the launch binding first, and silence the warning only
when the external binding has been verified.
