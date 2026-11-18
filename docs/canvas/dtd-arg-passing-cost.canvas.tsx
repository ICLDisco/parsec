import {
  Callout,
  Divider,
  H1,
  H2,
  LineChart,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const ARITIES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];

/* 200,000 tasks per measurement, 30 reps, ns per task. */
type Series = { im: number[]; isd: number[]; em: number[]; esd: number[] };

const DIRECT: Series = {
  im:  [153.61, 148.41, 179.27, 178.00, 194.03, 224.86, 222.14, 266.11, 272.70, 286.10],
  isd: [  4.73,   6.01,   4.27,  10.34,   5.34,   6.98,   7.64,   6.04,   6.92,   7.88],
  em:  [ 88.17,  99.07, 110.89, 118.34, 130.77, 131.81, 135.46, 157.26, 159.75, 169.78],
  esd: [  2.26,   4.49,   3.24,  15.48,  26.02,   4.60,   3.37,   3.28,   5.84,   4.70],
};
const TASKCLASS: Series = {
  im:  [150.65, 144.76, 167.88, 168.24, 182.20, 217.44, 212.58, 249.91, 261.76, 274.14],
  isd: [  4.81,   5.49,   4.90,   3.48,   7.52,  12.49,   7.69,   9.32,   6.24,   8.96],
  em:  [ 89.48,  99.53, 107.02, 113.27, 121.88, 128.49, 134.34, 152.85, 156.07, 165.32],
  esd: [  2.63,   3.70,   2.82,   3.40,   4.57,   2.46,   3.19,   5.05,   3.71,   4.18],
};
const GENERATED: Series = {
  im:  [150.86, 143.68, 171.22, 194.24, 220.27, 214.64, 243.63, 267.06, 286.65, 300.39],
  isd: [  5.03,   4.63,   4.26,   3.47,   6.64,  22.74,   8.27,   9.57,   6.40,   7.86],
  em:  [ 85.46,  87.92,  96.54,  95.18, 100.93, 102.39, 111.34, 108.35, 110.77, 110.49],
  esd: [  2.22,   2.53,   1.70,   1.47,   2.83,   2.49,  17.15,   2.63,   1.98,   2.37],
};
const INPLACE: Series = {
  im:  [151.19, 146.25, 166.23, 191.13, 213.01, 202.80, 231.45, 255.23, 282.18, 289.14],
  isd: [  5.55,   9.61,   6.89,  10.39,   6.45,   7.22,   8.38,   8.65,  22.19,  15.11],
  em:  [ 85.76,  88.27,  98.40,  95.37,  98.08,  99.10, 104.62, 105.22, 110.64, 106.99],
  esd: [  2.83,   2.92,  21.64,  12.26,   2.56,   2.78,   2.63,   3.34,  12.77,  12.64],
};

const pm = (mean: number[], sd: number[], i: number) =>
  `${mean[i].toFixed(0)} ± ${sd[i].toFixed(0)}`;

export default function DTDArgPassingCost() {
  const theme = useHostTheme();

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 1080 }}>
      <Stack gap={6}>
        <H1>DTD argument passing: what the copy out of the task costs</H1>
        <Text tone="secondary">
          Empty tasks carrying 1–10 `PARSEC_VALUE` int arguments and no data flows, inserted and
          executed on one core, no MPI, no CUDA. 200,000 tasks per measurement, 30 reps, reported as
          mean ± sample standard deviation.
        </Text>
      </Stack>

      <Row gap={32} align="start">
        <Stat value="8.6 → 2.7 ns" label="unpack cost per argument, old vs generated" />
        <Stat value="−35%" label="execution time at 10 arguments" />
        <Stat value="0 ns" label="cost of offering both reader styles" />
      </Row>

      <Text>
        Four variants. `direct` and `taskclass` are the two existing ways of building the task and
        both unpack with `parsec_dtd_unpack_args`. `generated` and `inplace` share the same
        generated task class and the same insertion code, and differ only in the body: `generated`
        copies every argument into a named local, `inplace` reads them where the runtime already
        put them. Insertion costs the same in all four. Execution does not.
      </Text>

      <Divider />

      <Stack gap={4}>
        <H2>Execution time per task</H2>
        <Text tone="secondary" size="small">
          y: mean ns per task, insertion excluded · x: number of `PARSEC_VALUE` int arguments ·
          30 reps × 200,000 tasks, one core, dgx-gaia compute node
        </Text>
      </Stack>

      <LineChart
        categories={ARITIES}
        series={[
          { name: "direct — unpack_args", data: DIRECT.em },
          { name: "taskclass — unpack_args", data: TASKCLASS.em },
          { name: "generated — copy to locals", data: GENERATED.em },
          { name: "inplace — read in place", data: INPLACE.em },
        ]}
        valueSuffix=" ns"
        beginAtZero={false}
        height={300}
      />
      <Text size="small" tone="tertiary">
        The two existing variants climb at 8.4–8.8 ns per argument. The two generated ones climb at
        2.5–3.0 ns and are indistinguishable from each other, which is the useful part: once the
        compiler knows the layout as a struct type, copying the values into locals is free, so a
        body can be written in whichever style reads better without paying for the choice.
      </Text>

      <Divider />

      <Stack gap={4}>
        <H2>Insertion time per task</H2>
        <Text tone="secondary" size="small">
          y: mean ns per task, execution excluded · x: number of arguments
        </Text>
      </Stack>

      <LineChart
        categories={ARITIES}
        series={[
          { name: "direct", data: DIRECT.im },
          { name: "taskclass", data: TASKCLASS.im },
          { name: "generated", data: GENERATED.im },
          { name: "inplace", data: INPLACE.im },
        ]}
        valueSuffix=" ns"
        beginAtZero={false}
        height={260}
      />
      <Text size="small" tone="tertiary">
        `generated` and `inplace` run byte-identical insertion code, so the gap between those two
        lines is pure measurement noise and sets the resolution of this chart: about 1.3 ns per
        argument on the fitted slope. Every difference between the four variants is inside it.
      </Text>

      <Divider />

      <H2>Marginal cost of one argument</H2>
      <Text tone="secondary" size="small">
        Least-squares slope over arities 1–10, ± standard error of the fit. The intercept is the
        fixed per-task cost that argument handling does not explain: task allocation, scheduling and
        completion.
      </Text>
      <Table
        headers={["Variant", "Insert", "Execute", "Total per argument", "Fixed per task"]}
        columnAlign={["left", "right", "right", "right", "right"]}
        rows={[
          ["direct — parsec_dtd_insert_task", "16.1 ± 1.2 ns", "8.75 ± 0.46 ns", "24.9 ± 1.4 ns", "206 ns"],
          ["taskclass — hand-written", "15.2 ± 1.2 ns", "8.35 ± 0.31 ns", "23.6 ± 1.4 ns", "200 ns"],
          ["generated — copy to locals", "18.0 ± 1.0 ns", "3.00 ± 0.34 ns", "21.0 ± 1.0 ns", "205 ns"],
          ["inplace — read in place", "16.7 ± 1.1 ns", "2.49 ± 0.31 ns", "19.2 ± 1.2 ns", "207 ns"],
        ]}
      />
      <Text size="small" tone="tertiary">
        The execution column is the real result: 8.4–8.8 ns falls to 2.5–3.0 ns, a difference of
        roughly 12 standard errors. The insert column shows no significant difference, and the
        generated-versus-inplace pair confirms that by disagreeing with itself by 1.3 ns.
      </Text>

      <Divider />

      <H2>Measured cost per task, mean ± sd</H2>
      <Table
        headers={[
          "args",
          "ins: direct",
          "ins: taskclass",
          "ins: generated",
          "ins: inplace",
          "exec: direct",
          "exec: taskclass",
          "exec: generated",
          "exec: inplace",
        ]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right"]}
        striped
        rows={ARITIES.map((n, i) => [
          n,
          pm(DIRECT.im, DIRECT.isd, i),
          pm(TASKCLASS.im, TASKCLASS.isd, i),
          pm(GENERATED.im, GENERATED.isd, i),
          pm(INPLACE.im, INPLACE.isd, i),
          pm(DIRECT.em, DIRECT.esd, i),
          pm(TASKCLASS.em, TASKCLASS.esd, i),
          pm(GENERATED.em, GENERATED.esd, i),
          pm(INPLACE.em, INPLACE.esd, i),
        ])}
      />
      <Text size="small" tone="tertiary">
        All values ns per task. The curves zig-zag with arity because task size grows with the
        argument count and a given process gets better or worse allocation luck at particular sizes.
        All four variants are measured inside the same process, so that luck is common to them and
        cancels in the comparison, but it does move the absolute numbers between processes. Read the
        slope, not the individual points.
      </Text>

      <Callout tone="success" title="Where the saving comes from">
        `parsec_dtd_unpack_args` walks a per-task descriptor array, pulls one `void *` per parameter
        off a `va_list`, branches on the parameter's kind and calls `memcpy` with a runtime length.
        None of that depends on the task, only on its class. Declaring the parameter list once lets
        the value block be described as a plain struct, so the body either reads through a typed
        pointer or assigns from it, and the whole loop disappears.
      </Callout>

      <Divider />

      <Stack gap={6}>
        <Text weight="semibold">Method</Text>
        <Text size="small" tone="secondary">
          All four variants' argument lists are emitted from one X-macro per arity, so the tokens
          the compiler sees are what a person would have typed. Every body sums the arguments it was
          handed and the total is checked against the value the run must produce, so a variant that
          read from the wrong offset would fail rather than merely time differently. The DTD window
          is raised past the task count and the context is not started until the insertion loop
          finishes, which separates insert from execute. Each measurement is preceded by its own
          discarded run, each sweep by a discarded warm-up sweep, and the order of the variants
          rotates from rep to rep — the first variant measured at a given arity runs slightly slow
          whatever it is, so without rotating, that penalty reads as a property of whichever
          interface went first. Source: `parsec/tests/dsl/dtd/dtd_bench_arg_passing.c`.
        </Text>
        <Text size="small" style={{ color: theme.text.quaternary }}>
          Compiled -O3 -std=gnu11, GCC 11, x86_64; single process, one core, CUDA and MPI unused.
        </Text>
      </Stack>
    </Stack>
  );
}
