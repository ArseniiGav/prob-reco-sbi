# Generating data on a cluster

Large data sets are generated as **shards**: one job per shard, each with its own seed and its own range of event ids,
combined at the end with `flashsim merge`. The shards run in parallel, each is reproducible from its seed, and a failed
job is simply run again.

| file                                 | content                                                                     |
|--------------------------------------|-----------------------------------------------------------------------------|
| [`run_shard.sh`](run_shard.sh)       | generates shard `i`: seed `BASE_SEED + i`, event ids from `i * EVENTS`; holds all settings |
| [`slurm_array.sh`](slurm_array.sh)   | SLURM array job that runs `run_shard.sh` once per array index               |
| [`htcondor.sub`](htcondor.sub)       | HTCondor submit file that runs `run_shard.sh` once per job                  |

These files are **examples to adapt to your cluster**: partitions, accounts, modules, file systems and limits differ
from one cluster to the next. Ask your cluster's documentation or its administrators when in doubt.

## Before the first job

1. Install the package in a virtual environment on the cluster, as in the [README](../../README.md#installation) and [docs/install.md](../../docs/install.md): the
   CPU build of PyTorch for CPU nodes, a CUDA build for GPU nodes. The jobs must be able to read the environment.
2. Choose where the data go (`OUT_DIR`, default `data/shards`): a work or scratch file system with enough space, not
   your home directory, which usually has a small quota.
3. Try one small shard in an interactive session (not on the login node), from the repository root:

   ```bash
   EVENTS=1000 bash examples/cluster/run_shard.sh 0
   flashsim info data/shards/full_shard0000.h5
   rm data/shards/full_shard0000.h5   # it would be skipped otherwise: run_shard.sh never replaces a shard
   ```

4. Create the folder for the job logs: `mkdir -p logs`.

## Submitting

Set the number of shards (`--array=0-9` or `queue 10`) and the events per shard (`EVENTS`), then submit from the
repository root:

```bash
sbatch examples/cluster/slurm_array.sh          # SLURM
condor_submit examples/cluster/htcondor.sub     # HTCondor
```

Another preset is chosen with `PRESET` (e.g. `PRESET=absorb`). Further options of `flashsim generate`, such as
`--save-hits` or `--dark-rate 1e6`, are passed after the shard index (`run_shard.sh 3 --save-hits`) and appended to
the command. A shard whose file exists already is skipped, so after a failure you can submit the same array again and
only the missing shards are generated.

## Combining the shards

```bash
ls data/shards/full_shard*.h5 | wc -l                       # all shards there?
flashsim merge -o data/full_n1M.h5 data/shards/full_shard*.h5
flashsim info data/full_n1M.h5
```

`flashsim merge` refuses shards with different configurations or options, shards with the same seed (their events
would be identical) and shards whose event ids overlap. It does not know how many shards you meant to make, so check
their number first. The merged file keeps the provenance of every shard.

## Resources

Measured for the default model (`full`) on a shared server CPU in October 2026; your numbers will differ, and T0.2 of
[TASKS.md](../../TASKS.md) asks you to measure them on your cluster:

| job                                      | simulation time | peak memory | file size |
|------------------------------------------|-----------------|-------------|-----------|
| 100 000 events, 4 threads                | 34 s            | 0.76 GB     | 26 MB     |
| 10 000 events with `--save-hits`, 8 threads | 2.6 s        | 1.3 GB      | 70 MB     |
| 30 000 events with `--save-hits`, 8 threads | (not timed)  | 3.3 GB      | 209 MB    |

- **Cores:** the speed grows slowly beyond a few threads (about 1100, 2900 and 4000 events per second with 1, 4 and
  8 threads), so 4 cores per job are a good choice. Pass the number of cores with `THREADS` (`--threads`).
- **Memory:** 2 GB per job are enough without hits. The hit list is kept in memory until the file is written, so with
  `--save-hits` keep the shards small (10 000 events) and request 3 GB.
- **GPU nodes:** set `DEVICE=cuda` and request a GPU (the commented lines in the scripts). Whether a GPU job is worth it
  for your shard size is what T0.2 finds out.
- **Time limit:** shards of the size above finish within minutes; ask for a generous limit the first time and adjust it
  from the logs.

## Good practice

- Do not run simulations on the login node; use jobs or an interactive session.
- Limit the number of jobs that run at the same time if your cluster asks for it (`--array=0-99%10` for SLURM,
  `materialize_max_idle` for HTCondor).
- Use separate seed ranges for data sets that must be independent, e.g. `BASE_SEED=1000` for a training set and
  `BASE_SEED=2000` for a test set (with fewer than 1000 shards each). The same seed gives the same events.
- Keep the logs (`logs/` is ignored by git) until the data set is checked.
