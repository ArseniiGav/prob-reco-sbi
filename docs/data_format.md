# Data format

Every run of `flashsim generate` (or `save()` in Python) writes one HDF5 file. This page describes format version 1,
written by `prob_reco_sbi` 0.1.0. Below, `N` is the number of events and `M` the number of sensors (32 by default).
The physics behind the stored quantities is described in [physics.md](physics.md).

```text
file.h5
├── attrs: format_version, n_events, config (JSON), provenance (JSON)
├── truth/       true event parameters                    [N]
├── response/    charge and times per sensor              [N, M]
├── photons/     photon bookkeeping per event             [N]
├── hits/        every hit (optional: --save-hits)        [n_hits]
├── tracks/      photon paths (optional: --save-tracks)   [n_tracks], [n_points]
└── detector/    sensor layout                            [M]
```

## Reading the data

The examples use the file written by `flashsim generate -n 10000 --seed 1`. In Python, `load` returns the same
`SimResult` object as `generate`, with NumPy arrays:

```python
from prob_reco_sbi.flash import load

data = load("data/flash_full_n10000_seed1.h5")
data.truth.energy  # [N] true energies
data.response.charge  # [N, M] charges
data.config  # the SimConfig used, e.g. data.config.optics.absorption_length
data.provenance["seed"]  # seed, versions, creation time, ...

light = load("data/flash_full_n10000_seed1.h5", hits=False, tracks=False)  # skip the large optional groups
```

`flashsim info FILE` prints the datasets, the configuration, the provenance and a short photon bookkeeping. The files
are plain HDF5, so any HDF5 reader works, e.g. `h5py`:

```python
import h5py

with h5py.File("data/flash_full_n10000_seed1.h5", "r") as f:
    charge = f["response/charge"][()]  # numpy array [N, M]
    print(f["truth/energy"].attrs["units"])  # MeV
```

A table of the per-event quantities is one line with pandas:

```python
import pandas as pd

table = pd.DataFrame(data.truth.arrays() | data.photons.arrays())
```

## Root attributes

| attribute        | type   | content                                                                         |
|------------------|--------|---------------------------------------------------------------------------------|
| `format_version` | int64  | 1                                                                               |
| `n_events`       | int64  | number of events `N`                                                            |
| `config`         | string | JSON of the complete configuration (`SimConfig.to_dict()`), read back by `SimConfig.from_json` |
| `provenance`     | string | JSON: how the file was made (see [Provenance](#provenance))                     |

`config` has the sections `geometry`, `optics`, `readout` and `prior` with every parameter of
[physics.md](physics.md); `fixed_vertex` and `fixed_energy` are `null` when not used.

## `/truth`: event parameters

| dataset    | dtype   | shape | units | content                                                    |
|------------|---------|-------|-------|------------------------------------------------------------|
| `event_id` | int64   | [N]   |       | event number, unique within a file (`--first-event-id` sets the first one) |
| `x`, `y`   | float32 | [N]   | m     | vertex $(x_0, y_0)$                                        |
| `energy`   | float32 | [N]   | MeV   | energy $E$                                                 |
| `t0`       | float32 | [N]   | ns    | emission time $t_0$ (a nuisance parameter)                 |

## `/response`: what the detector records

| dataset      | dtype   | shape  | units | content                                                       |
|--------------|---------|--------|-------|---------------------------------------------------------------|
| `charge`     | int32   | [N, M] |       | number of hits $n_j$ per sensor (detected photons and noise hits) |
| `time_mean`  | float32 | [N, M] | ns    | mean hit time $\tau_j$; NaN where `charge == 0`               |
| `time_first` | float32 | [N, M] | ns    | earliest hit time; NaN where `charge == 0`                    |

Column `j` is sensor `j`, whose position is in `/detector`. The network input of the next stage is computed from
`charge` and `time_mean` by `network_input` (definition in [physics.md](physics.md#network-input)).

## `/photons`: photon bookkeeping per event

All datasets are int32 arrays of shape [N].

| dataset                | content                                                              |
|------------------------|----------------------------------------------------------------------|
| `n_emitted`            | photons emitted                                                      |
| `n_detected`           | photons detected by a sensor                                         |
| `n_absorbed_bulk`      | photons absorbed in the bulk                                         |
| `n_absorbed_wall`      | photons absorbed by the wall between the sensors                     |
| `n_absorbed_sensor`    | photons that reached a sensor but were not detected (efficiency < 1) |
| `n_lost`               | photons still in flight after `max_steps` interactions               |
| `n_detected_direct`    | detected photons that were neither scattered nor reflected           |
| `n_detected_scattered` | detected photons scattered at least once and never reflected         |
| `n_detected_reflected` | detected photons reflected at least once                             |
| `n_noise`              | dark-noise hits                                                      |

These identities hold exactly for every event:

```text
n_emitted  = n_detected + n_absorbed_bulk + n_absorbed_wall + n_absorbed_sensor + n_lost
n_detected = n_detected_direct + n_detected_scattered + n_detected_reflected
sum over sensors of charge = n_detected + n_noise
```

## `/hits`: every hit (optional)

Written with `--save-hits` (`save_hits=True`). One row per hit, shape [n_hits]:

| dataset       | dtype   | units | content                                                    |
|---------------|---------|-------|------------------------------------------------------------|
| `event_id`    | int64   |       | event of the hit                                           |
| `sensor`      | int32   |       | sensor index `j`                                           |
| `time`        | float32 | ns    | hit time (including $t_0$ and the time jitter)             |
| `n_scatter`   | int32   |       | number of scatterings of the photon (-1 for noise hits)    |
| `n_reflect`   | int32   |       | number of wall reflections of the photon (-1 for noise hits) |
| `path_length` | float32 | m     | total path length of the photon (NaN for noise hits)       |
| `is_noise`    | bool    |       | true for dark-noise hits                                   |

The rows are grouped by event: first the detected photons of an event, then its noise hits. Within an event they are
not ordered in time. In a file written with hits, the hits of one event are selected with a mask, e.g.
`data.hits.time[data.hits.event_id == data.truth.event_id[0]]`.

Hit lists are large: about 900 hits per event of the default model, so 10 000 events take 70 MB with hits and 2.6 MB
without them (measured with the default prior).

## `/tracks`: photon paths (optional)

Written with `--save-tracks K --track-photons P`: the paths of the first `P` photons (`--track-photons 0`: all of
them) of each of the first `K` events, as polylines flattened into one list of points.

| dataset       | dtype   | shape          | units | content                                                  |
|---------------|---------|----------------|-------|----------------------------------------------------------|
| `event_id`    | int64   | [n_tracks]     |       | event of each track                                      |
| `photon`      | int32   | [n_tracks]     |       | number of the photon within its event                    |
| `offsets`     | int64   | [n_tracks + 1] |       | track `k` has the points `offsets[k]:offsets[k + 1]`     |
| `x`, `y`      | float32 | [n_points]     | m     | point coordinates                                        |
| `path_length` | float32 | [n_points]     | m     | path length from the vertex to the point                 |
| `kind`        | int8    | [n_points]     |       | type of the point, see the table below                   |

The group attribute `point_types` holds the codes as JSON:

| code | name            | meaning                                                  |
|------|-----------------|----------------------------------------------------------|
| 0    | `EMIT`          | emission at the vertex (the first point of every track)  |
| 1    | `SCATTER`       | Rayleigh scattering in the bulk                          |
| 2    | `REFLECT`       | reflection on the wall between the sensors               |
| 3    | `DETECT`        | detection by a sensor                                    |
| 4    | `ABSORB_BULK`   | absorption in the bulk                                   |
| 5    | `ABSORB_WALL`   | absorption by the wall between the sensors               |
| 6    | `ABSORB_SENSOR` | reached a sensor but not detected                        |
| 7    | `LOST`          | still in flight after `max_steps` interactions           |

The last point of a track is its fate (codes 3 to 7). In Python, `Tracks` has helpers (the file below was made with
`flashsim generate -n 100 --seed 2 --save-tracks 5 --track-photons 200 -o data/flash_full_tracks.h5`):

```python
from prob_reco_sbi.flash import Interaction, load

tracks = load("data/flash_full_tracks.h5").tracks
for k in tracks.of_event(tracks.event_id[0]):  # the tracks of the first stored event
    x, y, kind = tracks.polyline(k)  # points of track k and their types
fates = tracks.fate  # last point type of every track
n_detected = (fates == Interaction.DETECT).sum()
```

The path length at every point gives the time at which the photon passes it, $t_0 + s / c_n$, which is what an
animation needs.

## `/detector`: sensor layout

| dataset      | dtype   | shape | units | content                    |
|--------------|---------|-------|-------|----------------------------|
| `sensor_x`   | float64 | [M]   | m     | x of the sensor centres    |
| `sensor_y`   | float64 | [M]   | m     | y of the sensor centres    |
| `sensor_phi` | float64 | [M]   | rad   | angle of the sensor centres, $2\pi j / M$ |

Attributes: `radius` [m], `n_sensors`, `sensor_width` [m] (arc length of a sensor) and `photocoverage`.

## Provenance

The `provenance` attribute records how a file was made:

| key                                     | content                                                     |
|-----------------------------------------|-------------------------------------------------------------|
| `seed`                                  | random seed                                                 |
| `first_event_id`                        | event id of the first event                                 |
| `photons_per_chunk`, `n_chunks`         | chunking of the photon transport (the output depends on it) |
| `save_hits`, `save_tracks`, `track_photons` | the optional output requested                           |
| `device`, `device_name`, `torch_threads`| where the simulation ran                                    |
| `package_version`, `git_commit`         | version of this package; the commit (with `-dirty` for local changes) when run from a git checkout, otherwise `null` |
| `python_version`, `numpy_version`, `torch_version`, `h5py_version`, `platform` | software environment |
| `created`, `wall_time_s`                | creation time (UTC) and simulation time in seconds          |
| `preset`, `overrides`, `command`        | only from `flashsim generate`: the preset, the changed parameters and the full command line |

A file made by `flashsim merge` has the keys `merged_from` (the list of input files, each with its file name and its own
provenance), `package_version` and `created`.

## Merging files

`flashsim merge -o OUT FILE...` (or `merge(paths, out)`) concatenates files in the given order, copying block by block
so that the inputs need not fit in memory. It refuses inputs with different configurations or different optional
groups, inputs with the same seed (their events would be identical copies) and inputs whose event ids overlap: give
every shard its own seed and its own `--first-event-id` range. [examples/cluster](../examples/cluster/README.md) shows
the complete workflow.

## Storage details

- Every dataset has a `units` attribute (an empty string for counts and codes).
- Datasets with more than 1024 elements are compressed with gzip (level 4) and the shuffle filter.
- A file is written as `<name>.part` and renamed when it is complete, so an interrupted job never leaves a truncated
  file under the final name.
- `load` rejects files without `format_version` and files with a newer format version than the installed package
  reads.
