import json

import h5py
import numpy as np
import pytest

from helpers import check_conservation
from prob_reco_sbi.flash import POINT_TYPES, describe, generate, load, merge, preset, save


def assert_same(a, b, groups=("truth", "response", "photons", "hits", "tracks")):
    for group in groups:
        ca, cb = getattr(a, group), getattr(b, group)
        assert (ca is None) == (cb is None), group
        if ca is None:
            continue
        for name, value in ca.arrays().items():
            other = getattr(cb, name)
            assert other.dtype == value.dtype, f"{group}/{name}"
            np.testing.assert_array_equal(other, value, err_msg=f"{group}/{name}")


def test_round_trip_with_hits_and_tracks(full_small, tmp_path):
    path = save(full_small, tmp_path / "sub" / "run.h5")
    assert path.exists()
    assert not (tmp_path / "sub" / "run.h5.part").exists()
    back = load(path)
    assert_same(full_small, back)
    assert back.config == full_small.config
    assert back.provenance == full_small.provenance
    light = load(path, hits=False, tracks=False)
    assert light.hits is None
    assert light.tracks is None
    with pytest.raises(FileExistsError):
        save(full_small, path, overwrite=False)


def test_file_layout(full_small, tmp_path):
    path = save(full_small, tmp_path / "run.h5")
    with h5py.File(path, "r") as f:
        assert f.attrs["format_version"] == 1
        assert f.attrs["n_events"] == 200
        assert json.loads(f.attrs["config"]) == full_small.config.to_dict()
        assert f["response/charge"].shape == (200, 32)
        assert f["response/charge"].dtype == np.int32
        assert f["response/time_mean"].dtype == np.float32
        assert f["truth/event_id"].dtype == np.int64
        assert f["truth/x"].attrs["units"] == "m"
        assert f["hits/time"].attrs["units"] == "ns"
        assert json.loads(f["tracks"].attrs["point_types"]) == POINT_TYPES
        assert f["detector/sensor_x"].shape == (32,)
        assert f["detector"].attrs["photocoverage"] == pytest.approx(0.5093, abs=1e-4)
        for name in ("n_emitted", "n_detected", "n_absorbed_bulk", "n_absorbed_wall", "n_absorbed_sensor"):
            assert f[f"photons/{name}"].dtype == np.int32


def test_merge_shards(tmp_path):
    kwargs = {"config": preset("absorb-scatter"), "device": "cpu", "save_hits": True, "save_tracks": 2}
    a = generate(30, seed=1, first_event_id=0, **kwargs)
    b = generate(20, seed=2, first_event_id=30, **kwargs)
    paths = [save(a, tmp_path / "a.h5"), save(b, tmp_path / "b.h5")]
    merged = load(merge(paths, tmp_path / "ab.h5"))
    assert len(merged) == 50
    np.testing.assert_array_equal(merged.truth.event_id, np.arange(50))
    np.testing.assert_array_equal(merged.response.charge, np.concatenate([a.response.charge, b.response.charge]))
    np.testing.assert_array_equal(merged.hits.time, np.concatenate([a.hits.time, b.hits.time]))
    check_conservation(merged)
    assert len(merged.tracks) == len(a.tracks) + len(b.tracks)
    x, _, kind = merged.tracks.polyline(len(a.tracks) + 3)
    xb, _, kb = b.tracks.polyline(3)
    np.testing.assert_array_equal(x, xb)
    np.testing.assert_array_equal(kind, kb)
    sources = merged.provenance["merged_from"]
    assert [s["seed"] for s in sources] == [1, 2]
    assert [s["file"] for s in sources] == ["a.h5", "b.h5"]
    text = describe(tmp_path / "ab.h5")
    assert "merged from 2 files" in text
    assert "absorb-scatter" in text


def test_merge_refuses_inconsistent_inputs(tmp_path):
    def shard(name, seed, first, **overrides):
        data = generate(5, config=preset("full", **overrides), seed=seed, device="cpu", first_event_id=first)
        return save(data, tmp_path / name)

    a = shard("a.h5", 1, 0)
    with pytest.raises(ValueError, match="different configuration"):
        merge([a, shard("b.h5", 2, 5, reflectivity=0.9)], tmp_path / "out.h5")
    with pytest.raises(ValueError, match="identical copies"):
        merge([a, shard("c.h5", 1, 5)], tmp_path / "out.h5")
    with pytest.raises(ValueError, match="event ids"):
        merge([a, shard("d.h5", 3, 0)], tmp_path / "out.h5")
    hits = save(generate(5, seed=4, device="cpu", first_event_id=5, save_hits=True), tmp_path / "e.h5")
    with pytest.raises(ValueError, match="groups"):
        merge([a, hits], tmp_path / "out.h5")
    with pytest.raises(ValueError, match="also an input"):
        merge([a, hits], a)
    assert not (tmp_path / "out.h5").exists()


def test_load_rejects_foreign_files(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.arange(3)
    with pytest.raises(ValueError, match="not a flash data file"):
        load(path)


def test_describe(full_small, tmp_path):
    text = describe(save(full_small, tmp_path / "run.h5"))
    for expected in ("200 events", "preset: full", "/response/charge", "seed: 11", "detected", "lost"):
        assert expected in text
