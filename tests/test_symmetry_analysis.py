import math

import numpy as np

from script.symmetry_analysis import (
    classify_measurement_profile,
    pattern_geometry,
    radial_profile,
    radial_profiles_batch,
)


def test_pattern_geometry_exact_and_one_pair_mismatch():
    rng = np.random.default_rng(7)
    left = rng.integers(0, 2, (25, 12), dtype=np.uint8)
    p = np.zeros((25, 25), dtype=np.uint8)
    p[:, :12] = left
    p[:, 13:] = left[:, ::-1]
    p[:, 12] = rng.integers(0, 2, 25)
    exact = pattern_geometry(p)
    assert exact["geometry_mismatch_fraction"] == 0.0
    assert exact["symmetry_class"] == "exact"

    p[3, 2] ^= 1
    changed = pattern_geometry(p)
    assert changed["geometry_mismatch_fraction"] == 1 / (25 * 12)
    assert changed["symmetry_class"] == "near"


def test_radial_profile_synthetic_mirror_and_flat_peak_centroid():
    theta = np.arange(-180, 181, 2, dtype=float)
    mirrored = -0.002 * theta ** 2
    metrics = radial_profile(theta, mirrored)
    assert metrics["mirror_power_45"] == 0.0
    assert metrics["mirror_power_90"] == 0.0
    assert abs(metrics["power_centroid_deg"]) < 1e-12

    # A flat top has an arbitrary first argmax, while the power centroid stays 0.
    flat = np.full(theta.shape, -30.0)
    flat[np.abs(theta) <= 40] = 5.0
    metrics = radial_profile(theta, flat)
    assert metrics["argmax_deg_diagnostic"] == -40.0
    assert abs(metrics["power_centroid_deg"]) < 1e-12


def test_radial_profile_missing_and_nonfinite_are_explicit():
    theta = np.arange(-180, 181, 2, dtype=float)
    missing = radial_profile(theta, np.full(theta.shape, np.nan))
    assert math.isnan(missing["mirror_power_90"])
    assert missing["mirror_pairs_90"] == 0
    assert math.isnan(missing["power_centroid_deg"])

    curve = -np.abs(theta) / 20
    curve[theta == 10] = np.inf
    curve[theta == -20] = np.nan
    metrics = radial_profile(theta, curve)
    assert metrics["mirror_pairs_45"] == 20  # 22 pairs minus the two invalid pairs
    assert np.isfinite(metrics["mirror_power_45"])


def test_batch_radial_profile_matches_single_curve_path():
    theta = np.arange(-180, 181, 2, dtype=float)
    curves = np.vstack((-np.abs(theta) / 10, 3 * np.cos(np.deg2rad(theta))))
    curves[1, 20] = np.nan
    batch = radial_profiles_batch(theta, curves)
    for i in range(2):
        single = radial_profile(theta, curves[i])
        for key, values in batch.items():
            assert np.isclose(values[i], single[key], equal_nan=True)


def test_measurement_profile_keeps_bridge_mesh_and_pixel_conditions():
    setup = {"diag_bridge_w": 0.075, "pixel_count": 50,
             "max_delta_s": 0.005, "max_passes": 20}
    manifest = {"kind": "meshconv", "port": "single"}
    stratum, digest, profile = classify_measurement_profile(setup, manifest)
    assert "meshconv" in stratum
    assert "bridge_w=0.075" in stratum
    assert "pixels=50" in stratum
    assert "solver=" in stratum
    assert len(digest) == 64
    assert profile["port"] == "single"


def test_measurement_profile_legacy_is_not_falsely_resolved():
    stratum, digest, profile = classify_measurement_profile({}, None)
    assert stratum == "standard_or_legacy_unspecified"
    assert len(digest) == 64
    assert profile["diag_bridge_w"] is None
