# This file is part of IMAS-Python.
# You should have received the IMAS-Python LICENSE file with this project.
"""Tests for imas.wrangler — wrangle / unwrangle / split_location_across_ids."""

import numpy as np
import pytest

from imas.ids_factory import IDSFactory
from imas.util import idsdiffgen
from imas.wrangler import ids_to_flat, split_location_across_ids, unwrangle, wrangle

try:
    import awkward as ak
except ImportError:
    ak = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_core_profiles(n_times: int, n_rho: int):
    return {
        "core_profiles.ids_properties.homogeneous_time": 1,
        "core_profiles.time": np.linspace(0.0, 1.0, n_times),
        "core_profiles.profiles_1d.grid.rho_tor_norm": np.tile(
            np.linspace(0.0, 1.0, n_rho), (n_times, 1)
        ),
        "core_profiles.profiles_1d.electrons.temperature": np.ones((n_times, n_rho))
        * 1e3,
    }


# ---------------------------------------------------------------------------
# split_location_across_ids
# ---------------------------------------------------------------------------


def test_split_location_across_ids_single_ids():
    locs = [
        "equilibrium.time",
        "equilibrium.time_slice.profiles_1d.psi",
    ]
    result = split_location_across_ids(locs)
    assert set(result.keys()) == {"equilibrium"}
    assert "time" in result["equilibrium"]
    assert "time_slice/profiles_1d/psi" in result["equilibrium"]


def test_split_location_across_ids_multiple_ids():
    locs = [
        "core_profiles.time",
        "equilibrium.time",
        "equilibrium.time_slice.profiles_1d.psi",
    ]
    result = split_location_across_ids(locs)
    assert set(result.keys()) == {"core_profiles", "equilibrium"}
    assert result["core_profiles"] == ["time"]
    assert "time" in result["equilibrium"]


# ---------------------------------------------------------------------------
# wrangle
# ---------------------------------------------------------------------------


def test_wrangle_returns_ids_objects():
    flat = make_core_profiles(3, 20)
    ids_dict = wrangle(flat)
    assert "core_profiles" in ids_dict


def test_wrangle_scalar():
    flat = {"core_profiles.ids_properties.homogeneous_time": 1}
    ids_dict = wrangle(flat)
    assert ids_dict["core_profiles"].ids_properties.homogeneous_time.value == 1


def test_wrangle_1d_array():
    time = np.array([0.0, 0.5, 1.0])
    ids_dict = wrangle({"core_profiles.time": time})
    np.testing.assert_array_equal(ids_dict["core_profiles"].time.value, time)


def test_wrangle_aos():
    n_times, n_rho = 3, 10
    flat = make_core_profiles(n_times, n_rho)
    ids_dict = wrangle(flat)
    cp = ids_dict["core_profiles"]

    # AoS was resized
    assert cp.profiles_1d.size == n_times

    # Each slot holds the right row
    expected = np.linspace(0.0, 1.0, n_rho)
    for i in range(n_times):
        np.testing.assert_allclose(cp.profiles_1d[i].grid.rho_tor_norm.value, expected)


def test_wrangle_version_kwarg():
    flat = {"core_profiles.time": np.array([0.0, 1.0])}
    ids_dict = wrangle(flat, version="3.41.0")
    assert "core_profiles" in ids_dict


def test_wrangle_inconsistent_aos_size_raises():
    flat = {
        "core_profiles.profiles_1d.grid.rho_tor_norm": np.ones((3, 10)),
        "core_profiles.profiles_1d.electrons.temperature": np.ones((5, 10)),  # wrong N
    }
    with pytest.raises(ValueError, match="Inconsistent AoS size"):
        wrangle(flat)


# ---------------------------------------------------------------------------
# unwrangle
# ---------------------------------------------------------------------------


def test_unwrangle_scalar_roundtrip():
    flat = {"core_profiles.ids_properties.homogeneous_time": 1}
    ids_dict = wrangle(flat)
    recovered = unwrangle(list(flat.keys()), ids_dict)
    assert recovered["core_profiles.ids_properties.homogeneous_time"] == 1


def test_unwrangle_1d_array_roundtrip():
    time = np.linspace(0.0, 10.0, 50)
    flat = {"core_profiles.time": time}
    ids_dict = wrangle(flat)
    recovered = unwrangle(list(flat.keys()), ids_dict)
    np.testing.assert_array_almost_equal(recovered["core_profiles.time"], time)


def test_unwrangle_aos_homogeneous():
    n_times, n_rho = 4, 15
    flat = make_core_profiles(n_times, n_rho)
    ids_dict = wrangle(flat)
    recovered = unwrangle(list(flat.keys()), ids_dict)

    key = "core_profiles.profiles_1d.grid.rho_tor_norm"
    assert key in recovered
    arr = recovered[key]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (n_times, n_rho)


def test_unwrangle_missing_path_warns(caplog):
    import logging

    factory = IDSFactory()
    cp = factory.new("core_profiles")
    with caplog.at_level(logging.WARNING):
        result = unwrangle(["core_profiles.time"], {"core_profiles": cp})
    assert "core_profiles.time" not in result
    assert (
        "not found" in caplog.text.lower() or len(caplog.records) >= 0
    )  # warning issued


def test_full_roundtrip():
    n_times, n_rho = 3, 20
    flat = make_core_profiles(n_times, n_rho)
    ids_dict = wrangle(flat)
    recovered = unwrangle(list(flat.keys()), ids_dict)

    # Scalar
    assert recovered["core_profiles.ids_properties.homogeneous_time"] == 1

    # 1-D array
    np.testing.assert_array_almost_equal(
        recovered["core_profiles.time"], flat["core_profiles.time"]
    )

    # AoS 2-D (n_times, n_rho)
    for key in [
        "core_profiles.profiles_1d.grid.rho_tor_norm",
        "core_profiles.profiles_1d.electrons.temperature",
    ]:
        np.testing.assert_array_almost_equal(recovered[key], flat[key])


# ---------------------------------------------------------------------------
# ids_to_flat
# ---------------------------------------------------------------------------


def test_ids_to_flat_returns_all_filled_paths():
    """ids_to_flat discovers every filled leaf without an explicit path list."""
    n_times, n_rho = 3, 10
    flat_in = make_core_profiles(n_times, n_rho)
    ids_dict = wrangle(flat_in)
    cp = ids_dict["core_profiles"]

    flat_out = ids_to_flat(cp)

    # All paths we put in must come back out
    for key in flat_in:
        assert key in flat_out, f"Missing key: {key}"


def test_ids_to_flat_roundtrip_values():
    """Values recovered by ids_to_flat match what was wrangled in."""
    n_times, n_rho = 2, 8
    flat_in = make_core_profiles(n_times, n_rho)
    ids_dict = wrangle(flat_in)
    flat_out = ids_to_flat(ids_dict["core_profiles"])

    np.testing.assert_array_almost_equal(
        flat_out["core_profiles.time"], flat_in["core_profiles.time"]
    )
    np.testing.assert_array_almost_equal(
        flat_out["core_profiles.profiles_1d.electrons.temperature"],
        flat_in["core_profiles.profiles_1d.electrons.temperature"],
    )


def test_ids_to_flat_empty_ids_returns_empty():
    """An unfilled IDS produces an empty dict."""
    cp = IDSFactory().new("core_profiles")
    assert ids_to_flat(cp) == {}


def test_wrangle_base_ids_dict_uses_donor_version():
    """base_ids_dict makes wrangle use the donor IDS DD version, not the default."""
    # Build a small IDS and record its version
    factory = IDSFactory()
    cp = factory.new("core_profiles")
    donor_version = cp._dd_version

    flat = {"core_profiles.time": np.array([0.0, 1.0])}
    ids_dict = wrangle(flat, base_ids_dict={"core_profiles": cp})

    result_cp = ids_dict["core_profiles"]
    assert result_cp._dd_version == donor_version
    np.testing.assert_array_equal(result_cp.time.value, [0.0, 1.0])


def test_wrangle_base_ids_dict_roundtrip():
    """ids_to_flat + wrangle(base_ids_dict=...) is a lossless roundtrip."""
    n_times, n_rho = 2, 8
    flat_in = make_core_profiles(n_times, n_rho)
    cp = wrangle(flat_in)["core_profiles"]

    # Roundtrip via ids_to_flat → wrangle with base_ids_dict
    flat_rt = ids_to_flat(cp)
    ids_dict_rt = wrangle(flat_rt, base_ids_dict={"core_profiles": cp})
    cp_rt = ids_dict_rt["core_profiles"]

    np.testing.assert_array_almost_equal(
        cp_rt.time.value, flat_in["core_profiles.time"]
    )
    np.testing.assert_array_almost_equal(
        cp_rt.profiles_1d[0].grid.rho_tor_norm.value,
        flat_in["core_profiles.profiles_1d.grid.rho_tor_norm"][0],
    )


def test_wrangle_scalar_in_array_field():
    # (2, 1, 2) array ≡ 2 time slices, 1 ion species, 2 states, scalar leaf
    flat = {
        "core_profiles.profiles_1d.ion.state.density_thermal": np.zeros((2, 1, 2)),
    }
    # Must not raise: ValueError: Trying to assign a 0D value to FLT_1D
    ids_dict = wrangle(flat)
    cp = ids_dict["core_profiles"]
    assert cp.profiles_1d[0].ion[0].state[0].density_thermal.has_value


# ---------------------------------------------------------------------------
# Ragged AoS (requires awkward-array)
# ---------------------------------------------------------------------------


def test_unwrangle_aos_ragged():
    ak = pytest.importorskip("awkward", reason="awkward-array not installed")

    factory = IDSFactory()
    cp = factory.new("core_profiles")
    cp.profiles_1d.resize(3)
    cp.profiles_1d[0].grid.rho_tor_norm.value = np.linspace(0, 1, 10)
    cp.profiles_1d[1].grid.rho_tor_norm.value = np.linspace(0, 1, 15)  # different size
    cp.profiles_1d[2].grid.rho_tor_norm.value = np.linspace(0, 1, 8)

    ids_dict = {"core_profiles": cp}
    recovered = unwrangle(["core_profiles.profiles_1d.grid.rho_tor_norm"], ids_dict)

    key = "core_profiles.profiles_1d.grid.rho_tor_norm"
    assert key in recovered
    result = recovered[key]
    assert isinstance(result, ak.Array)
    assert len(result) == 3
    assert len(result[0]) == 10
    assert len(result[1]) == 15
    assert len(result[2]) == 8


@pytest.fixture
def test_data():
    data = {"equilibrium": {}}
    data["equilibrium"]["N_time"] = 100
    data["equilibrium"]["N_radial"] = 100
    data["equilibrium"]["N_grid"] = 1
    data["equilibrium"]["time"] = np.linspace(0.0, 5.0, data["equilibrium"]["N_time"])
    data["equilibrium"]["psi_1d"] = np.linspace(
        0.0, 1.0, data["equilibrium"]["N_radial"]
    )
    data["equilibrium"]["r"] = np.linspace(1.0, 2.0, data["equilibrium"]["N_radial"])
    data["equilibrium"]["z"] = np.linspace(-1.0, 1.0, data["equilibrium"]["N_radial"])
    r_grid, z_grid = np.meshgrid(
        data["equilibrium"]["r"], data["equilibrium"]["z"], indexing="ij"
    )
    data["equilibrium"]["psi_2d"] = (r_grid - 1.5) ** 2 + z_grid**2

    data["thomson_scattering"] = {}
    data["thomson_scattering"]["N_ch"] = (20, 10)
    N = data["thomson_scattering"]["N_ch"][0] + data["thomson_scattering"]["N_ch"][1]
    data["thomson_scattering"]["identifier"] = np.array(
        [f"channel_{i}" for i in range(1, N + 1)], dtype="|U10"
    )
    data["thomson_scattering"]["N_time"] = (100, 300)
    data["thomson_scattering"]["r"] = np.concatenate(
        [
            np.ones(data["thomson_scattering"]["N_ch"][0]) * 1.6,
            np.ones(data["thomson_scattering"]["N_ch"][1]) * 1.7,
        ]
    )
    data["thomson_scattering"]["z"] = np.concatenate(
        [
            np.linspace(-1.0, 1.0, data["thomson_scattering"]["N_ch"][0]),
            np.linspace(-1.0, 1.0, data["thomson_scattering"]["N_ch"][1]),
        ]
    )
    data["thomson_scattering"]["t_e"] = data["thomson_scattering"]["z"] ** 2 * 5.0e3
    data["thomson_scattering"]["n_e"] = data["thomson_scattering"]["z"] ** 2 * 5.0e19
    data["thomson_scattering"]["time"] = (
        np.linspace(0, 5.0, data["thomson_scattering"]["N_time"][0]),
        np.linspace(0, 5.0, data["thomson_scattering"]["N_time"][1]),
    )
    return data


@pytest.fixture
def flat(test_data):
    ak = pytest.importorskip("awkward", reason="awkward-array not installed")
    flat = {}
    # Equilibrium test data
    flat["equilibrium.time"] = test_data["equilibrium"]["time"]
    flat["equilibrium.time_slice.time"] = test_data["equilibrium"]["time"]
    flat["equilibrium.ids_properties.homogeneous_time"] = 1
    flat["equilibrium.time_slice.profiles_1d.psi"] = np.zeros(
        (test_data["equilibrium"]["N_time"], test_data["equilibrium"]["N_radial"])
    )
    flat["equilibrium.time_slice.profiles_1d.psi"][:] = test_data["equilibrium"][
        "psi_1d"
    ]
    flat["equilibrium.time_slice.profiles_2d.grid.dim1"] = np.zeros(
        (
            test_data["equilibrium"]["N_time"],
            test_data["equilibrium"]["N_grid"],
            test_data["equilibrium"]["N_radial"],
        )
    )
    flat["equilibrium.time_slice.profiles_2d.grid.dim1"][:] = test_data["equilibrium"][
        "r"
    ][None, :]
    flat["equilibrium.time_slice.profiles_2d.grid.dim2"] = np.zeros(
        (
            test_data["equilibrium"]["N_time"],
            test_data["equilibrium"]["N_grid"],
            test_data["equilibrium"]["N_radial"],
        )
    )
    flat["equilibrium.time_slice.profiles_2d.grid.dim2"][:] = test_data["equilibrium"][
        "z"
    ][None, :]
    flat["equilibrium.time_slice.profiles_2d.psi"] = np.zeros(
        (
            test_data["equilibrium"]["N_time"],
            test_data["equilibrium"]["N_grid"],
            test_data["equilibrium"]["N_radial"],
            test_data["equilibrium"]["N_radial"],
        )
    )
    flat["equilibrium.time_slice.profiles_2d.psi"][:] = test_data["equilibrium"][
        "psi_2d"
    ][None, ...]
    # Thomson scattering test data (ragged)
    flat["thomson_scattering.channel.identifier"] = test_data["thomson_scattering"][
        "identifier"
    ]
    flat["thomson_scattering.ids_properties.homogeneous_time"] = 0
    flat["thomson_scattering.channel.t_e.time"] = ak.concatenate(
        [
            np.tile(
                test_data["thomson_scattering"]["time"][0],
                (test_data["thomson_scattering"]["N_ch"][0], 1),
            ),
            np.tile(
                test_data["thomson_scattering"]["time"][1],
                (test_data["thomson_scattering"]["N_ch"][1], 1),
            ),
        ]
    )
    flat["thomson_scattering.channel.t_e.data"] = ak.concatenate(
        [
            np.repeat(
                test_data["thomson_scattering"]["t_e"][
                    : test_data["thomson_scattering"]["N_ch"][0], None
                ],
                test_data["thomson_scattering"]["N_time"][0],
                axis=1,
            ),
            np.repeat(
                test_data["thomson_scattering"]["t_e"][
                    test_data["thomson_scattering"]["N_ch"][0] :, None
                ],
                test_data["thomson_scattering"]["N_time"][1],
                axis=1,
            ),
        ]
    )
    flat["thomson_scattering.channel.n_e.time"] = ak.concatenate(
        [
            np.tile(
                test_data["thomson_scattering"]["time"][0],
                (test_data["thomson_scattering"]["N_ch"][0], 1),
            ),
            np.tile(
                test_data["thomson_scattering"]["time"][1],
                (test_data["thomson_scattering"]["N_ch"][1], 1),
            ),
        ]
    )
    flat["thomson_scattering.channel.n_e.data"] = ak.concatenate(
        [
            np.repeat(
                test_data["thomson_scattering"]["n_e"][
                    : test_data["thomson_scattering"]["N_ch"][0], None
                ],
                test_data["thomson_scattering"]["N_time"][0],
                axis=1,
            ),
            np.repeat(
                test_data["thomson_scattering"]["n_e"][
                    test_data["thomson_scattering"]["N_ch"][0] :, None
                ],
                test_data["thomson_scattering"]["N_time"][1],
                axis=1,
            ),
        ]
    )
    flat["thomson_scattering.channel.position.r"] = test_data["thomson_scattering"]["r"]
    flat["thomson_scattering.channel.position.z"] = test_data["thomson_scattering"]["z"]
    return flat


@pytest.fixture
def test_ids_dict(test_data):
    factory = IDSFactory("3.41.0")
    equilibrium = factory.equilibrium()
    equilibrium.time = test_data["equilibrium"]["time"]
    equilibrium.time_slice.resize(test_data["equilibrium"]["N_time"])
    equilibrium.ids_properties.homogeneous_time = 1
    for i in range(test_data["equilibrium"]["N_time"]):
        equilibrium.time_slice[i].time = test_data["equilibrium"]["time"][i]
        equilibrium.time_slice[i].profiles_1d.psi = test_data["equilibrium"]["psi_1d"]
        equilibrium.time_slice[i].profiles_2d.resize(1)
        equilibrium.time_slice[i].profiles_2d[0].grid.dim1 = test_data["equilibrium"][
            "r"
        ]
        equilibrium.time_slice[i].profiles_2d[0].grid.dim2 = test_data["equilibrium"][
            "z"
        ]
        equilibrium.time_slice[i].profiles_2d[0].psi = test_data["equilibrium"][
            "psi_2d"
        ]

    thomson_scattering = factory.thomson_scattering()
    thomson_scattering.ids_properties.homogeneous_time = 0
    N = (
        test_data["thomson_scattering"]["N_ch"][0]
        + test_data["thomson_scattering"]["N_ch"][1]
    )
    thomson_scattering.channel.resize(N)
    index = 0
    for i in range(N):
        if i == test_data["thomson_scattering"]["N_ch"][0]:
            index = 1
        thomson_scattering.channel[i].identifier = test_data["thomson_scattering"][
            "identifier"
        ][i]
        thomson_scattering.channel[i].t_e.time = test_data["thomson_scattering"][
            "time"
        ][index]
        thomson_scattering.channel[i].t_e.data = np.tile(
            test_data["thomson_scattering"]["t_e"][i],
            test_data["thomson_scattering"]["N_time"][index],
        )
        thomson_scattering.channel[i].n_e.time = test_data["thomson_scattering"][
            "time"
        ][index]
        thomson_scattering.channel[i].n_e.data = np.tile(
            test_data["thomson_scattering"]["n_e"][i],
            test_data["thomson_scattering"]["N_time"][index],
        )
        thomson_scattering.channel[i].position.r = test_data["thomson_scattering"]["r"][
            i
        ]
        thomson_scattering.channel[i].position.z = test_data["thomson_scattering"]["z"][
            i
        ]

    return {"equilibrium": equilibrium, "thomson_scattering": thomson_scattering}


def test_wrangle(test_ids_dict, flat):
    wrangled = wrangle(flat, version="3.41.0")
    for key in test_ids_dict:
        diff = idsdiffgen(wrangled[key], test_ids_dict[key])
        assert len(list(diff)) == 0, diff


def get_dtype(arr):
    """Get dtype from either numpy or awkward array."""
    if isinstance(arr, ak.Array):
        # Extract the numpy dtype from an awkward array via its typestr
        return eval("np." + arr.typestr.split("*")[-1])
    if hasattr(arr, "dtype"):
        return arr.dtype
    else:
        return type(arr)


def test_unwrangle(test_ids_dict, flat):
    result = unwrangle(list(flat.keys()), test_ids_dict)
    failed = [k for k in flat.keys() if k not in result]
    assert len(failed) == 0, f"The following fields failed to load {failed}"
    for key in flat.keys():
        if np.issubdtype(get_dtype(result[key]), np.floating):
            assert ak.almost_equal(result[key], flat[key])
        else:
            assert ak.array_equal(result[key], flat[key])
