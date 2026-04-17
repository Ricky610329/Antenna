"""單元測試：antenna.utils.data 的 helpers 與 DataManager。"""

import numpy as np
import pytest
import torch

from antenna.utils.data import (
    DataManager,
    atomic_pickle_save,
    compute_data_id,
    ensure_tensor,
    make_hashable,
    numpy_to_tensor,
    pickle_load,
    tensor_to_numpy,
    timestamped_backup,
)

# ----------------------------------------------------------------------------
# make_hashable
# ----------------------------------------------------------------------------


def test_make_hashable_primitives_pass_through():
    assert make_hashable(1) == 1
    assert make_hashable(1.5) == 1.5
    assert make_hashable("hi") == "hi"
    assert make_hashable(None) is None
    assert make_hashable(b"bytes") == b"bytes"


def test_make_hashable_tensor_returns_bytes():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = make_hashable(tensor)
    assert isinstance(result, bytes)


def test_make_hashable_list_returns_tuple_recursively():
    result = make_hashable([1, [2, 3], (4, 5)])
    assert result == (1, (2, 3), (4, 5))
    assert isinstance(result, tuple)


def test_make_hashable_dict_is_sorted_tuple():
    d = {"b": 1, "a": 2}
    result = make_hashable(d)
    assert result == (("a", 2), ("b", 1))


def test_make_hashable_set_returns_frozenset():
    result = make_hashable({3, 1, 2})
    assert result == frozenset({1, 2, 3})


def test_make_hashable_unhashable_raises():
    class Unhashable:
        __hash__ = None

    with pytest.raises(TypeError):
        make_hashable(Unhashable())


def test_make_hashable_is_deterministic():
    """同樣的輸入應該產生同樣的 hashable 形式。"""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert make_hashable(tensor) == make_hashable(tensor.clone())

    nested = {"x": [1, 2, {"a": 1}], "y": np.array([1, 2])}
    assert make_hashable(nested) == make_hashable({"y": np.array([1, 2]), "x": [1, 2, {"a": 1}]})


def test_make_hashable_is_sensitive_to_changes():
    """不同輸入應該產生不同的 hashable 形式。"""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 3.0])
    assert make_hashable(a) != make_hashable(b)

    assert make_hashable([1, 2, 3]) != make_hashable([1, 2, 4])
    assert make_hashable({"a": 1}) != make_hashable({"a": 2})


# ----------------------------------------------------------------------------
# compute_data_id
# ----------------------------------------------------------------------------


def test_compute_data_id_deterministic():
    a = [1, 2, {"k": torch.tensor([1.0, 2.0])}]
    b = [1, 2, {"k": torch.tensor([1.0, 2.0])}]
    assert compute_data_id(a) == compute_data_id(b)


def test_compute_data_id_sensitive():
    a = [1, 2, 3]
    b = [1, 2, 4]
    assert compute_data_id(a) != compute_data_id(b)


def test_compute_data_id_returns_md5_hex():
    result = compute_data_id([1, 2, 3])
    assert isinstance(result, str)
    assert len(result) == 32  # MD5 hex digest


# ----------------------------------------------------------------------------
# tensor ↔ numpy
# ----------------------------------------------------------------------------


def test_tensor_to_numpy_roundtrip():
    original = torch.tensor([1.0, 2.0, 3.5, -4.25])
    arr = tensor_to_numpy(original)
    assert isinstance(arr, np.ndarray)
    recovered = numpy_to_tensor(arr)
    assert torch.allclose(original, recovered)


def test_numpy_to_tensor_roundtrip_preserves_values():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = numpy_to_tensor(arr)
    arr2 = tensor_to_numpy(tensor)
    assert np.allclose(arr, arr2)


def test_tensor_to_numpy_detaches_and_moves_cpu():
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    arr = tensor_to_numpy(tensor)
    assert isinstance(arr, np.ndarray)
    # 原 tensor 仍保留 grad 需求
    assert tensor.requires_grad


def test_tensor_to_numpy_type_check():
    with pytest.raises(TypeError):
        tensor_to_numpy([1, 2, 3])


def test_numpy_to_tensor_type_check():
    with pytest.raises(TypeError):
        numpy_to_tensor([1, 2, 3])


def test_numpy_to_tensor_with_dtype():
    arr = np.array([1, 2, 3])
    tensor = numpy_to_tensor(arr, dtype=torch.float32)
    assert tensor.dtype == torch.float32


def test_ensure_tensor_from_tensor():
    t = torch.tensor([1.0, 2.0])
    assert torch.allclose(ensure_tensor(t), t)


def test_ensure_tensor_from_list():
    t = ensure_tensor([1, 2, 3], dtype=torch.float32)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))


def test_ensure_tensor_from_numpy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = ensure_tensor(arr)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------------
# pickle I/O
# ----------------------------------------------------------------------------


def test_atomic_pickle_save_and_load(tmp_path):
    path = tmp_path / "foo.pkl"
    payload = {"a": 1, "b": [1, 2, 3]}
    atomic_pickle_save(path, payload)
    assert path.exists()
    loaded = pickle_load(path)
    assert loaded == payload


def test_atomic_pickle_save_creates_parent(tmp_path):
    path = tmp_path / "deeper" / "dir" / "foo.pkl"
    atomic_pickle_save(path, [1, 2])
    assert path.exists()


def test_pickle_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pickle_load(tmp_path / "nonexistent.pkl")


def test_timestamped_backup(tmp_path):
    src = tmp_path / "data.pkl"
    src.write_bytes(b"hello")
    backup_path = timestamped_backup(src, name="data", suffix="bak")
    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path.read_bytes() == b"hello"


def test_timestamped_backup_missing_returns_none(tmp_path):
    result = timestamped_backup(tmp_path / "not_there.pkl")
    assert result is None


# ----------------------------------------------------------------------------
# DataManager
# ----------------------------------------------------------------------------


def test_data_manager_append_and_retrieve(tmp_path):
    dm = DataManager("unit_test", rootdir=str(tmp_path), verbose=False)
    sample = torch.tensor([1.0, 2.0, 3.0])
    label = torch.tensor([0.5])
    dm.add_and_save([[sample, label]])

    assert len(dm) == 1
    got_sample, got_label = dm[0]
    assert torch.allclose(got_sample, sample)
    assert torch.allclose(got_label, label)


def test_data_manager_persists_across_instances(tmp_path):
    sample = torch.tensor([1.0, 2.0])
    label = torch.tensor([1.0])

    dm1 = DataManager("persist_test", rootdir=str(tmp_path), verbose=False)
    dm1.add_and_save([[sample, label]])
    assert len(dm1) == 1

    # 建立第二個實例應從檔案載入
    dm2 = DataManager("persist_test", rootdir=str(tmp_path), verbose=False)
    assert len(dm2) == 1
    got_sample, got_label = dm2[0]
    assert torch.allclose(got_sample, sample)
    assert torch.allclose(got_label, label)


def test_data_manager_deduplicates_on_append(tmp_path):
    dm = DataManager("dedup_test", rootdir=str(tmp_path), verbose=False)
    sample = torch.tensor([1.0, 2.0])
    label = torch.tensor([0.0])
    dm.add_and_save([[sample, label]])
    dm.add_and_save([[sample, label]])  # 重複
    assert len(dm) == 1


def test_data_manager_batch_append(tmp_path):
    dm = DataManager("batch_test", rootdir=str(tmp_path), verbose=False)
    batch = [
        [torch.tensor([1.0]), torch.tensor([0.0])],
        [torch.tensor([2.0]), torch.tensor([1.0])],
        [torch.tensor([3.0]), torch.tensor([2.0])],
    ]
    dm.add_and_save(batch)
    assert len(dm) == 3


def test_data_manager_overwrite_mode(tmp_path):
    dm = DataManager("overwrite_test", rootdir=str(tmp_path), verbose=False)
    dm.add_and_save([[torch.tensor([1.0]), torch.tensor([0.0])]])
    assert len(dm) == 1

    dm.add_and_save([[torch.tensor([9.0]), torch.tensor([9.0])]], mode="overwrite")
    assert len(dm) == 1
    got_sample, _ = dm[0]
    assert torch.allclose(got_sample, torch.tensor([9.0]))


def test_data_manager_empty_input_warns(tmp_path):
    dm = DataManager("empty_test", rootdir=str(tmp_path), verbose=False)
    dm.add_and_save([])
    assert len(dm) == 0


def test_data_manager_index_error_when_empty(tmp_path):
    dm = DataManager("ix_test", rootdir=str(tmp_path), verbose=False)
    with pytest.raises(IndexError):
        dm[0]
