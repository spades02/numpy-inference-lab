import numpy as np
import pytest

from layout import element_offset


def address_of(a, i, j):
    """Real memory address of a[i, j], read from NumPy itself."""
    return a[i:, j:].__array_interface__["data"][0]


@pytest.mark.parametrize("shape", [(4, 5), (3, 7), (6, 12)])
def test_c_order_matches_numpy(shape):
    a = np.zeros(shape)
    base = address_of(a, 0, 0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert base + element_offset(a.strides, i, j) == address_of(a, i, j)


def test_transpose_is_a_view_with_swapped_strides():
    a = np.zeros((4, 5))
    t = a.T
    assert np.shares_memory(a, t)
    assert t.strides == (a.strides[1], a.strides[0])
    assert not t.flags["C_CONTIGUOUS"]


def test_transposed_offsets_match_numpy():
    t = np.zeros((4, 5)).T
    base = address_of(t, 0, 0)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            assert base + element_offset(t.strides, i, j) == address_of(t, i, j)


def test_row_neighbours_are_8_bytes_apart():
    a = np.zeros((6, 12))
    assert element_offset(a.strides, 2, 4) - element_offset(a.strides, 2, 3) == 8
    assert element_offset(a.strides, 3, 3) - element_offset(a.strides, 2, 3) == 96