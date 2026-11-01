
import numpy as np
import pytest

from matrix import matmul, matmul_numpy


@pytest.mark.parametrize(
    "A, B, expected",
    [
        (
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[19, 22], [43, 50]],
        ),
        (
            [[1, 2, 3]],
            [[4], [5], [6]],
            [[32]],
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[58, 64], [139, 154]],
        ),
    ],
)
def test_matrix_multiplication(A, B, expected):
    python_result = matmul(A, B)
    numpy_result = matmul_numpy(A, B)

    np.testing.assert_array_equal(
        python_result,
        expected,
    )

    np.testing.assert_array_equal(
        numpy_result,
        expected,
    )


def test_incompatible_dimensions():
    A = [[1, 2, 3]]
    B = [[4, 5], [6, 7]]

    with pytest.raises(ValueError):
        matmul(A, B)

    with pytest.raises(ValueError):
        matmul_numpy(A, B)


def test_empty_matrix():
    with pytest.raises(ValueError):
        matmul([], [[1]])


def test_rectangular_output_shape():
    A = [[1, 2], [3, 4], [5, 6]]
    B = [[7, 8], [9, 10]]

    expected = [
        [25, 28],
        [57, 64],
        [89, 100],
    ]

    assert matmul(A, B) == expected
