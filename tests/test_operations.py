
import numpy as np

from operations import (
    add_vectors,
    multiply_vectors,
    dot_product,
    vector_magnitude,
)


def test_vector_addition():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    result = add_vectors(a, b)

    np.testing.assert_array_equal(
        result,
        np.array([5, 7, 9]),
    )


def test_vector_multiplication():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    result = multiply_vectors(a, b)

    np.testing.assert_array_equal(
        result,
        np.array([4, 10, 18]),
    )


def test_dot_product():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    result = dot_product(a, b)

    assert result == 32


def test_zero_vectors():
    a = np.zeros(3)
    b = np.zeros(3)

    result = add_vectors(a, b)

    np.testing.assert_array_equal(
        result,
        np.zeros(3),
    )


def test_vector_magnitude():
    a = np.array([3, 4])

    result = vector_magnitude(a)

    assert result == 5.0

def test_vector_magnitude_zero():
    a = np.array([0,0,0])

    result = vector_magnitude(a)

    assert result == 0.0

def test_vector_magnitude_negative():
    a = np.array([-3, -4])

    result = vector_magnitude(a)

    assert result == 5.0