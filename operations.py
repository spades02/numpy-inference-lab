
import numpy as np


def add_vectors(a, b):
    """Add two vectors element by element."""
    return np.add(a, b)


def multiply_vectors(a, b):
    """Multiply two vectors element by element."""
    return np.multiply(a, b)


def dot_product(a, b):
    """Calculate the dot product of two vectors."""
    return np.dot(a, b)

# This implementation was done before getting familiar with NumPy
# def vector_magnitude(a):
#     result = 0
#     for n in a:
#         result = n * n + result
#     return np.sqrt(result)
def vector_magnitude(a):
    return np.sqrt(np.sum(a ** 2))

if __name__ == "__main__":
    a = np.array([-3, -4])
    b = np.array([4, 5, 6])

    
    print("Vector magnitude:", vector_magnitude(a))
