
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


def vector_magnitude(a):
    return np.sqrt(np.sum(a ** 2))

if __name__ == "__main__":
    a = np.array([-3, -4])

    
    print("Vector magnitude:", vector_magnitude(a))
