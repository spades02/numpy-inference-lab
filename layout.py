def element_offset(strides, i, j):
    """Byte offset of element [i, j] from the start of the array."""
    return i * strides[0] + j * strides[1]