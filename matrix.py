import numpy as np

def matmul(A, B):

    if not A or not B or not A[0] or not B[0]:
        raise ValueError("Matrices must be non-empty")

    if any(len(row) != len(A[0]) for row in A):
        raise ValueError("A must be rectangular")

    if any(len(row) != len(B[0]) for row in B):
        raise ValueError("B must be rectangular")

    m = len(A)
    k = len(A[0])
    n = len(B[0])

    if k != len(B):
        raise ValueError("Matrices are incompatible")

    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0 
            for t in range(k):
                s += A[i][t] * B[t][j] 
            C[i][j] = s

    return C
 
    

def matmul_numpy(A,B):
    A = np.asarray(A)
    B = np.asarray(B)

    return A @ B

