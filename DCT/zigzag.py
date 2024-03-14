import numpy as np

def zigzag_scan(matrix):
    """
    Perform a zigzag scan on the input matrix.
    
    Args:
        matrix (numpy.ndarray): The input matrix.
        
    Returns:
        numpy.ndarray: The flattened array after zigzag scan.
    """
    # Check if the input is a valid 2D matrix
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    
    rows, cols = matrix.shape
    output = np.zeros(rows * cols)
    
    # Helper function for zigzag scan
    def scan(row, col, direction):
        index = 0
        while 0 <= row < rows and 0 <= col < cols:
            output[index] = matrix[row, col]
            index += 1
            row += direction[0]
            col += direction[1]
        return index
    
    index = 0
    for diag in range(rows + cols - 1):
        if diag % 2 == 0:  # Going up
            row = 0 if diag < cols else diag - cols + 1
            col = diag if diag < cols else cols - 1
            index += scan(row, col, (1, -1))
        else:  # Going down
            row = diag if diag < rows else rows - 1
            col = 0 if diag >= rows else diag
            index += scan(row, col, (-1, 1))
    
    return output

def inverse_zigzag_scan(flattened, rows, cols):
    """
    Perform an inverse zigzag scan to reconstruct the matrix from the flattened array.
    
    Args:
        flattened (numpy.ndarray): The flattened array after zigzag scan.
        rows (int): The number of rows in the output matrix.
        cols (int): The number of columns in the output matrix.
        
    Returns:
        numpy.ndarray: The reconstructed matrix.
    """
    output = np.zeros((rows, cols))
    
    # Helper function for inverse zigzag scan
    def scan(row, col, direction):
        index = 0
        while 0 <= row < rows and 0 <= col < cols:
            output[row, col] = flattened[index]
            index += 1
            row += direction[0]
            col += direction[1]
        return index
    
    index = 0
    for diag in range(rows + cols - 1):
        if diag % 2 == 0:  # Going up
            row = 0 if diag < cols else diag - cols + 1
            col = diag if diag < cols else cols - 1
            index += scan(row, col, (1, -1))
        else:  # Going down
            row = diag if diag < rows else rows - 1
            col = 0 if diag >= rows else diag
            index += scan(row, col, (-1, 1))
    
    return output

__all__ = ['zigzag_scan', 'inverse_zigzag_scan']