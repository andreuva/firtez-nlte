def print_eq_system(A, b, x_vars):
    """
    Prints a system of equations in Ax = b matrix form.
    
    Args:
        A (list of lists or np.ndarray): The coefficient matrix.
        b (list or np.ndarray): The constant vector.
        x_vars (list of str): The names of the variables.
    """
    # Safely check for empty arrays/lists using length
    if len(A) == 0 or len(b) == 0 or len(x_vars) == 0:
        print("Empty system.")
        return
        
    if len(A) != len(b) or len(A[0]) != len(x_vars):
        print("Error: Matrix and vector dimensions do not match.")
        return

    # Format values to scientific notation first to find their exact string length
    fmt_A = [[f"{val:.3e}" for val in row] for row in A]
    fmt_b = [f"{val:.3e}" for val in b]
    
    # Find maximum widths for flawless alignment
    max_val_width = max(len(val) for row in fmt_A for val in row)
    max_val_width = max(max_val_width, max(len(val) for val in fmt_b))
    max_x_width = max(len(str(x)) for x in x_vars)
    
    pad = max_val_width + 1

    # Find the middle row to print the equals sign
    mid_row = (len(A) - 1) // 2

    print("The system of equations in Ax = b form:")

    # Loop through each row of the system
    for i in range(len(A)):
        
        # Format the A matrix row
        a_row = "[ " + " ".join(f"{val:>{pad}}" for val in fmt_A[i]) + " ]"
        
        # Format the x vector row (pad strings so brackets align vertically)
        x_row = f"[ {str(x_vars[i]):<{max_x_width}} ]"
        
        # Add the equals sign only to the middle row
        eq_sign = "=" if i == mid_row else " "
        
        # Format the b vector row
        b_row = f"[ {fmt_b[i]:>{pad}} ]"
        
        # Print the combined line
        print(f"{a_row} {x_row} {eq_sign} {b_row}")