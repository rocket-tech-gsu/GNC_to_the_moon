from sympy import symbols, Matrix, pprint

# Define symbols
x1, x2, x3, x7, x8, x9 = symbols('x1 x2 x3 x7 x8 x9')

# Create a symbolic matrix
matrix = Matrix([
    ["cos(x8)*cos(x7)", "-cos(x8)*sin(x7)", "sin(x7)", "cos(x8)*x1 + sin(x8)*x3"],
    ["(-sin(x9)*sin(x8)*cos(x7)) + (cos(x9)*sin(x7))", "(sin(x9)*sin(x8)*cos(x7)) + (cos(x9)*cos(x7))", "sin(x9)*cos(x8)", "cos(x9)*x2 + sin(x9)*(-sin(x8)*x1 + cos(x8)*x3)"],
    ["-cos(x9)*sin(x8)*cos(x7) - sin(x9)*sin(x7)", "cos(x9)*sin(x8)*cos(x7) - sin(x9)*cos(x7)", "cos(x9)*cos(x8)", "-sin(x9)*x2 + cos(x9)*(-sin(x8)*x1 + cos(x8)*x3)"],
    ["0", "0", "0", "x7"]
])

# Pretty print the matrix
print("Symbolic Matrix:")
pprint(matrix)