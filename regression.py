import numpy as np
from typing import List, Tuple, Dict, Callable

def parse_data(data_str: str) -> List[Tuple[float, float]]:
    """Parse input data string to list of (x, y) tuples"""
    points = []
    for line in data_str.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))
            except ValueError:
                continue
    return points

def linear_regression(points: List[Tuple[float, float]]) -> Dict:
    """Linear regression: y = mx + c"""
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_x2 = sum(p[0] ** 2 for p in points)
    
    det = n * sum_x2 - sum_x ** 2
    m = (n * sum_xy - sum_x * sum_y) / det
    c = (sum_x2 * sum_y - sum_x * sum_xy) / det
    
    # Build steps
    steps = []
    steps.append(f"Step 1: Given data points (n = {n})")
    steps.append(f"Σx = {sum_x:.4f}, Σy = {sum_y:.4f}")
    steps.append(f"Σx² = {sum_x2:.4f}, Σxy = {sum_xy:.4f}")
    steps.append(f"Step 2: Normal Equations")
    steps.append(f"{sum_y:.4f} = {n}c + {sum_x:.4f}m")
    steps.append(f"{sum_xy:.4f} = {sum_x:.4f}c + {sum_x2:.4f}m")
    steps.append(f"Step 3: Solve")
    steps.append(f"m = {m:.4f}, c = {c:.4f}")
    
    equation = f"y = {m:.4f}x + {c:.4f}"
    
    def predict(x: float) -> float:
        return m * x + c
    
    # Generate graph points
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    return {
        "method": "linear",
        "equation": equation,
        "coefficients": {"m": m, "c": c},
        "steps": steps,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def polynomial_regression(points: List[Tuple[float, float]], degree: int) -> Dict:
    """Polynomial regression: y = a0 + a1*x + a2*x² + ..."""
    n = len(points)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    # Build normal equations matrix
    N = degree + 1
    A = [[0.0] * N for _ in range(N)]
    B = [0.0] * N
    
    for i in range(N):
        for j in range(N):
            A[i][j] = sum(x ** (i + j) for x in x_vals)
        B[i] = sum((x ** i) * y for x, y in points)
    
    # Gaussian elimination
    coeffs = gauss_elimination(A, B)
    
    # Build equation string
    terms = []
    for i, coef in enumerate(coeffs):
        if i == 0:
            terms.append(f"{coef:.4f}")
        elif i == 1:
            terms.append(f"{coef:.4f}x")
        else:
            terms.append(f"{coef:.4f}x^{i}")
    equation = "y = " + " + ".join(terms).replace("+ -", "- ")
    
    def predict(x: float) -> float:
        return sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))
    
    # Graph
    min_x = min(x_vals)
    max_x = max(x_vals)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    steps = [f"Polynomial degree {degree} regression", f"Coefficients: {coeffs}"]
    
    return {
        "method": "polynomial",
        "equation": equation,
        "coefficients": coeffs,
        "steps": steps,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def exponential_regression(points: List[Tuple[float, float]]) -> Dict:
    """Exponential: y = a*e^(bx)"""
    # Transform: ln(y) = ln(a) + bx
    transformed = [(x, np.log(y)) for x, y in points if y > 0]
    n = len(transformed)
    
    sum_x = sum(p[0] for p in transformed)
    sum_y = sum(p[1] for p in transformed)
    sum_xy = sum(p[0] * p[1] for p in transformed)
    sum_x2 = sum(p[0] ** 2 for p in transformed)
    
    det = n * sum_x2 - sum_x ** 2
    b = (n * sum_xy - sum_x * sum_y) / det
    A = (sum_x2 * sum_y - sum_x * sum_xy) / det
    a = np.exp(A)
    
    equation = f"y = {a:.4f}·e^({b:.4f}x)"
    
    def predict(x: float) -> float:
        return a * np.exp(b * x)
    
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    steps = ["Exponential model: y = a·e^(bx)", f"ln(y) = ln(a) + bx", f"a = {a:.4f}, b = {b:.4f}"]
    
    return {
        "method": "exponential",
        "equation": equation,
        "coefficients": {"a": a, "b": b},
        "steps": steps,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def logarithmic_regression(points: List[Tuple[float, float]]) -> Dict:
    """Logarithmic: y = a·ln(x) + b"""
    valid = [(x, y) for x, y in points if x > 0]
    n = len(valid)
    
    # Transform: X = ln(x), then y = a*X + b
    transformed = [(np.log(x), y) for x, y in valid]
    
    sum_x = sum(p[0] for p in transformed)
    sum_y = sum(p[1] for p in transformed)
    sum_xy = sum(p[0] * p[1] for p in transformed)
    sum_x2 = sum(p[0] ** 2 for p in transformed)
    
    det = n * sum_x2 - sum_x ** 2
    a = (n * sum_xy - sum_x * sum_y) / det
    b = (sum_x2 * sum_y - sum_x * sum_xy) / det
    
    equation = f"y = {a:.4f}·ln(x) + {b:.4f}"
    
    def predict(x: float) -> float:
        return a * np.log(x) + b
    
    min_x = min(p[0] for p in valid)
    max_x = max(p[0] for p in valid)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    steps = ["Logarithmic model: y = a·ln(x) + b", f"a = {a:.4f}, b = {b:.4f}"]
    
    return {
        "method": "logarithmic",
        "equation": equation,
        "coefficients": {"a": a, "b": b},
        "steps": steps,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def gauss_elimination(A: List[List[float]], B: List[float]) -> List[float]:
    """Solve Ax = B using Gaussian elimination with partial pivoting"""
    n = len(A)
    # Create copies
    a = [row[:] for row in A]
    b = B[:]
    
    for i in range(n):
        # Partial pivoting
        max_row = i
        for k in range(i + 1, n):
            if abs(a[k][i]) > abs(a[max_row][i]):
                max_row = k
        a[i], a[max_row] = a[max_row], a[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Elimination
        for k in range(i + 1, n):
            factor = a[k][i] / a[i][i]
            b[k] -= factor * b[i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / a[i][i]
        for k in range(i - 1, -1, -1):
            b[k] -= a[k][i] * x[i]
    
    return x