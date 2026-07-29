def fibonacci_series(n):
    """Generate a list containing the Fibonacci series up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    series = [0, 1]
    for i in range(2, n):
        next_value = series[-1] + series[-2]
        series.append(next_value)
    
    return series

if __name__ == "__main__":
    num_terms = int(input("Enter the number of terms: "))
    print(f"Fibonacci series up to {num_terms} terms: {fibonacci_series(num_terms)}")
