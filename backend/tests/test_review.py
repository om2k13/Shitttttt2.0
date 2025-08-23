
"""
Test file for code review analysis
Contains various code quality issues for testing
"""

import os
import sys
from typing import List, Dict, Optional

def calculate_factorial(n):
    """Calculate factorial of n"""
    if n < 0:
        return None
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def process_data(data):
    """Process data with potential issues"""
    if not data:
        return []
    
    processed = []
    for item in data:
        if item > 0:
            processed.append(item * 2)
    
    return processed

def complex_function(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    """Function with too many parameters"""
    result = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y + z
    return result

def hardcoded_values():
    """Function with hardcoded values"""
    api_key = "sk-1234567890abcdef"
    database_url = "postgresql://user:pass@localhost:5432/db"
    timeout = 30000
    max_retries = 5
    
    return {
        "api_key": api_key,
        "database_url": database_url,
        "timeout": timeout,
        "max_retries": max_retries
    }

def unused_variables():
    """Function with unused variables"""
    important_data = "very important"
    temp_var = "temporary"
    unused_var = "never used"
    
    print(important_data)
    return temp_var

def exception_handling():
    """Function with poor exception handling"""
    try:
        result = 10 / 0
    except:
        print("Something went wrong")
    
    return result

if __name__ == "__main__":
    # Test the functions
    print(calculate_factorial(5))
    print(process_data([1, 2, 3, 4, 5]))
    print(complex_function(*range(26)))
    print(hardcoded_values())
    print(unused_variables())
    print(exception_handling())
