import io 
import re
import time
from PIL import Image
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication
from sympy import (
    Eq, solve, sympify, solveset, S, latex,
    parse_expr, FiniteSet, nsolve, Symbol
)
from scipy.optimize import fsolve
import joblib
import os
import pandas as pd
import numpy as np
import sympy as sp  

if os.path.exists('solver_model.pkl'):
    model = joblib.load('solver_model.pkl')
else:
    model = None   

def extract_features(parsed_equations):
    """Extract features for ML prediction"""
    total_terms = 0
    max_deg = 0
    variables = set()
    
    for eq in parsed_equations:
        if isinstance(eq, Eq):
            expr = eq.lhs - eq.rhs
        else:
            expr = eq            
        terms = expr.as_ordered_terms()
        total_terms += len(terms) 
        
        try:
            deg = sp.degree(expr) if expr.is_polynomial() else 0
        except Exception: 
            deg = 0
        max_deg = max(max_deg, deg)
        
        variables.update(expr.free_symbols)
        
    return {
        'Degree': max_deg,
        'Num_Terms': total_terms,
        'Num_Variables': len(variables)
    }

def solve_equation_system_numeric(equations, user_guesses=None):
    """Solve system using SciPy's fsolve."""
    start_time = time.perf_counter()
    all_vars = sorted(list(set().union(*[eq.free_symbols for eq in equations])), key=str)
    
    residuals = []
    for eq in equations:
        if isinstance(eq, Eq):
            expr = eq.lhs - eq.rhs
        else:
            expr = eq
        func = sp.lambdify(all_vars, expr, 'numpy')
        residuals.append(func)
        
    def system_func(vals):
        return np.array([f(*vals) for f in residuals], dtype=float)
    
    guesses = [user_guesses.get(str(var), 1.0) for var in all_vars]
    try:
        sol = fsolve(system_func, guesses)
        result = {str(var): sol[i] for i, var in enumerate(all_vars)}
        numeric_error = None
    except Exception as e:
        result = None
        numeric_error = str(e)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    return result, elapsed_time, numeric_error

def algebra_solve(parsed_eqs,user_guesses):
        
        features = extract_features(parsed_eqs)
        if model is None:
            recommended_method = 'symbolic'
        else:
            recommended_method = model.predict([[features['Degree'], features['Num_Terms'], 
                                                  features['Num_Variables']]])[0]

        result = {}
        mcqs = []
        correct_idx = -1

        if recommended_method == 'symbolic':
            try:
                start = time.perf_counter()
                symbolic_sol = solve(parsed_eqs, dict=True)
                elapsed_time = (time.perf_counter() - start) * 1000
                
                result = {
                    'method': 'symbolic',
                    '_raw_solution': symbolic_sol,
                    'solution': format_solution(symbolic_sol),
                    'time_ms': f"{elapsed_time:.2f} ms",
                    'error': None
                }
            except Exception as e:
                
                fallback_error = e
                recommended_method = 'numeric' 
                try:
                    start = time.perf_counter()
                    numeric_sol, numeric_time, numeric_error = solve_equation_system_numeric(
                        parsed_eqs, user_guesses
                    )
                    result = {
                        'method': 'numeric',
                        '_raw_solution': numeric_sol,
                        'solution': format_dict_solution(numeric_sol),
                        'time_ms': f"{numeric_time:.2f} ms",
                        'error': f"Symbolic solver failed: {str(fallback_error)}. Numeric fallback executed."
                                 f" {numeric_error or ''}".strip()
                    }
                except Exception as e2:
                    result = {
                        'method': 'numeric',
                        '_raw_solution': None,
                        'solution': None,
                        'time_ms': None,
                        'error': f"Both symbolic and numeric solvers failed. Symbolic error: {str(fallback_error)}; "
                                 f"Numeric error: {str(e2)}"
                    }

        elif recommended_method == 'numeric':
            try:
                start = time.perf_counter()
                numeric_sol, numeric_time, numeric_error = solve_equation_system_numeric(
                    parsed_eqs, user_guesses
                )                
                result = {
                    'method': 'numeric',
                    'solution': format_dict_solution(numeric_sol),
                    'time_ms': f"{numeric_time:.2f} ms",
                    'error': numeric_error
                }
            except Exception as e:
                result = {
                    'method': 'numeric',
                    '_raw_solution': None,
                    'solution': None,
                    'time_ms': None,
                    'error': f"Numeric solver failed: {str(e)}"
                }

        return result


def format_solution(solution):
    if isinstance(solution, list):
        return "\n".join([f"Solution {i}:\n  " + "\n  ".join(f"{k} = {v.evalf():.2f}" for k, v in s.items())
                         for i, s in enumerate(solution, 1)])
    if isinstance(solution, FiniteSet):
        return f"Solution set: {solution}"
    return str(solution)

def format_dict_solution(solution_dict):
    if solution_dict:
        return ", ".join(f"{k} = {v:.2f}" for k, v in solution_dict.items())
    return "No solution"


def train_ml_model(training_data_file='training_data.csv'):
    
    data = pd.read_csv(r"C:\Users\vengi\Desktop\AIMATHSOLVER\my-react-app\src\data.csv")

    X = data[['Degree', 'Num_Terms', 'Num_Variables']]
    y = data['Best_Method']
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X, y)    
    joblib.dump(clf, 'solver_model.pkl')
    print("Training complete. Model saved as solver_model.pkl.")

