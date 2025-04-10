from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import time
from sympy import (
    Eq, solve, sympify, solveset, S, latex,
    parse_expr, FiniteSet
)
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication
import re
import random
import os
from io import BytesIO
from flask_cors import CORS
from sympy import nsolve, Symbol
from sympy.core.relational import Relational
import numpy as np
from scipy.optimize import fsolve
import sympy as sp  

app = Flask(__name__)
CORS(app)

import joblib 
model = joblib.load('solver_model.pkl')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MATH_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=(){}[]^/*%'
OCR_SUBSTITUTIONS = {
    r'\bmod\b': '^',
    r'\{': '(',
    r'\}': ')',
    r'\[': '(',
    r'\]': ')',
    r'\s*([/])\s*': r'\1',
    r'\bO\b': '0',
    r'\bI\b': '1'
}

def preprocess_ocr_text(text):
    text = re.sub(r'\s*([+/*^\-=])\s*', r'\1', text)
    for p, r in OCR_SUBSTITUTIONS.items():
        text = re.sub(p, r, text)
    return text.strip()

def image_to_equation(image):
    cleaned_text = preprocess_ocr_text(pytesseract.image_to_string(image, config=MATH_CONFIG))
    for sep in ['\n', ';', 'and']:
        if sep in cleaned_text:
            return [eq.strip() for eq in cleaned_text.split(sep) if eq.strip()]
    return [cleaned_text]

def parse_equation(equation_list):
    transformations = standard_transformations + (implicit_multiplication,)
    parsed = []
    for eq in equation_list:
        if '=' in eq:
            lhs, rhs = eq.split('=', 1)
            parsed.append(Eq(parse_expr(lhs, transformations=transformations),
                             parse_expr(rhs, transformations=transformations)))
        else:
            parsed.append(parse_expr(eq, transformations=transformations))
    return parsed

def is_complex_equation(equations, symbol_threshold=5, ops_threshold=50, degree_threshold=2):
    total_symbols = sum(len(eq.free_symbols) for eq in equations)
    total_ops = sum(len(eq.atoms(Relational)) + len(eq.atoms()) for eq in equations)

    max_degree = 0
    for eq in equations:
        if isinstance(eq, Relational):
            poly = eq.lhs - eq.rhs
        else:
            poly = eq
        try:
            deg = poly.as_poly().total_degree()
        except Exception:
            deg = 0  # Non-polynomial, fallback
        max_degree = max(max_degree, deg)

    return (
        total_symbols > symbol_threshold or
        total_ops > ops_threshold or
        max_degree > degree_threshold
    )

def solve_equation_system_hybrid(equations, user_guesses=None):
    start_time = time.perf_counter()
    try:
        all_vars = list(set().union(*[eq.free_symbols for eq in equations]))
        guesses = [user_guesses.get(str(var), 1.0) for var in all_vars]

        if is_complex_equation(equations):
            numeric_sol = nsolve(equations, all_vars, guesses)
            result = [{str(var): numeric_sol[i].evalf() for i, var in enumerate(all_vars)}]
        else:
            result = solve(equations, dict=True)
            if not result: 
                numeric_sol = nsolve(equations, all_vars, guesses)
                result = [{str(var): numeric_sol[i].evalf() for i, var in enumerate(all_vars)}]
    except Exception:
        all_vars = list(set().union(*[eq.free_symbols for eq in equations]))
        guesses = [user_guesses.get(str(var), 1.0) for var in all_vars]
        numeric_sol = nsolve(equations, all_vars, guesses)
        result = [{str(var): numeric_sol[i].evalf() for i, var in enumerate(all_vars)}]

    elapsed_time = (time.perf_counter() - start_time) * 1000
    return result, elapsed_time


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

def generate_mcqs(solution):
    if isinstance(solution, FiniteSet):
        value = next(iter(solution))
        options = {float(value) + o for o in random.sample([-2.5, -1.5, 1, 1.5], 3)} | {float(value)}
        options = list(options)
        random.shuffle(options)
        idx = options.index(float(value))
        return [f"{opt:.2f}" for opt in options], idx

    if isinstance(solution, list) and solution:
        first_sol = solution[0]
        if len(first_sol) == 1:
            value = next(iter(first_sol.values()))
            options = {float(value) + o for o in random.sample([-2.5, -1.5, 1, 1.5], 3)} | {float(value)}
            options = list(options)
            random.shuffle(options)
            idx = options.index(float(value))
            return [f"{opt:.2f}" for opt in options], idx
        else:
            correct = {str(k): float(v) for k, v in first_sol.items()}
            distractors = [{
                var: round(val + random.choice([-2, -1.5, 1, 2]) + random.uniform(-0.5, 0.5), 2)
                for var, val in correct.items()
            } for _ in range(3)]
            options = [correct] + distractors
            random.shuffle(options)
            formatted = [", ".join(f"{k} = {v:.2f}" for k, v in o.items()) for o in options]
            idx = formatted.index(", ".join(f"{k} = {v:.2f}" for k, v in correct.items()))
            return formatted, idx
    return [], -1



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
            
        # Count terms
        terms = expr.as_ordered_terms()
        total_terms += len(terms)
        
        # Get degree
        try:
            deg = degree(expr)
        except:
            deg = 0
        max_deg = max(max_deg, deg)
        
        # Collect variables
        variables.update(expr.free_symbols)
        
    return {
        'Degree': max_deg,
        'Num_Terms': total_terms,
        'Num_Variables': len(variables)
    }



@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(BytesIO(file.read()))
    
    initial_guesses_str = request.form.get('initial_guesses', '')
    user_guesses = {}
    if initial_guesses_str:
        try:
            for pair in initial_guesses_str.split(','):
                var, val = pair.split('=')
                user_guesses[var.strip()] = float(val.strip())
        except Exception:
            return jsonify({'error': 'Invalid format for initial guesses. Use var=value format, comma-separated.'}), 400

    try:
        eq_texts = image_to_equation(image)
        parsed_eqs = parse_equation(eq_texts)
        parsed_latex = [latex(eq) for eq in parsed_eqs]
        
        # ----- Symbolic Solve -----
        try:
            start_sym = time.perf_counter()
            symbolic_sol = solve(parsed_eqs, dict=True)
            symbolic_time = (time.perf_counter() - start_sym) * 1000
            symbolic_error = None
        except Exception as e:
            symbolic_sol = None
            symbolic_time = 0
            symbolic_error = str(e)
            
        symbolic_solution_str = format_solution(symbolic_sol) if symbolic_sol else symbolic_error

        # ----- Hybrid Solve -----
        try:
            hybrid_sol, hybrid_time = solve_equation_system_hybrid(parsed_eqs, user_guesses)
            hybrid_error = None
        except Exception as e:
            hybrid_sol = None
            hybrid_time = 0
            hybrid_error = str(e)
        hybrid_solution_str = format_solution(hybrid_sol) if hybrid_sol else hybrid_error

        # ----- Numerical Solve with SciPy -----
        numeric_sol, numeric_time, numeric_error = solve_equation_system_numeric(parsed_eqs, user_guesses)
        numeric_solution_str = format_dict_solution(numeric_sol) if numeric_sol else numeric_error

        # Generate MCQs based on hybrid solution (if available)
        mcqs, correct_idx = generate_mcqs(hybrid_sol) if hybrid_sol else ([], -1)

        return jsonify({
            'equations': parsed_latex,
            'symbolic': {
                'solution': symbolic_solution_str,
                'time_ms': f"{symbolic_time:.2f} ms",
                'error': symbolic_error
            },
            'hybrid': {
                'solution': hybrid_solution_str,
                'time_ms': f"{hybrid_time:.2f} ms",
                'error': hybrid_error
            },
            'numeric': {
                'solution': numeric_solution_str,
                'time_ms': f"{numeric_time:.2f} ms",
                'error': numeric_error
            },
            'mcqs': [f"{chr(65+i)}: {opt}" for i, opt in enumerate(mcqs)],
            'correct': chr(65+correct_idx) if correct_idx >= 0 else ""
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
