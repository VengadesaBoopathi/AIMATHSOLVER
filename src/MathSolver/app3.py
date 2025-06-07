# .\venv\Scripts\Activate
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import time
from sympy import (
    Eq, solve, sympify, solveset, S, latex,
    parse_expr, FiniteSet, nsolve, Symbol
)
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication
import re
import random
import os
import sys
import numpy as np
from scipy.optimize import fsolve
import sympy as sp  
import joblib
import pandas as pd 
import io 
from flask_cors import CORS

# Azure Computer Vision imports
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)
CORS(app)


if os.path.exists('solver_model.pkl'):
    model = joblib.load('solver_model.pkl')
else:
    model = None   


def parse_equation(equations):
    transforms = standard_transformations + (implicit_multiplication,)
    parsed = []
    for eq in equations:
        # Remove non-math characters (strict)
        eq_clean = re.sub(r"[^0-9a-z=+\-*/^().]", "", eq.lower())
        if not eq_clean:
            raise ValueError(f"Invalid equation: {eq}")
        
        # Log cleaned equation for debugging
        print(f"[DEBUG] Cleaned equation: {eq_clean}")
        
        if '=' in eq_clean:
            lhs, rhs = eq_clean.split('=', 1)
            parsed.append(Eq(
                parse_expr(lhs, transformations=transforms),
                parse_expr(rhs, transformations=transforms)
            ))
        else:
            parsed.append(parse_expr(eq_clean, transformations=transforms))
    return parsed

# Azure Configuration
AZURE_KEY = "G3LQgxG0tx3eL0aCx7RBPQPU6alEjh6EIX3zAyxr8KDMepbz9OrgJQQJ99BDACYeBjFXJ3w3AAAFACOGwBU4"
AZURE_ENDPOINT = "https://venkiboo.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

# Update OCR substitutions for plain text output
OCR_SUBSTITUTIONS = {
    r'\s*([/])\s*': r'\1',      # Remove spaces around /
    r'×': '*',                  # Multiplication symbol to *
    r'÷': '/',                  # Division symbol to /
    r'°': '*pi/180',            # Degrees to radians
    r'\^': '**',                # Handle exponents
    r'\bmod\b': '%',            # mod to percentage
    r'\s+': ' ',                # Remove extra spaces
    r'\\left': '',              # Remove LaTeX formatting artifacts
    r'\\right': '',
    r'\\cdot': '*'
}
def azure_ocr(image):
    """Process image using Azure Computer Vision OCR"""
    try:
        # Convert PIL Image to BytesIO stream
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Reset buffer position to start

        # Call Azure OCR with stream directly
        read_response = computervision_client.read_in_stream(
            image=img_byte_arr,  # Pass the BytesIO object directly
            raw=True
        )

        # Rest of your existing code remains the same...
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(0.5)

        if read_result.status == OperationStatusCodes.succeeded:
            text_lines = []
            for page in read_result.analyze_result.read_results:
                for line in page.lines:
                    text_lines.append(line.text)
            return '\n'.join(text_lines)
            
        return ""
    
    except Exception as e:
        raise RuntimeError(f"Azure OCR failed: {str(e)}")
    
def preprocess_ocr_text(text):
    # Normalize text first
    text = text.replace('\\\n', '')  # Remove LaTeX line breaks
    text = re.sub(r'\s*=\s*', '=', text)  # Clean spaces around =
    
    # Apply substitutions
    for pattern, replacement in OCR_SUBSTITUTIONS.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle fractions in different formats
    text = re.sub(r'(\d+)/(\d+)', r'(\1)/(\2)', text)  # Protect fractions
    return text.strip()

def image_to_equation(image):
    try:
        raw_text = azure_ocr(image)
        cleaned_text = preprocess_ocr_text(raw_text)
        
        # Split equations on multiple lines or semicolons
        equations = []
        for line in [ln.strip() for ln in cleaned_text.split('\n') if ln.strip()]:
            equations.extend([eq.strip() for eq in re.split(r'[;,]', line) if eq.strip()])
            
        return equations
    
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

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
            distractors = [dict((var, round(val + random.choice([-2, -1.5, 1, 2]) + random.uniform(-0.5, 0.5), 2))
                           for var, val in correct.items())
                           for _ in range(3)]
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

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream)
    
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
                mcqs, correct_idx = generate_mcqs(symbolic_sol)
                result = {
                    'method': 'symbolic',
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
                    mcqs, correct_idx = generate_mcqs(numeric_sol if numeric_sol else {})
                    result = {
                        'method': 'numeric',
                        'solution': format_dict_solution(numeric_sol),
                        'time_ms': f"{numeric_time:.2f} ms",
                        'error': f"Symbolic solver failed: {str(fallback_error)}. Numeric fallback executed."
                                 f" {numeric_error or ''}".strip()
                    }
                except Exception as e2:
                    result = {
                        'method': 'numeric',
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
                mcqs, correct_idx = generate_mcqs(numeric_sol if numeric_sol else {})
                result = {
                    'method': 'numeric',
                    'solution': format_dict_solution(numeric_sol),
                    'time_ms': f"{numeric_time:.2f} ms",
                    'error': numeric_error
                }
            except Exception as e:
                result = {
                    'method': 'numeric',
                    'solution': None,
                    'time_ms': None,
                    'error': f"Numeric solver failed: {str(e)}"
                }

        return jsonify({
            'equations': parsed_latex,
            'recommended_method': recommended_method,
            'result': result,
            'mcqs': [f"{chr(65+i)}: {opt}" for i, opt in enumerate(mcqs)],
            'correct': chr(65+correct_idx) if correct_idx >= 0 else ""
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def train_ml_model(training_data_file='training_data.csv'):
    
    data = pd.read_csv(r"C:\Users\vengi\Desktop\AIMATHSOLVER\my-react-app\src\data.csv")

    X = data[['Degree', 'Num_Terms', 'Num_Variables']]
    y = data['Best_Method']
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X, y)    
    joblib.dump(clf, 'solver_model.pkl')
    print("Training complete. Model saved as solver_model.pkl.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        train_ml_model()
    else:
        app.run(debug=True, port=5001)
