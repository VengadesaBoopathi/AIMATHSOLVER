import logging
import time
import traceback
import io
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import sympy as sp
from ocr_extraction import parse_equations, image_to_equations
from Solver_Engine.algebra_solver_engine import algebra_solve
from Solver_Engine.trignometry_solver import trig_solve
from Solver_Engine.differentiation_solver import Differentiator
from Solver_Engine.matrix_solver import register_matrix_routes
from mcq_generator import generate_mcqs
from sympy import Basic, Eq, Symbol, Derivative

# Azure Computer Vision imports
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials 

# Azure Configuration
AZURE_KEY = "G3LQgxG0tx3eL0aCx7RBPQPU6alEjh6EIX3zAyxr8KDMepbz9OrgJQQJ99BDACYeBjFXJ3w3AAAFACOGwBU4"
AZURE_ENDPOINT = "https://venkiboo.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
# register_matrix_routes(app)    # <<–– make sure to call this!

def detect_equation_type(parsed_eqs, raw_eq_strings=None):
    # 1) Check for any derivative hints first (SymPy or raw text)
    if is_differentiation_equation(parsed_eqs, raw_eq_strings):
        return 'calculus'
    # 2) Only if no derivative do we check for trig
    if is_trig_equation(parsed_eqs):
        return 'trigonometry'
    return 'algebra'


@app.route('/upload', methods=['POST'])


def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({'error': 'Uploaded file is not a valid image.'}), 400

    # parse initial guesses, selected_solver…
    initial_guesses = {}
    init_str = request.form.get('initial_guesses', '').strip()
    selected_solver = request.form.get('selected_solver', 'all')
    if init_str:
        try:
            for pair in init_str.split(','):
                var, val = pair.split('=')
                initial_guesses[var.strip()] = float(val.strip())
        except Exception:
            return jsonify({
                'error': "Invalid format for initial_guesses. Use comma-separated var=value pairs."
            }), 400

    try:
        # 1) OCR → raw strings
        eq_strings = image_to_equations(image)
        logging.info(f"Extracted equations: {eq_strings}")

        # 2) parse into SymPy objects
        parsed_eqs = parse_equations(eq_strings)

        # 3) detect type
        detected_type = detect_equation_type(parsed_eqs)
        solver_message = ""
        error_message = ""

        # 4) ensure solver choice matches detection
        if selected_solver != 'all' and selected_solver != detected_type:
            error_message = (
                f"Selected {selected_solver} solver but detected {detected_type} equation. "
                f"Using {detected_type} solver instead."
            )

        # 5) dispatch to the right engine
        if detected_type == 'calculus':
            diff_solve = Differentiator()
            solve_result = diff_solve.solve(
                input_expr=parsed_eqs[0],
                ics=initial_guesses,
                variables=list(parsed_eqs[0].free_symbols)
            )
            solver_message = "Differential equation solved using calculus solver"
        elif detected_type == 'trigonometry':
            solve_result = trig_solve(parsed_eqs, initial_guesses)
            solver_message = "Trigonometric equation solved using trigonometric solver"
        
        else:
            solve_result = algebra_solve(parsed_eqs, initial_guesses)
            solver_message = "Algebraic equation solved using algebraic solver"

        # 6) extract and stringify the SymPy result
        raw_sol = solve_result.get('solution', solve_result.get('result'))
        if isinstance(raw_sol, Basic):
            raw_sol = str(raw_sol)

        # 7) build the payload
        result = {
            'method':        solve_result.get('method'),
            'solution':      raw_sol,
            'time_ms':       solve_result.get('time') or solve_result.get('time_ms'),
            'error':         solve_result.get('error'),
            'message':       solver_message,
            'detected_type': detected_type
        }
        if error_message:
            result['error'] = error_message

        # 8) generate MCQs
        mcq_options, correct_idx = generate_mcqs(solve_result.get('_raw_solution'))
        mcqs = [f"{chr(65+i)}: {opt}" for i, opt in enumerate(mcq_options)]
        correct = chr(65 + correct_idx) if correct_idx is not None and correct_idx >= 0 else None

        # 9) return JSON
        return jsonify({
            'equations':          eq_strings,
            'recommended_method': detected_type,
            'result':             result,
            'mcqs':               mcqs,
            'correct':            correct
        })

    except Exception as e:
        logging.error("Unhandled error in /upload:", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/upload_matrix', methods=['POST'])
def handle_matrix_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream)
        rows = int(request.form.get('rows', 2))
        cols = int(request.form.get('cols', 2))
        matrix_data = process_matrix_image(img, rows, cols)
        return jsonify(matrix_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_matrix_image(image: Image.Image, target_rows: int, target_cols: int):
    """
    OCR → numeric tokens with centers → sort & chunk →
    returns {'rows':…, 'cols':…, 'data': [[…],[…],…]}
    """
    # 1) Run Azure OCR & get word-level boxes
    raw = azure_ocr_with_coordinates(image)
    # raw: [{'text': '6', 'bounding_box': [x1,y1,x2,y2,x3,y3,x4,y4]}, …]

    # 2) Extract only tokens that are purely numbers
    numbered = []
    for w in raw:
        txt = w['text'].strip()
        if re.fullmatch(r'[-+]?\d*\.?\d+', txt):
            # compute center of the 4 points
            xs = w['bounding_box'][0::2]
            ys = w['bounding_box'][1::2]
            x_c = sum(xs) / 4.0
            y_c = sum(ys) / 4.0
            numbered.append({
                'value': float(txt),
                'x_center': x_c,
                'y_center': y_c
            })

    expected = target_rows * target_cols
    if len(numbered) != expected:
        raise ValueError(f"Found {len(numbered)} numbers, but expected {expected}")

    # 3) Sort all tokens by Y (top→bottom)
    numbered.sort(key=lambda w: w['y_center'])

    # 4) Chunk into rows of length `target_cols`
    matrix = []
    for r in range(target_rows):
        row_slice = numbered[r*target_cols:(r+1)*target_cols]
        # 5) Within each row, sort left→right
        row_slice.sort(key=lambda w: w['x_center'])
        matrix.append([w['value'] for w in row_slice])

    return {
        'rows': target_rows,
        'cols': target_cols,
        'data': matrix
    }


def azure_ocr_with_coordinates(image: Image.Image):
    """Call Azure Read API, return list of {'text','bounding_box'}."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    resp = computervision_client.read_in_stream(image=img_bytes, raw=True)
    op_loc = resp.headers["Operation-Location"]
    op_id = op_loc.split("/")[-1]

    while True:
        status = computervision_client.get_read_result(op_id)
        if status.status not in (OperationStatusCodes.running, OperationStatusCodes.not_started):
            break
        time.sleep(0.5)

    words = []
    if status.status == OperationStatusCodes.succeeded:
        for page in status.analyze_result.read_results:
            for line in page.lines:
                for w in line.words:
                    words.append({
                        'text': w.text,
                        'bounding_box': w.bounding_box   # list of 8 coords
                    })
    return words

def is_trig_equation(parsed_eqs):
    trig_functions = (sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan)
    for eq in parsed_eqs:
        expr = eq.lhs - eq.rhs if isinstance(eq, Eq) else eq
        if any(expr.has(func) for func in trig_functions):
            return True
    return False

def is_differentiation_equation(parsed_eqs, raw_eq_strings=None):
   
    # 1) SymPy-level detection
    for eq in parsed_eqs:
        expr = eq.lhs - eq.rhs if isinstance(eq, Eq) else eq
        if expr.has(Derivative):
            return True

    # 2) Raw-text fallback
    if raw_eq_strings:
        for line in raw_eq_strings:
            # match d/dx, d/dy, etc.
            if re.search(r'\bd/d[A-Za-z]\b', line):
                return True
            # match prime notation y' or y’
            if re.search(r"[A-Za-z][\'\u2019]\b", line):
                return True

    return False

if __name__ == '__main__':
    register_matrix_routes(app)
    app.run(debug=True, port=5001)