import io 
import re
import time
from PIL import Image
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication
from sympy import (
    Derivative, Eq, solve, symbols, sympify, solveset, S, latex,
    parse_expr, FiniteSet, nsolve, Symbol
)
transforms = standard_transformations + (implicit_multiplication,)

# Azure Computer Vision imports
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials 

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
    # r'\s+': ' ',                # Remove extra spaces
    r'\\left': '',              # Remove LaTeX formatting artifacts
    r'\\right': '',
    r'\\cdot': '*',
    r'\b(sin|cos|tan|arcsin|arccos|arctan)\s*([a-zA-Z0-9]+)\b': r'\1(\2)',  # Add parentheses to trig functions
    r'\b(sin|cos|tan|arcsin|arccos|arctan)\s*\(': r'\1(',  # Ensure space before (
    r'°\s*\)': '*pi/180)',
    r"\b(d/d)([a-zA-Z])": r"Derivative(\2)",       # d/dx → Derivative(x)
    r"\b(d\^2/d[a-zA-Z]\^2)\b": r"Derivative(\1, 2)",  # d²/dx²
    r"([a-zA-Z])''": r"Derivative(\1, 2)",        # y'' → Derivative(y, 2)
    r"([a-zA-Z])'": r"Derivative(\1)",            # y' → Derivative(y)
    r"\b(∂/∂([a-zA-Z]))": r"Derivative(\2)"        # Partial derivatives
}

def azure_ocr(image):
    """Process image using Azure Computer Vision OCR"""
    try:
         
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)   
        
        read_response = computervision_client.read_in_stream(
            image=img_byte_arr,  
            raw=True
        )

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
    text = re.sub(r'(\d+)°', r'(\1*pi/180)', text)
    
    # Apply substitutions
    for pattern, replacement in OCR_SUBSTITUTIONS.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle fractions in different formats
    text = re.sub(r'(\d+)/(\d+)', r'(\1)/(\2)', text)  # Protect fractions
    return text.strip()

def parse_equations(equations):
    parsed = []
    for raw in equations:
        # 1) strip out junk but keep letters, digits, =, ±, (), commas, primes
        eq = re.sub(r"[^0-9A-Za-z=+\-*/^().,']", "", raw)
        print(f"[DEBUG] Cleaned equation: {eq}")

        # 2) **force** the call‐pattern fix
        fixed = _fix_derivative_call_str(eq)
        if fixed != eq:
            print(f"[DEBUG] Applied call‐fix → {fixed}")
            eq = fixed

        # 3) now safely parse with our transformations
        if "=" in eq:
            lhs, rhs = eq.split("=", 1)
            left  = parse_expr(lhs, transformations=transforms)
            right = parse_expr(rhs, transformations=transforms)
            node = Eq(left, right)
        else:
            node = parse_expr(eq, transformations=transforms)

        parsed.append(node)

    return parsed

def _fix_derivative_call_str(s: str) -> str:
    
    if "Derivative" not in s or ")(" not in s:
        return s

    # find the first "Derivative("
    start = s.find("Derivative(")
    # now find the matching ')' for the var
    level = 0
    i = start + len("Derivative(")
    for i in range(i, len(s)):
        if s[i] == "(":
            level += 1
        elif s[i] == ")":
            if level == 0:
                end_var = i
                break
            level -= 1
    else:
        return s

    var = s[start + len("Derivative("):end_var].strip()

    # find the '(' after that for the call
    call_open = s.find("(", end_var + 1)
    if call_open < 0:
        return s

    # find the matching ')' for the body
    level = 1
    for j in range(call_open + 1, len(s)):
        if s[j] == "(":
            level += 1
        elif s[j] == ")":
            level -= 1
            if level == 0:
                end_body = j
                break
    else:
        return s

    body = s[call_open + 1:end_body].strip()
    # rebuild the string
    return s[:start] + f"Derivative({body}, {var})" + s[end_body+1:]



def image_to_equations(image):
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
    