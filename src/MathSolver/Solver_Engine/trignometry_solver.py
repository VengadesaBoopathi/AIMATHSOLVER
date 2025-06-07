import time
import sympy as sp
from sympy import (
    Eq, solveset, S, nsolve, Symbol, Union, ImageSet, Lambda, ConditionSet,
    symbols, Integer, sin, cos, tan, asin, acos, atan, pi
)

# Initialize integer symbols for general solutions
n, m = symbols('n m', integer=True)

def format_trig_solution(solution, variables):
    """Enhanced solution formatter with special handling for trigonometric solutions"""
    try:
        # Handle general trigonometric solutions with patterns
        if isinstance(solution, list) and all(isinstance(s, dict) for s in solution):
            formatted = []
            for sol in solution:
                parts = []
                for var, expr in sol.items():
                    # Add periodic terms with integer coefficients
                    if expr.has(sp.sin, sp.cos, sp.tan):
                        parts.append(f"{var} = {expr} + 2{pi}n")
                    else:
                        parts.append(f"{var} = {expr}")
                formatted.append(" AND ".join(parts))
            return "\nOR\n".join(formatted)
        
        # Handle ImageSet and ConditionSet for single-variable solutions 
        if isinstance(solution, ConditionSet):
            return f"Solution: {variables[0]} = {solution.lamda.expr} where {solution.lamda.variables[0]} ∈ ℤ"
        
        if isinstance(solution, ImageSet):
            base_sol = solution.lamda.expr
            param = solution.lamda.variables[0] 
            return f"{variables[0]} = {base_sol} where {param} ∈ ℤ"
        
        return str(solution)
    
    except Exception:
        return str(solution)

def solve_trig_symbolic(expr, variables):
    """Enhanced symbolic solver for trigonometric equations"""
    try:
        # Special handling for sin(a) = sin(b) type equations
        if expr.has(sp.sin):
            sol = sp.solve(expr, dict=True, force=True)
            if sol:
                return [{
                    k: v + 2*n*pi if not v.has(sp.I) else v + n*pi
                    for k, v in s.items()
                } for s in sol]
        
        # Special handling for cos(a) = cos(b)
        if expr.has(sp.cos):
            sol = sp.solve(expr, dict=True, force=True)
            if sol:
                return [{
                    k: v + 2*n*pi if not v.has(sp.I) else v + n*pi
                    for k, v in s.items()
                } for s in sol]
        
        # General case solving
        solution = solveset(expr, variables[0], domain=S.Reals)
        return solution
    
    except Exception:
        return None

def solve_trig_numeric(equations, user_guesses):
    """Robust numeric solver with better error handling"""
    start_time = time.time()
    try:
        all_vars = sorted(
            list(set().union(*[eq.free_symbols for eq in equations])),
            key=lambda x: str(x)
        )
        exprs = [(eq.lhs - eq.rhs) if isinstance(eq, Eq) else eq for eq in equations]
        initial_guess = [float(user_guesses.get(str(var), 0.0)) for var in all_vars]
        
        # Use higher precision and maximum iterations
        solution = nsolve(
            exprs, all_vars, initial_guess,
            prec=15,  # Increased precision
            verify=False,
            maxsteps=100
        )
        
        sol_dict = {
            str(var): float(solution[i]) 
            for i, var in enumerate(all_vars)
        }
        elapsed = (time.time() - start_time) * 1000
        return sol_dict, elapsed
    
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return None, elapsed

def trig_solve(parsed_eqs, user_guesses):
    """Main solver function with enhanced trigonometric handling"""
    try:
        # System of equations
        if len(parsed_eqs) > 1:
            start_time = time.time()
            try:
                sym_sol = sp.solve(parsed_eqs, dict=True, force=True)
                elapsed = (time.time() - start_time) * 1000
                return {
                    'method': 'symbolic',
                    'solution': format_trig_solution(sym_sol, list(sym_sol[0].keys())),
                    'time_ms': f"{elapsed:.2f} ms",
                    'error': None
                }
            except Exception:
                num_sol, elapsed = solve_trig_numeric(parsed_eqs, user_guesses)
                return {
                    'method': 'numeric' if num_sol else 'failed',
                    'solution': format_dict_solution(num_sol),
                    'time_ms': f"{elapsed:.2f} ms",
                    'error': None
                }

        # Single equation handling
        eq = parsed_eqs[0]
        expr = eq.lhs - eq.rhs if isinstance(eq, Eq) else eq
        variables = list(expr.free_symbols)
        
        # Try symbolic solution first
        start_time = time.time()
        sym_sol = solve_trig_symbolic(expr, variables)
        elapsed = (time.time() - start_time) * 1000
        
        if sym_sol:
            return {
                'method': 'symbolic',
                'solution': format_trig_solution(sym_sol, variables),
                'time_ms': f"{elapsed:.2f} ms",
                'error': None
            }
        
        # Fallback to numeric if symbolic fails
        num_sol, num_elapsed = solve_trig_numeric(parsed_eqs, user_guesses)
        return {
            'method': 'numeric' if num_sol else 'failed',
            'solution': format_dict_solution(num_sol),
            'time_ms': f"{num_elapsed:.2f} ms",
            'error': None
        }
    
    except Exception as e:
        return {
            'method': 'error',
            'solution': None,
            'error': str(e)
        }

def format_dict_solution(solution_dict):
    """Format numeric solutions with precision"""
    if solution_dict:
        return ", ".join(f"{k} = {v:.6f}" for k, v in solution_dict.items())
    return "No solution found"