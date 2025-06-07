import time
import sympy as sp
import numpy as np
from scipy.integrate import odeint
from sympy import (
    diff, dsolve, Function, Derivative, Eq, symbols,
    Symbol, solve, pde_separate, classify_ode, init_printing
)

import re

init_printing()

class Differentiator:
    def __init__(self):
        self.default_vars = {
            'x': symbols('x'),
            't': symbols('t'),
            'y': Function('y'),
            'z': Function('z')
        }
        self.numeric_precision = 6

    def solve(self, input_expr, variables=None, ics=None, method='auto'):
        """Main solver entry point"""
        start_time = time.time()
        result = {
            'success': False,
            'result': None,
            'method': None,
            'time': 0,
            'error': None
        }
        if variables is None:
            variables = list(input_expr.free_symbols)

        try:
            # Parse input
            parsed = self.parse_input(input_expr)
            
            # Determine problem type
            problem_type = self.classify_problem(parsed, variables)
            
            # Solve based on problem type
            if problem_type == 'direct_derivative':
                result.update(self.solve_direct_derivative(parsed, variables))
            elif problem_type == 'ode':
                result.update(self.solve_ode(parsed, ics, method))
            elif problem_type == 'pde':
                result.update(self.solve_pde(parsed, variables))
            elif problem_type == 'implicit':
                result.update(self.solve_implicit(parsed, variables))
            
            result['time'] = time.time() - start_time
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        result['solution'] = result['result']
        return result
    
    def parse_input(self, input_expr):
        """Parse various derivative notations"""
        substitutions = {
            r'\bd/d(\w+)\(([^)]+)\)': r'Derivative(\2, \1)',  # Capture entire expression
            r'\bDerivative\(([^)]+)\)\(([^)]+)\)': r'Derivative(\1, \2)',
            r'\bd/d(\w+)\b': r'Derivative(\1)',
            r'\b(d\^?\d?/d\w+\^?\d?)\b': lambda m: self._convert_deriv(m),
            r"(\w+)''": r'Derivative(\1, 2)',
            r"(\w+)'": r'Derivative(\1)',
            r'\b∂/∂(\w+)\b': r'Derivative(\1)'
        }

        if isinstance(input_expr, str):
            
            input_expr = re.sub(
                r'\bd/d(\w+)\(([^)]+)\)',
                r'Derivative(\2, \1)',
                input_expr
            )
            # Additional processing...
            for pattern, repl in substitutions.items():
                input_expr = re.sub(pattern, repl, input_expr)
            return sp.sympify(input_expr, locals=self.default_vars)
        return input_expr
 
    def _convert_deriv(self, match):
        """Convert Leibniz notation to SymPy format"""
        parts = match.group(1).split('/')
        var = parts[1][1:]  # Remove 'd' from denominator
        order = parts[0].count('d')
        return f'Derivative({{expr}}, {var}, {order})'  # Placeholder for expression

    def classify_problem(self, expr, variables):
        """Determine problem type"""
        if isinstance(expr, Derivative):
            if len(expr.variables) > 1:
                return 'pde'
            return 'direct_derivative'
        elif isinstance(expr, Eq) and expr.has(Derivative):
            if any(func in expr.rhs.free_symbols for func in [Function, Symbol]):
                return 'ode'
            return 'implicit'
        return 'unknown'

    def solve_direct_derivative(self, deriv, variables):
        """Compute explicit derivatives"""
        result = {}
        try:
            if isinstance(deriv, Derivative):
                expr = deriv.expr
                vars = deriv.variables
                order = len(vars)
                result['result'] = diff(expr, *vars)
                result['method'] = 'symbolic'
            else:
                result['result'] = diff(deriv, variables)
                result['method'] = 'symbolic'
        except Exception as e:
            result['error'] = str(e)
        return result

    def solve_ode(self, equation, ics=None, method='auto'):
        """Solve ordinary differential equations"""
        result = {}
        try:
            # Symbolic solution
            if method in ['auto', 'symbolic']:
                sol = dsolve(equation, ics=ics)
                result['result'] = sol
                result['method'] = 'symbolic'
                result['classification'] = classify_ode(equation)
                return result
        except NotImplementedError:
            pass

        # Numerical solution fallback
        try:
            numeric_sol = self._solve_ode_numeric(equation, ics)
            result['result'] = numeric_sol
            result['method'] = 'numeric'
            return result
        except Exception as e:
            result['error'] = f"Both methods failed: {str(e)}"
            return result

    def _solve_ode_numeric(self, equation, ics):
        """Numerical ODE solver"""
        t = np.linspace(0, 10, 100)
        y0 = [float(v) for v in ics.values()]
        
        # Convert to numeric functions
        lhs = equation.lhs
        rhs = equation.rhs
        deriv = solve(equation, lhs)[0]
        func = sp.lambdify((lhs.args[0], lhs.variables[0]), deriv, 'numpy')
        
        solution = odeint(func, y0, t)
        return {
            't': t.round(self.numeric_precision).tolist(),
            'y': solution.round(self.numeric_precision).tolist()
        }

    def solve_pde(self, equation, variables):
        """Solve partial differential equations (limited support)"""
        result = {}
        try:
            sep_vars = pde_separate(equation, 
                                  equation.args[0],
                                  variables)
            result['result'] = sep_vars
            result['method'] = 'separation_of_variables'
            return result
        except ValueError:
            result['error'] = "PDE solution not supported for this equation"
            return result

    def solve_implicit(self, equation, variables):
        """Solve implicit differentiation problems"""
        result = {}
        try:
            y_var = next(v for v in variables if isinstance(v, Function))
            x_var = next(v for v in variables if v != y_var)
            
            derivative = solve(equation, Derivative(y_var(x_var), x_var))[0]
            result['result'] = derivative
            result['method'] = 'implicit_differentiation'
            return result
        except Exception as e:
            result['error'] = f"Implicit differentiation failed: {str(e)}"
            return result

    def pretty_output(self, result):
        """Format results for human reading"""
        if not result['success']:
            return f"Error: {result['error']}"
        
        if result['method'] == 'symbolic':
            return sp.pretty(result['result'])
        
        if result['method'] == 'numeric':
            return ("Numerical solution:\n" 
                    f"Time points: {len(result['result']['t'])}\n"
                    f"Final value: {result['result']['y'][-1][0]:.4f}")
        
        return str(result['result'])
