import random
from sympy import (
    Eq, solve, sympify, solveset, S, latex,
    parse_expr, FiniteSet, nsolve, Symbol
)

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