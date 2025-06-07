import numpy as np
from flask import jsonify, request

class MatrixSolver:
    def solve(self, matrices, operation):
        try:
            np_matrices = [np.array(m['data']) for m in matrices]
            
            if operation == 'transpose':
                result = [m.T.tolist() for m in np_matrices]
            elif operation == 'inverse':
                result = [np.linalg.inv(m).tolist() for m in np_matrices]
            elif operation == 'addition':
                result = [ (np_matrices[0] + np_matrices[1]).tolist() ]
            elif operation == 'multiplication':
                result = [ np.matmul(np_matrices[0], np_matrices[1]).tolist() ]
                
            return {'result': result, 'error': None}
            
        except Exception as e:
            return {'result': None, 'error': str(e)}

def register_matrix_routes(app):
    @app.route('/matrix_solver', methods=['POST'])
    def handle_matrix_operation():
        data = request.json
        solver = MatrixSolver()
        result = solver.solve(data['matrices'], data['operation'])
        return jsonify(result)
    
    