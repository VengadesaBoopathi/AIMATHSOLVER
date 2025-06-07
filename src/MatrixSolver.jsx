import React, { useState, useRef } from 'react';
import { Button, Form, Card, Row, Col, Alert } from 'react-bootstrap';
import axios from 'axios';

const MatrixSolver = () => {
    const [matrices, setMatrices] = useState([]); // now array of matrices
    const [newMatrix, setNewMatrix] = useState({
        name: '',
        rows: 2,
        cols: 2
    });
    const [operation, setOperation] = useState('transpose');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const fileInputRef = useRef();

    const handleImageUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('image', file);
        formData.append('rows', newMatrix.rows);
        formData.append('cols', newMatrix.cols);

        try {
            const { data } = await axios.post(
                'http://localhost:5001/upload_matrix',
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            if (!data.data || data.data.flat().length !== newMatrix.rows * newMatrix.cols) {
                throw new Error('Matrix dimensions mismatch with extracted data');
            }

            const mapped = data.data.map(row =>
                row.map(cell =>
                    Number.isInteger(cell) ? cell.toString() : cell.toFixed(2)
                )
            );

            setMatrices(prev => [
                ...prev,
                {
                    name: newMatrix.name || `Matrix ${prev.length + 1}`,
                    rows: newMatrix.rows,
                    cols: newMatrix.cols,
                    data: mapped
                }
            ]);

            setNewMatrix({ name: '', rows: 2, cols: 2 });  // RESET after upload
            setError('');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || err.message || 'Failed to process matrix image');
        }
    };


    const handleMatrixOperation = async () => {
        if (matrices.length === 0) {
            setError('No matrices available. Upload at least one matrix.');
            return;
        }

        const allFilled = matrices.every(m =>
            m.data.length === m.rows &&
            m.data.every(row => row.length === m.cols && row.every(cell => cell !== ''))
        );

        if (!allFilled) {
            setError('All matrix cells must be filled before solving.');
            return;
        }

        try {
            const payload = matrices.map(m => ({
                name: m.name,
                rows: m.rows,
                cols: m.cols,
                data: m.data.map(row => row.map(Number))
            }));

            const resp = await axios.post('http://localhost:5001/matrix_solver', {
                matrices: payload,
                operation
            });

            setResult(resp.data.result);
            setError('');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || 'Matrix operation failed');
        }
    };

    const handleClear = () => {
        setMatrices([]);
        setNewMatrix({ name: '', rows: 2, cols: 2 });
        setResult(null);
        setError('');
    };

    return (
        <Card className="shadow-sm mb-4" style={{ margin: 'auto', borderRadius: '15px', width: '100%' }}>
            <Card.Body>
                <Form>
                    {/* Add Matrix Section */}
                    <h5>Add New Matrix</h5>
                    <Row className="mb-3">
                        <Col md={3}>
                            <Form.Group>
                                <Form.Label>Matrix Name</Form.Label>
                                <Form.Control
                                    type="text"
                                    value={newMatrix.name}
                                    onChange={e => setNewMatrix(prev => ({ ...prev, name: e.target.value }))}
                                    placeholder="e.g. A"
                                />
                            </Form.Group>
                        </Col>
                        <Col md={3}>
                            <Form.Group>
                                <Form.Label>Rows</Form.Label>
                                <Form.Control
                                    type="number"
                                    min={1}
                                    value={newMatrix.rows}
                                    onChange={e => setNewMatrix(prev => ({ ...prev, rows: +e.target.value }))}
                                />
                            </Form.Group>
                        </Col>
                        <Col md={3}>
                            <Form.Group>
                                <Form.Label>Columns</Form.Label>
                                <Form.Control
                                    type="number"
                                    min={1}
                                    value={newMatrix.cols}
                                    onChange={e => setNewMatrix(prev => ({ ...prev, cols: +e.target.value }))}
                                />
                            </Form.Group>
                        </Col>
                        <Col md={3} className="d-flex align-items-end">
                            <Button variant="primary" onClick={() => fileInputRef.current.click()}>
                                Upload Matrix Image
                            </Button>
                            <Form.Control
                                type="file"
                                accept="image/*"
                                hidden
                                ref={fileInputRef}
                                onChange={handleImageUpload}
                            />
                        </Col>
                    </Row>

                    {error && <Alert variant="danger">{error}</Alert>}

                    {/* Matrices Display Section */}
                    {matrices.length > 0 && (
                        <>
                            <h5 className="mt-4">Uploaded Matrices</h5>
                            {matrices.map((matrix, idx) => (
                                <Card key={idx} className="mb-3">
                                    <Card.Body>
                                        <h6>{matrix.name}</h6>
                                        {matrix.data.map((row, i) => (
                                            <Row key={i} className="mb-2">
                                                {row.map((cell, j) => (
                                                    <Col key={j} xs="auto">
                                                        <Form.Control
                                                            type="number"
                                                            step="any"
                                                            value={cell}
                                                            style={{ width: '80px' }}
                                                            onChange={e => {
                                                                const updated = [...matrices];
                                                                updated[idx].data[i][j] = e.target.value;
                                                                setMatrices(updated);
                                                            }}
                                                        />
                                                    </Col>
                                                ))}
                                            </Row>
                                        ))}
                                    </Card.Body>
                                </Card>
                            ))}
                        </>
                    )}

                    {/* Operation Section */}
                    {matrices.length > 0 && (
                        <>
                            <Form.Group className="mb-3" controlId="operationSelect">
                                <Form.Label>Select Operation</Form.Label>
                                <Form.Control
                                    as="select"
                                    value={operation}
                                    onChange={e => setOperation(e.target.value)}
                                >
                                    <option value="transpose">Transpose</option>
                                    <option value="inverse">Inverse</option>
                                    <option value="addition">Addition</option>
                                    <option value="multiplication">Multiplication</option>
                                </Form.Control>
                            </Form.Group>

                            <div className="d-flex gap-2 mb-3">
                                <Button variant="success" onClick={handleMatrixOperation}>
                                    Perform Operation
                                </Button>
                                <Button variant="outline-secondary" onClick={handleClear}>
                                    Clear All
                                </Button>
                            </div>
                        </>
                    )}

                    {/* Result Section */}
                    {result && (
                        <div className="mt-4">
                            <h5>Result:</h5>
                            {Array.isArray(result) ? (
                                result.map((mat, idx) => (
                                    <div key={idx} className="mb-3">
                                        <h6>Result Matrix {idx + 1}</h6>
                                        {mat.map((r, i) => (
                                            <Row key={i} className="mb-1">
                                                {r.map((c, j) => (
                                                    <Col xs="auto" key={j}>
                                                        <Form.Control
                                                            readOnly
                                                            value={typeof c === 'number' ? c.toFixed(3) : c}
                                                            style={{ width: '80px', textAlign: 'center' }}
                                                        />
                                                    </Col>
                                                ))}
                                            </Row>
                                        ))}
                                    </div>
                                ))
                            ) : (
                                <Alert variant="info">No result matrix returned.</Alert>
                            )}
                        </div>
                    )}
                </Form>
            </Card.Body>
        </Card>
    );
};

export default MatrixSolver;
