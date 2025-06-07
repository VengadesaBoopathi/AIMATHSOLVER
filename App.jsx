import React, { useState } from 'react';
import { Container, Card, Button, Form, Alert, Spinner, Row, Col, Image } from 'react-bootstrap';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [results, setResults] = useState(null);
  const [initialValues, setInitialValues] = useState({});
  const [variables, setVariables] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Handle file upload
  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreviewURL(URL.createObjectURL(uploadedFile));
      setResults(null);
      setInitialValues({});
      setVariables([]);
    }
  };

  // Handle initial guess changes
  const handleInitialValueChange = (varName, value) => {
    setInitialValues((prev) => ({
      ...prev,
      [varName]: value,
    }));
  };

  // Handle form submit
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload an image.');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    const formData = new FormData();
    formData.append('image', file);

    // Build initial guess string using template literals.
    const initialGuessString = Object.entries(initialValues)
      .map(([varName, val]) => `${varName}=${val}`)
      .join(', ');
    if (initialGuessString) {
      formData.append('initial_guesses', initialGuessString);
    }

    try {
      const response = await axios.post('http://localhost:5001/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const resData = response.data;
      setResults(resData);

      // Extract variable names from parsed equations (simple heuristic)
      const varRegex = /[a-zA-Z]+/g;
      const vars = Array.from(
        new Set(
          resData.equations
            .join(' ')
            .match(varRegex)
            .filter((v) => v.length === 1)
        )
      );
      setVariables(vars);
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="container mt-5">
      <h1 className="text-center mb-4">
        Math Equation Solver with<br /> MCQ Generator
      </h1>

      <Card className="shadow-lg mb-4">
        <Card.Body>
          <Form onSubmit={handleSubmit}>
            <Form.Group controlId="formFile" className="mb-3">
              <Form.Label>Upload Equation Image</Form.Label>
              <Form.Control type="file" accept="image/*" onChange={handleFileChange} />
            </Form.Group>

            {previewURL && (
              <div className="mb-3 text-center">
                <Image src={previewURL} alt="Preview" thumbnail fluid style={{ maxHeight: '200px' }} />
              </div>
            )}

            {variables.length > 0 && (
              <div className="mb-3">
                <h5>Provide Initial Guesses (optional)</h5>
                <Row>
                  {variables.map((varName, idx) => (
                    <Col md={3} key={idx}>
                      <Form.Group className="mb-2">
                        <Form.Label>{varName}</Form.Label>
                        <Form.Control
                          type="number"
                          placeholder={`Enter ${varName}`}
                          value={initialValues[varName] || ''}
                          onChange={(e) => handleInitialValueChange(varName, e.target.value)}
                        />
                      </Form.Group>
                    </Col>
                  ))}
                </Row>
              </div>
            )}

            <Button variant="primary" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <Spinner animation="border" size="sm" className="me-2" />
                  Solving...
                </>
              ) : (
                'Solve Equation'
              )}
            </Button>
          </Form>
        </Card.Body>
      </Card>

      {error && <Alert variant="danger">{error}</Alert>}

      {results && (
        <div>
          <Card className="shadow-sm mb-3">
            <Card.Body>
              <Card.Title>Parsed Equations</Card.Title>
              <pre>{results.equations.join('\n')}</pre>
            </Card.Body>
          </Card>

          <Card className="shadow-sm mb-3">
            <Card.Body>
              <Card.Title>Recommended Method</Card.Title>
              <p><strong>{results.recommended_method.toUpperCase()}</strong></p>
              <Card.Title>Result</Card.Title>
              <pre>{results.result.solution}</pre>
              <p>
                <strong>Time Taken:</strong> {results.result.time_ms}
              </p>
              {results.result.error && (
                <p className="text-danger">Error: {results.result.error}</p>
              )}
            </Card.Body>
          </Card>

          {results.mcqs && results.mcqs.length > 0 && (
            <Card className="shadow-sm mb-3">
              <Card.Body>
                <Card.Title>MCQ Challenge</Card.Title>
                <ul>
                  {results.mcqs.map((opt, idx) => (
                    <li key={idx}>{opt}</li>
                  ))}
                </ul>
                <p><strong>Correct Answer:</strong> {results.correct}</p>
              </Card.Body>
            </Card>
          )}
        </div>
      )}
    </Container>
  );
}

export default App;
