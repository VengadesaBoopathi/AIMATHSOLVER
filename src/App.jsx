import React, { useState } from 'react';
import {
  Container,
  Card,
  Button,
  Form,
  Alert,
  Spinner,
  Row,
  Col,
  Image,
  ListGroup
} from 'react-bootstrap';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import MatrixSolver from './MatrixSolver';

function App() {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [results, setResults] = useState(null);
  const [initialValues, setInitialValues] = useState({});
  const [variables, setVariables] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTopic, setActiveTopic] = useState('all');
  const [detectedType, setDetectedType] = useState('');

  // Reset everything except activeTopic
  const resetAll = () => {
    setFile(null);
    setPreviewURL(null);
    setResults(null);
    setInitialValues({});
    setVariables([]);
    setDetectedType('');
    setError('');
    setLoading(false);
  };

  const topics = [
    { name: 'Algebra', value: 'algebra' },
    { name: 'Trigonometry', value: 'trigonometry' },
    { name: 'Calculus', value: 'calculus' },
    { name: 'Matrix', value: 'matrix' }
  ];

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreviewURL(URL.createObjectURL(uploadedFile));
      setResults(null);
      setInitialValues({});
      setVariables([]);
      setDetectedType('');
      setError('');
    }
  };

  const handleInitialValueChange = (varName, value) => {
    setInitialValues((prev) => ({
      ...prev,
      [varName]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload an image.');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);
    setDetectedType('');

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('selected_solver', activeTopic);

      const initialGuessString = Object.entries(initialValues)
        .map(([varName, val]) => `${varName}=${val}`)
        .join(', ');
      if (initialGuessString) {
        formData.append('initial_guesses', initialGuessString);
      }

      const response = await axios.post('http://localhost:5001/upload', formData);
      const resData = response.data;

      if (resData.detected_type) {
        setDetectedType(resData.detected_type);
      }
      if (resData.error) {
        setError(resData.error);
      }
      setResults(resData);

      // extract single-letter vars from the OCRed equations
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

  const getSolverTitle = () => {
    if (detectedType) {
      const detected = topics.find(t => t.value === detectedType);
      return `${detected.name} Solver (Auto-detected)`;
    }
    const active = topics.find(t => t.value === activeTopic);
    return activeTopic === 'all'
      ? 'Math Problem Solver with MCQ Generator'
      : `${active.name} Solver`;
  };

  return (
    <Container fluid className="d-flex p-0" style={{ minHeight: '100vh' }}>
      {/* Left Sidebar */}
      <div style={{ backgroundColor: 'white', width: '280px' }}>
        <div className="p-3">
          <h4 className="mb-4">Math Problem Solver</h4>
          <ListGroup variant="flush">
            {topics.map(topic => (
              <ListGroup.Item
                key={topic.value}
                action
                active={activeTopic === topic.value}
                onClick={() => {
                  setActiveTopic(topic.value);
                  resetAll();
                }}
                style={{ cursor: 'pointer' }}
              >
                {topic.name}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-grow-1 p-4" style={{ backgroundColor: '#f8f9fa' }}>
        <h2 className="mb-4">{getSolverTitle()}</h2>

        {activeTopic === 'matrix' ? (
          <MatrixSolver />
        ) : (
          <>
            {/* Upload & Preview Card */}
            <Card className="mb-4" style={{ maxWidth: '800px', margin: 'auto' }}>
              <Card.Body>
                <Form onSubmit={handleSubmit}>
                  <Form.Group className="mb-4 text-center">
                    <Button
                      variant="outline-primary"
                      onClick={() => document.getElementById('file-upload').click()}
                      style={{ width: '100%', padding: '2rem', fontSize: '1.5rem' }}
                    >
                      📷 Upload Image
                    </Button>
                    <Form.Control
                      type="file"
                      id="file-upload"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="d-none"
                    />
                  </Form.Group>

                  {previewURL && (
                    <div className="mb-4 text-center">
                      <Image
                        src={previewURL}
                        alt="Preview"
                        fluid
                        style={{ maxHeight: '400px', margin: 'auto' }}
                      />
                    </div>
                  )}

                  {variables.length > 0 && (
                    <div className="mb-4">
                      <h5>Initial Values</h5>
                      <Row>
                        {variables.map((varName, idx) => (
                          <Col md={4} key={idx}>
                            <Form.Group className="mb-3">
                              <Form.Label>{varName}</Form.Label>
                              <Form.Control
                                type="number"
                                value={initialValues[varName] || ''}
                                onChange={e => handleInitialValueChange(varName, e.target.value)}
                              />
                            </Form.Group>
                          </Col>
                        ))}
                      </Row>
                    </div>
                  )}

                  <div className="d-flex gap-2 mb-3">
                    <Button variant="primary" type="submit" disabled={loading}>
                      {loading ? (
                        <>
                          <Spinner animation="border" size="sm" className="me-2" /> Solving…
                        </>
                      ) : (
                        'Solve Now'
                      )}
                    </Button>
                    <Button variant="secondary" onClick={resetAll}>
                      Clear
                    </Button>
                  </div>

                  {error && <Alert variant="danger">{error}</Alert>}
                </Form>
              </Card.Body>
            </Card>

            {/* Result Card */}
            {results && (
              <Card style={{ maxWidth: '800px', margin: 'auto' }}>
                <Card.Body>
                  <h4 className="mb-3">Solution</h4>

                  <h6>Detected Equation Type</h6>
                  <Alert variant="info">
                    {results.result.message}
                    {detectedType && ` (Auto-switched to ${detectedType})`}
                  </Alert>

                  <h6>Method Used</h6>
                  <div className="badge bg-primary mb-3">
                    {results.recommended_method.toUpperCase()} SOLVER
                  </div>

                  <h6>Result</h6>
                  <pre className="bg-light p-3 rounded">
                    {results.result.solution}
                  </pre>
                  <p className="text-muted">
                    Solved in {results.result.time_ms} ms
                  </p>

                  {results.mcqs?.length > 0 && (
                    <>
                      <h6>Practice Questions</h6>
                      <ListGroup className="mb-2">
                        {results.mcqs.map((opt, i) => (
                          <ListGroup.Item key={i}>{opt}</ListGroup.Item>
                        ))}
                      </ListGroup>
                      <small className="text-muted">
                        Correct answer: {results.correct}
                      </small>
                    </>
                  )}
                </Card.Body>
              </Card>
            )}
          </>
        )}
      </div>
    </Container>
  );
}

export default App;
