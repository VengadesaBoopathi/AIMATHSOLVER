
# AIMATHSOLVER

AIMATHSOLVER is an **end-to-end platform for offline handwritten mathematical expression recognition (HMER)** and **adaptive problem solving**. It integrates image-based LaTeX transcription with intelligent solvers for algebra, calculus, trigonometry, and matrices — all deployable as a web app.


##  What It Does

**Input:** Handwritten math image  
**Output:** LaTeX expression + Computed solution

![image](https://github.com/user-attachments/assets/ad191b49-b842-46c2-afb4-b78e5d4c43b3)
![image](https://github.com/user-attachments/assets/c837d41e-1b8a-49ac-96cd-c5a3fef74e85)

## 🔧 Key Features

###  Handwritten Math Recognition
- ResNet-18-based feature extractor
- Self-supervised **patch-based embedding** (16×16, contrastive learning)
- Graph Attention Network (GAT) for structural modeling
- Transformer decoder for LaTeX output (bidirectional decoding)

###  Adaptive Math Solver
- Algebraic equation solver (uses SymPy or SciPy via decision tree classifier)
- Matrix operations (transpose, inverse, multiplication using NumPy)
- Trigonometric equation solving with symbolic simplification
- Differentiation and ODE solving (symbolic & numeric)

### Tech Stack
| Component     | Technology                        |
|---------------|-----------------------------------|
| Frontend      | React + Vite                      |
| Backend       | Flask                             |
| ML Framework  | PyTorch + DGL (Deep Graph Library)|
| Solver Engine | SymPy, SciPy, NumPy               |


## Demo

Input and Output for Algebra Solver
 ![image](https://github.com/user-attachments/assets/68627bcb-abe1-4d78-b43b-ae97cc98d11d)

Figure 1: HMER with Algebra Equation Solver  
Input and Output for Matrix Solver
 ![image](https://github.com/user-attachments/assets/ef8673da-75ee-48df-8c7a-bd6aa39932e1)

Figure 2: HMER with Matrix   Solver   
Input and Output for Trigonometry Solver
![image](https://github.com/user-attachments/assets/65252244-6f42-41a4-9237-3ee5ab4e072c)

Figure 3: HMER with Trigonometry Equation Solver

Input and Output for Differentiation Solver
![image](https://github.com/user-attachments/assets/402fdd2b-ca51-4164-889a-753d85388611)
Figure 4: HMER with   Differentiation Solver


##  Datasets Used

- **CROHME 2014 / 2016 / 2019**  
  Used for training and evaluating HMER accuracy

| Dataset | Samples | Accuracy (ExpRate) |
|---------|---------|-------------------|
| CROHME14 | 986     | 62.97%            |
| CROHME16 | 1,147   | 60.01%            |
| CROHME19 | 1,199   | 59.57%            |



##  Performance Metrics

- **ExpRate**: Exact match of LaTeX predictions
- **StruRate**: Structural similarity of predicted vs. ground truth
- **Top-k Accuracy**: Model prediction in top 2 candidates
- **Inference Time**: Average prediction time per image

- 
Training Loss Curve – CROHME2014
![image](https://github.com/user-attachments/assets/e723bc06-78ee-439e-b89e-d91b08bc7a5e)
Accuracy Progression – CROHME2014
![image](https://github.com/user-attachments/assets/d23c70ce-aafc-495e-b992-2ef8806b33a7)

## ⚙️ Installation & Setup

### 🖥 Backend (Flask + Solver Engine)

1. Clone the repository:

git clone https://github.com/VengadesaBoopathi/AIMATHSOLVER.git
cd AIMATHSOLVER

2. Create and activate a virtual environment:


python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate.bat   # Windows

3. Install dependencies:

pip install -r requirements.txt


4. Run Flask server:

cd src/MathSolver
python app3.py


###  Install Optional Libraries:

pip install sympy scipy numpy dgl torch torchvision opencv-python pillow scikit-learn


###  Frontend (React + Vite)

1. Navigate to frontend directory:

cd src

2. Install Node.js dependencies:

npm install


3. Start the development server:


npm run dev

4. Open `http://localhost:5173` in your browser


##  Project Structure

AIMATHSOLVER/
├── src/
│   ├── MathSolver/
│   │   ├── app3.py                  # Flask backend
│   │   ├── mcq_generator.py
│   │   ├── ocr_extraction.py
│   │   └── utils.py
│   ├── MatrixSolver.jsx            # Frontend component
│   ├── assets/
│   ├── index.css
│   ├── main.jsx                    # React entry point
├── offline-hmer*.ipynb            # Training notebooks
├── vite.config.js
└── README.md

## 🧠 Techniques Used

* **Patch-Based Embedding** with self-supervised contrastive loss (NT-Xent)
* **GAT** for learning inter-symbol relations
* **Transformer** decoder with positional encoding and autoregressive generation
* **Decision Tree Classifier** to choose between symbolic vs. numeric solving
* **SymPy / SciPy / NumPy** for equation solving




##  Future Work

* Add camera-based live input
* Improve ExpRate with beam search decoding
* Extend solver to handle integration and 3D plots



##  License

This project is licensed under the MIT License.


## Authors

* [Vengadesa Boopathi P](https://github.com/VengadesaBoopathi)
* [Kondi Nanda Gopal Umesh Raju](https://github.com/UmeshRaju) 

> Developed at Puducherry Technological University, 2025


