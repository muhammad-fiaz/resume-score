# AI Job Fit Evaluator

A Streamlit application that evaluates how well a candidate's profile matches a job description by analyzing their resume and GitHub profile.

## Features

- **Resume Analysis**: Extracts skills, education, experience, and projects from PDF resumes using NLP
- **GitHub Profile Analysis**: Analyzes repositories, programming languages, stars, and activity level
- **Job Description Matching**: Compares extracted data to job requirements using semantic similarity
- **Scoring System**: Provides three scores:
  - Resume Score: Relevance of resume to job description
  - GitHub Score: Relevance and quality of GitHub projects
  - Final Fit Score: Combined evaluation of the candidate's fit
- **Skills Gap Analysis**: Identifies missing or weak areas in the candidate's profile

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pycharmmiscproject
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   Or if using uv:
   ```
   uv pip install -e .
   ```

4. Download the spaCy model:
   ```
   python -m spacy download en_core_web_md
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run main.py
   ```

2. In the web interface:
   - Upload a candidate's resume (PDF format)
   - Enter the candidate's GitHub profile URL
   - Enter the job description and requirements
   - Click "Evaluate Candidate" to see the results

## How It Works

1. **Resume Parsing**:
   - Extracts text from PDF using PyPDF2
   - Uses spaCy for natural language processing
   - Identifies skills, education, experience, and projects

2. **GitHub Analysis**:
   - Fetches repository data using the GitHub API
   - Analyzes programming languages, stars, and activity
   - Evaluates relevance to the job role

3. **Matching Algorithm**:
   - Uses sentence transformers to create embeddings
   - Calculates semantic similarity between resume/GitHub and job description
   - Identifies missing skills required by the job description

4. **Scoring System**:
   - Resume Score: 0-100% based on semantic similarity to job description
   - GitHub Score: 0-100% based on relevance and quality of projects
   - Final Fit Score: Weighted average (70% resume, 30% GitHub)

## Dependencies

- streamlit: Web interface
- PyPDF2: PDF parsing
- spaCy: Natural language processing
- sentence-transformers: Semantic similarity
- PyGithub: GitHub API integration
- requests: HTTP requests

## License

[Specify your license here]
