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
   git clone https://github.com/muhammad-fiaz/resume-score.git
   cd resume-score
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

Run the Streamlit application:
   ```
   streamlit run main.py
   ```


## Dependencies

- streamlit: Web interface
- PyPDF2: PDF parsing
- spaCy: Natural language processing
- sentence-transformers: Semantic similarity
- PyGithub: GitHub API integration
- requests: HTTP requests

