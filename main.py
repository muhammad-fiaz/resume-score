import os
os.environ["STREAMLIT_WATCHDOG_IGNORE_MODULES"] = "torch,sentence_transformers"

import streamlit as st
import PyPDF2
import io
import re
import requests
from github import Github
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Load NLP models
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_md")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load Gemma model and tokenizer
        with st.spinner("Loading Gemma model... This might take a while."):
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
            model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")

        return nlp, sentence_model, tokenizer, model
    except OSError as e:
        if "en_core_web_md" in str(e):
            st.error("""
            Error: The spaCy model 'en_core_web_md' is not installed.

            Please install it by running the following command in your terminal:
            ```
            python -m spacy download en_core_web_md
            ```

            Then restart the application.
            """)
        else:
            st.error(f"Error loading models: {str(e)}")
        st.stop()

try:
    nlp, sentence_model, gemma_tokenizer, gemma_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to parse resume
def parse_resume(resume_text):
    doc = nlp(resume_text)

    # Extract skills (this is a simple approach, can be improved)
    skills = []
    skill_patterns = [
        "python", "java", "javascript", "c\\+\\+", "c#", "ruby", "php", "swift", "kotlin",
        "html", "css", "sql", "nosql", "mongodb", "mysql", "postgresql", "oracle",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "git",
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "react", "angular", "vue", "node.js", "django", "flask", "spring", "asp.net",
        "agile", "scrum", "devops", "ci/cd", "rest api", "graphql", "microservices"
    ]

    for pattern in skill_patterns:
        if re.search(pattern, resume_text.lower()):
            skills.append(pattern)

    # Extract education
    education = []
    edu_keywords = ["bachelor", "master", "phd", "degree", "university", "college", "school"]
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in edu_keywords):
            education.append(sent.text.strip())

    # Extract experience (sentences containing years or job titles)
    experience = []
    exp_keywords = ["year", "years", "engineer", "developer", "manager", "director", "lead"]
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in exp_keywords) or re.search(r'\b\d{4}\b', sent.text):
            experience.append(sent.text.strip())

    # Extract projects
    projects = []
    proj_keywords = ["project", "developed", "created", "built", "implemented", "designed"]
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in proj_keywords):
            projects.append(sent.text.strip())

    # Extract GCP and microservices experience
    gcp_experience = []
    gcp_keywords = ["gcp", "google cloud", "cloud platform", "app engine", "compute engine", 
                   "cloud storage", "bigquery", "cloud functions", "cloud run", "kubernetes engine",
                   "gke", "dataflow", "cloud sql", "firebase", "cloud spanner", "cloud bigtable"]

    microservices_experience = []
    microservices_keywords = ["microservice", "microservices", "service oriented", "soa", 
                             "api gateway", "service mesh", "containerization", "docker", "kubernetes",
                             "distributed system", "event-driven", "message queue", "service discovery"]

    # Extract sentences containing GCP keywords
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        if any(keyword in sent_lower for keyword in gcp_keywords):
            gcp_experience.append(sent.text.strip())

    # Extract sentences containing microservices keywords
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        if any(keyword in sent_lower for keyword in microservices_keywords):
            microservices_experience.append(sent.text.strip())

    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "gcp_experience": gcp_experience,
        "microservices_experience": microservices_experience,
        "full_text": resume_text
    }

# Function to analyze GitHub profile
def analyze_github_profile(github_url):
    # Extract username from URL
    username_match = re.search(r'github.com/([^/]+)', github_url)
    if not username_match:
        return None

    username = username_match.group(1)

    try:
        # Initialize PyGithub
        g = Github()
        user = g.get_user(username)

        # Get repositories
        repos = []
        languages = {}
        stars = 0
        commits = 0

        for repo in user.get_repos():
            if not repo.fork:  # Skip forked repositories
                repo_info = {
                    "name": repo.name,
                    "description": repo.description,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "last_updated": repo.updated_at
                }
                repos.append(repo_info)

                # Count stars
                stars += repo.stargazers_count

                # Count languages
                if repo.language:
                    if repo.language in languages:
                        languages[repo.language] += 1
                    else:
                        languages[repo.language] = 1

        # Sort languages by frequency
        sorted_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)

        return {
            "username": username,
            "repositories": repos,
            "languages": sorted_languages,
            "total_stars": stars,
            "total_repos": len(repos)
        }

    except Exception as e:
        st.error(f"Error analyzing GitHub profile: {str(e)}")
        return None


# Function to calculate similarity between two texts
def calculate_similarity(text1, text2):
    # Encode texts
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

# Function to display markdown content using native Streamlit components
def display_report_with_streamlit(markdown_text, container=None, is_detailed=False):
    # If no container is provided, use the current Streamlit context
    if container is None:
        container = st

    # Create a container with a border and padding
    report_container = container.container()

    # Add CSS to style the container - use more prominent styling for detailed analysis
    if is_detailed:
        report_container.markdown("""
        <style>
            [data-testid="stContainer"] {
                border: 1px solid #e0e0e0;
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                background-color: #f8f9fa;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        report_container.markdown("""
        <style>
            [data-testid="stContainer"] {
                border: 1px solid #e6e9ef;
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                background-color: white;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }
        </style>
        """, unsafe_allow_html=True)

    # If the markdown text is empty or None, display a message
    if not markdown_text:
        report_container.info("No analysis available.")
        return report_container

    # Split the markdown text into lines
    lines = markdown_text.strip().split('\n')

    # Process each line
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Handle headers
        if line.startswith('## '):
            report_container.header(line[3:])
        elif line.startswith('### '):
            report_container.subheader(line[4:])
        elif line.startswith('#### '):
            report_container.markdown(f"**{line[5:]}**")

        # Handle bullet points
        elif line.startswith('- '):
            # Collect all bullet points in this list
            bullet_points = [line[2:]]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('- '):
                bullet_points.append(lines[j].strip()[2:])
                j += 1

            # Display bullet points
            for point in bullet_points:
                # Process bold and italic formatting in bullet points
                point = process_markdown_formatting(point)
                report_container.markdown(f"• {point}")

            # Skip the lines we've already processed
            i = j - 1

        # Handle paragraphs
        elif line:
            # Check if this is a multi-line paragraph
            paragraph = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(('#', '-')):
                paragraph.append(lines[j].strip())
                j += 1

            # Join the paragraph lines
            full_paragraph = ' '.join(paragraph)

            # Process bold and italic formatting in paragraphs
            full_paragraph = process_markdown_formatting(full_paragraph)

            # Display paragraph
            report_container.markdown(full_paragraph)

            # Skip the lines we've already processed
            i = j - 1

        i += 1

    return report_container

# Helper function to process markdown formatting (bold, italic)
def process_markdown_formatting(text):
    # Process bold text (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # Process italic text (*text*)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    return text

# Function to generate detailed recommendation using Gemma model
@st.cache_data
def generate_recommendation(resume_data, github_data, job_description, evaluation):
    # Create a prompt for the model
    prompt = f"""
You are an AI Job Fit Evaluator. Based on the following information, provide a detailed recommendation about the candidate's fit for the job position.

Job Description:
{job_description}

Candidate Skills:
{', '.join(resume_data["skills"]) if resume_data["skills"] else "No skills extracted"}

Candidate Education:
{'; '.join(resume_data["education"]) if resume_data["education"] else "No education information extracted"}

Candidate Experience:
{'; '.join(resume_data["experience"]) if resume_data["experience"] else "No experience information extracted"}

Candidate Projects:
{'; '.join(resume_data["projects"]) if resume_data["projects"] else "No projects information extracted"}

GCP Experience:
{'; '.join(resume_data["gcp_experience"]) if resume_data["gcp_experience"] else "No GCP experience detected"}

Microservices Experience:
{'; '.join(resume_data["microservices_experience"]) if resume_data["microservices_experience"] else "No microservices experience detected"}

GitHub Information:
{f"Username: {github_data['username']}; Total Repositories: {github_data['total_repos']}; Top Languages: {', '.join([lang for lang, count in github_data['languages'][:3]])}" if github_data else "No GitHub information provided"}

Evaluation Scores:
Resume Score: {evaluation['resume_score']:.1f}%
GitHub Score: {evaluation['github_score']:.1f}%
{f"GCP Score: {evaluation['gcp_score']:.1f}%" if evaluation['gcp_required'] else ""}
{f"Microservices Score: {evaluation['microservices_score']:.1f}%" if evaluation['microservices_required'] else ""}
Final Fit Score: {evaluation['final_score']:.1f}%

Missing Skills:
{', '.join(evaluation['missing_skills']) if evaluation['missing_skills'] else "No missing skills identified"}

Based on the above information, provide a detailed recommendation about whether this candidate is a good fit for the job position. Consider that a Final Fit Score of 70% or higher indicates a Strong Match, 50-70% indicates a Potential Match, and below 50% indicates a Weak Match.

Pay special attention to the candidate's GCP and microservices experience, as these are often critical skills for modern cloud-based applications. If the candidate lacks experience in these areas, provide specific recommendations for how they could gain this experience.

Your response MUST be formatted in Markdown with clear sections and structure. Include the following sections:

## Overall Assessment
Provide a clear assessment of the candidate's fit for the position (Strong Match, Potential Match, or Weak Match) with a brief explanation.

## Key Strengths
List 3-5 key strengths of the candidate relevant to the job position using bullet points.

## Areas for Improvement
List specific areas where the candidate could improve to better match the job requirements using bullet points.

## Final Recommendation
Provide a final recommendation with specific next steps (e.g., proceed to interview, consider for different role, etc.).

Make your recommendation detailed, specific, and actionable. Use proper Markdown formatting including headers, bullet points, and emphasis where appropriate.
"""

    # Generate recommendation using Gemma model
    inputs = gemma_tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    recommendation = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (remove the prompt)
    recommendation = recommendation.replace(prompt, "").strip()

    return recommendation

# Function to generate specific recommendations for resume and GitHub
@st.cache_data
def generate_specific_recommendations(resume_data, github_data, job_description, evaluation):
    # Create a prompt for resume recommendations
    resume_prompt = f"""
You are an AI Job Fit Evaluator. Based on the following information, provide specific recommendations for improving the candidate's resume to better match the job requirements.

Job Description:
{job_description}

Candidate Skills:
{', '.join(resume_data["skills"]) if resume_data["skills"] else "No skills extracted"}

Candidate Education:
{'; '.join(resume_data["education"]) if resume_data["education"] else "No education information extracted"}

Candidate Experience:
{'; '.join(resume_data["experience"]) if resume_data["experience"] else "No experience information extracted"}

Candidate Projects:
{'; '.join(resume_data["projects"]) if resume_data["projects"] else "No projects information extracted"}

GCP Experience:
{'; '.join(resume_data["gcp_experience"]) if resume_data["gcp_experience"] else "No GCP experience detected"}

Microservices Experience:
{'; '.join(resume_data["microservices_experience"]) if resume_data["microservices_experience"] else "No microservices experience detected"}

Resume Score: {evaluation['resume_score']:.1f}%
{f"GCP Score: {evaluation['gcp_score']:.1f}%" if evaluation['gcp_required'] else ""}
{f"Microservices Score: {evaluation['microservices_score']:.1f}%" if evaluation['microservices_required'] else ""}

Missing Skills:
{', '.join(evaluation['missing_skills']) if evaluation['missing_skills'] else "No missing skills identified"}

Provide detailed recommendations for how the candidate could improve their resume to better match this job position. Your response MUST be formatted in Markdown with clear sections and structure.

Include the following sections:

## Resume Strengths
Identify 2-3 strengths in the candidate's resume that align well with the job requirements.

## Resume Improvement Areas
List specific areas where the resume could be improved, organized by category:

### Skills Improvements
Specific skills that should be added, emphasized, or developed further.

### GCP & Cloud Experience Improvements
Specific recommendations for gaining or highlighting GCP and cloud platform experience. If the candidate lacks GCP experience, provide detailed suggestions for how they could gain this experience through projects, certifications, or training.

### Microservices Experience Improvements
Specific recommendations for gaining or highlighting microservices experience. If the candidate lacks microservices experience, provide detailed suggestions for how they could gain this experience through projects or training.

### Experience Improvements
How the candidate could better present or enhance their work experience.

### Education/Certification Improvements
Any educational qualifications or certifications that would strengthen the application, especially cloud-related certifications like Google Cloud certifications.

### Project Highlights
Projects that should be emphasized or new projects that would demonstrate relevant skills, particularly projects that involve GCP and microservices.

## Action Plan
Provide 3-5 specific, actionable steps the candidate should take to improve their resume, in priority order.

Make your recommendations detailed, specific, and actionable. Use proper Markdown formatting including headers, bullet points, and emphasis where appropriate.
"""

    # Generate GitHub recommendations only if GitHub data is provided
    github_prompt = ""
    if github_data:
        github_prompt = f"""
You are an AI Job Fit Evaluator. Based on the following information, provide specific recommendations for improving the candidate's GitHub profile to better match the job requirements.

Job Description:
{job_description}

GitHub Information:
Username: {github_data['username']}
Total Repositories: {github_data['total_repos']}
Total Stars: {github_data['total_stars']}
Top Languages: {', '.join([lang for lang, count in github_data['languages'][:3]]) if github_data['languages'] else "None"}

GitHub Score: {evaluation['github_score']:.1f}%

Provide detailed recommendations for how the candidate could improve their GitHub profile to better match this job position. Your response MUST be formatted in Markdown with clear sections and structure.

Include the following sections:

## GitHub Profile Strengths
Identify 2-3 strengths in the candidate's GitHub profile that align well with the job requirements.

## GitHub Improvement Areas
List specific areas where the GitHub profile could be improved:

### Repository Improvements
Suggestions for new repositories or improvements to existing ones that would showcase relevant skills.

### Language/Technology Focus
Programming languages or technologies the candidate should focus on based on the job requirements.

### Contribution Quality
How the candidate could improve the quality of their contributions (code organization, documentation, etc.).

### Visibility Enhancements
Ways to increase the visibility and impact of their GitHub profile (stars, followers, etc.).

## Action Plan
Provide 3-5 specific, actionable steps the candidate should take to improve their GitHub profile, in priority order.

Make your recommendations detailed, specific, and actionable. Use proper Markdown formatting including headers, bullet points, and emphasis where appropriate.
"""

    # Generate resume recommendations
    inputs = gemma_tokenizer(resume_prompt, return_tensors="pt").to(gemma_model.device)
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    resume_recommendation = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    resume_recommendation = resume_recommendation.replace(resume_prompt, "").strip()

    # Generate GitHub recommendations if GitHub data is provided
    github_recommendation = ""
    if github_data:
        inputs = gemma_tokenizer(github_prompt, return_tensors="pt").to(gemma_model.device)
        with torch.no_grad():
            outputs = gemma_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        github_recommendation = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        github_recommendation = github_recommendation.replace(github_prompt, "").strip()

    return {
        "resume_recommendation": resume_recommendation,
        "github_recommendation": github_recommendation
    }

# Function to evaluate candidate
def evaluate_candidate(resume_data, github_data, job_description):
    # Combine resume text
    resume_text = resume_data["full_text"]

    # Create GitHub text representation
    github_text = ""
    if github_data:
        github_text += f"GitHub username: {github_data['username']}\n"
        github_text += f"Total repositories: {github_data['total_repos']}\n"
        github_text += f"Total stars: {github_data['total_stars']}\n"
        github_text += "Programming languages: "
        github_text += ", ".join([lang for lang, count in github_data['languages'][:5]])
        github_text += "\nRepository descriptions:\n"
        for repo in github_data['repositories'][:5]:  # Top 5 repos
            if repo['description']:
                github_text += f"- {repo['name']}: {repo['description']}\n"


    # Calculate resume score
    resume_similarity = calculate_similarity(resume_text, job_description)
    resume_score = min(resume_similarity * 100, 100)  # Convert to percentage

    # Calculate GitHub score
    if github_data:
        github_similarity = calculate_similarity(github_text, job_description)
        github_score = min(github_similarity * 100, 100)  # Convert to percentage
    else:
        github_score = 0

    # Calculate GCP and microservices score
    gcp_score = 0
    microservices_score = 0

    # Check if GCP is mentioned in the job description
    gcp_in_job = re.search(r'gcp|google cloud|cloud platform', job_description.lower()) is not None

    # Check if microservices is mentioned in the job description
    microservices_in_job = re.search(r'microservice|microservices|service oriented|distributed system', job_description.lower()) is not None

    # Calculate GCP score if it's mentioned in the job description
    if gcp_in_job:
        if resume_data["gcp_experience"]:
            # Calculate score based on the number of GCP experiences mentioned
            gcp_score = min(len(resume_data["gcp_experience"]) * 20, 100)  # 20 points per mention, max 100
        else:
            gcp_score = 0
    else:
        # If GCP is not mentioned in the job description, give full score
        gcp_score = 100

    # Calculate microservices score if it's mentioned in the job description
    if microservices_in_job:
        if resume_data["microservices_experience"]:
            # Calculate score based on the number of microservices experiences mentioned
            microservices_score = min(len(resume_data["microservices_experience"]) * 20, 100)  # 20 points per mention, max 100
        else:
            microservices_score = 0
    else:
        # If microservices is not mentioned in the job description, give full score
        microservices_score = 100

    # Calculate base score from resume and GitHub
    # Adjust weights: 70% resume, 30% GitHub
    base_score = (resume_score * 0.7) + (github_score * 0.3)

    # Initialize final_score with base_score
    final_score = base_score

    # Identify missing skills
    job_doc = nlp(job_description)

    # Extract potential required skills from job description
    job_skills = set()
    skill_patterns = [
        "python", "java", "javascript", "c\\+\\+", "c#", "ruby", "php", "swift", "kotlin",
        "html", "css", "sql", "nosql", "mongodb", "mysql", "postgresql", "oracle",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "git",
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "react", "angular", "vue", "node.js", "django", "flask", "spring", "asp.net",
        "agile", "scrum", "devops", "ci/cd", "rest api", "graphql", "microservices"
    ]

    for pattern in skill_patterns:
        if re.search(pattern, job_description.lower()):
            job_skills.add(pattern)

    # Find missing skills
    candidate_skills = set(resume_data["skills"])
    missing_skills = list(job_skills - candidate_skills)

    # Adjust score based on missing skills
    # If there are missing skills, calculate the proportion of skills that are present
    if job_skills:
        skill_coverage = 1.0 - (len(missing_skills) / len(job_skills))
        # Boost the score if the candidate has most of the required skills
        skill_bonus = skill_coverage * 10  # Up to 10% bonus
        final_score = min(base_score + skill_bonus, 100)

    # Adjust final score based on GCP and microservices scores if they are required in the job
    if gcp_in_job or microservices_in_job:
        # Calculate the average of GCP and microservices scores
        cloud_microservices_score = 0
        if gcp_in_job and microservices_in_job:
            cloud_microservices_score = (gcp_score + microservices_score) / 2
        elif gcp_in_job:
            cloud_microservices_score = gcp_score
        else:
            cloud_microservices_score = microservices_score

        # Adjust the final score by giving 15% weight to cloud/microservices score
        final_score = (final_score * 0.85) + (cloud_microservices_score * 0.15)
        final_score = min(final_score, 100)  # Ensure score doesn't exceed 100

    return {
        "resume_score": resume_score,
        "github_score": github_score,
        "gcp_score": gcp_score if gcp_in_job else None,
        "microservices_score": microservices_score if microservices_in_job else None,
        "final_score": final_score,
        "missing_skills": missing_skills,
        "gcp_required": gcp_in_job,
        "microservices_required": microservices_in_job
    }

# Streamlit UI
st.title("AI Job Fit Evaluator")
st.write("Upload a resume, provide a GitHub profile URL, and enter a job description to evaluate the candidate's fit.")

# Input sections
with st.expander("Resume Upload", expanded=True):
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with st.expander("GitHub Profile", expanded=True):
    github_url = st.text_input("GitHub Profile URL", placeholder="https://github.com/username")


with st.expander("Job Description", expanded=True):
    job_description = st.text_area("Enter Job Description and Requirements", height=200)

# Process and evaluate
if st.button("Evaluate Candidate"):
    if not resume_file:
        st.error("Please upload a resume.")
    elif not job_description:
        st.error("Please enter a job description.")
    else:
        with st.spinner("Analyzing candidate profile..."):
            # Process resume
            resume_text = extract_text_from_pdf(resume_file)
            resume_data = parse_resume(resume_text)

            # Process GitHub profile (if provided)
            github_data = None
            if github_url:
                github_data = analyze_github_profile(github_url)

            # Evaluate candidate
            evaluation = evaluate_candidate(resume_data, github_data, job_description)

            # Display results
            st.subheader("Evaluation Results")

            # Determine how many columns we need based on whether GCP and microservices are required
            num_metrics = 3  # Resume, GitHub, Final by default
            if evaluation['gcp_required']:
                num_metrics += 1
            if evaluation['microservices_required']:
                num_metrics += 1

            # Create columns based on the number of metrics
            cols = st.columns(num_metrics)

            # Always display Resume, GitHub, and Final scores
            cols[0].metric("Resume Score", f"{evaluation['resume_score']:.1f}%")
            cols[1].metric("GitHub Score", f"{evaluation['github_score']:.1f}%")

            # Add GCP and microservices scores if required
            col_index = 2
            if evaluation['gcp_required']:
                # Use color coding for GCP score
                if evaluation['gcp_score'] >= 70:
                    cols[col_index].metric("GCP Score", f"{evaluation['gcp_score']:.1f}%", delta="Strong")
                elif evaluation['gcp_score'] >= 40:
                    cols[col_index].metric("GCP Score", f"{evaluation['gcp_score']:.1f}%", delta="Moderate")
                else:
                    cols[col_index].metric("GCP Score", f"{evaluation['gcp_score']:.1f}%", delta="Weak", delta_color="inverse")
                col_index += 1

            if evaluation['microservices_required']:
                # Use color coding for microservices score
                if evaluation['microservices_score'] >= 70:
                    cols[col_index].metric("Microservices Score", f"{evaluation['microservices_score']:.1f}%", delta="Strong")
                elif evaluation['microservices_score'] >= 40:
                    cols[col_index].metric("Microservices Score", f"{evaluation['microservices_score']:.1f}%", delta="Moderate")
                else:
                    cols[col_index].metric("Microservices Score", f"{evaluation['microservices_score']:.1f}%", delta="Weak", delta_color="inverse")
                col_index += 1

            # Display Final Fit Score in the last column
            cols[col_index].metric("Final Fit Score", f"{evaluation['final_score']:.1f}%")

            # Generate specific recommendations for resume and GitHub
            with st.spinner("Generating specific recommendations..."):
                specific_recommendations = generate_specific_recommendations(resume_data, github_data, job_description, evaluation)

            # Display extracted information
            st.subheader("Extracted Information")

            tab_titles = ["Resume Analysis", "GitHub Analysis", "Missing Skills"]

            tabs = st.tabs(tab_titles)

            with tabs[0]:  # Resume Analysis
                st.write("**Skills:**")
                st.write(", ".join(resume_data["skills"]) if resume_data["skills"] else "No skills extracted")

                st.write("**Education:**")
                for edu in resume_data["education"]:
                    st.write(f"- {edu}")

                st.write("**Experience:**")
                for exp in resume_data["experience"]:
                    st.write(f"- {exp}")

                st.write("**Projects:**")
                for proj in resume_data["projects"]:
                    st.write(f"- {proj}")

                # Display GCP Experience
                st.write("**GCP Experience:**")
                if resume_data["gcp_experience"]:
                    for gcp_exp in resume_data["gcp_experience"]:
                        st.write(f"- {gcp_exp}")
                else:
                    st.warning("No GCP experience detected. This may be a gap in the candidate's profile.")

                # Display Microservices Experience
                st.write("**Microservices Experience:**")
                if resume_data["microservices_experience"]:
                    for ms_exp in resume_data["microservices_experience"]:
                        st.write(f"- {ms_exp}")
                else:
                    st.warning("No microservices experience detected. This may be a gap in the candidate's profile.")

                # Display resume-specific recommendations as an embedded report
                st.write("---")
                st.subheader("Resume Improvement Recommendations")

                # Display the report using native Streamlit components
                display_report_with_streamlit(specific_recommendations["resume_recommendation"])

            with tabs[1]:  # GitHub Analysis
                if github_data:
                    st.write(f"**Username:** {github_data['username']}")
                    st.write(f"**Total Repositories:** {github_data['total_repos']}")
                    st.write(f"**Total Stars:** {github_data['total_stars']}")

                    st.write("**Top Languages:**")
                    for lang, count in github_data['languages'][:5]:
                        st.write(f"- {lang}: {count} repositories")

                    st.write("**Top Repositories:**")
                    for repo in github_data['repositories'][:5]:
                        st.write(f"- **{repo['name']}** ({repo['language'] or 'Unknown'}) - {repo['stars']} stars")
                        if repo['description']:
                            st.write(f"  {repo['description']}")

                    # Display GitHub-specific recommendations as an embedded report
                    st.write("---")
                    st.subheader("GitHub Improvement Recommendations")

                    # Display the report using native Streamlit components
                    display_report_with_streamlit(specific_recommendations["github_recommendation"])
                else:
                    st.write("No GitHub profile provided or unable to analyze the profile.")


            # Missing Skills tab
            with tabs[2]:
                if evaluation["missing_skills"]:
                    st.write("**Missing Skills:**")
                    for skill in evaluation["missing_skills"]:
                        st.write(f"- {skill}")
                else:
                    st.write("No missing skills identified.")

            # Final recommendation
            st.subheader("Recommendation")

            # Generate detailed recommendation using Gemma model
            with st.spinner("Generating detailed recommendation..."):
                detailed_recommendation = generate_recommendation(resume_data, github_data, job_description, evaluation)

            # Display basic recommendation based on score
            if evaluation["final_score"] >= 70:  # Lowered threshold from 80 to 70
                st.success("Strong Match: This candidate appears to be a strong fit for the position.")
            elif evaluation["final_score"] >= 50:  # Lowered threshold from 60 to 50
                st.info("Potential Match: This candidate has potential but may need additional training or experience.")
            else:
                st.warning("Weak Match: This candidate may not be the best fit for this specific position.")

            # Display detailed recommendation as an embedded report
            st.subheader("Detailed Analysis")

            # Display the report using native Streamlit components with detailed styling
            display_report_with_streamlit(detailed_recommendation, is_detailed=True)
