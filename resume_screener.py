import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

NLP = spacy.blank("en")


SKILL_DB = {
    "programming": ["python", "java", "c++", "c", "javascript", "r", "sql", "scala", "bash"],
    "ml_ai": ["machine learning", "deep learning", "nlp", "natural language processing",
              "computer vision", "neural networks", "regression", "classification",
              "clustering", "feature engineering", "model deployment", "statistics"],
    "libraries": ["pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch",
                  "spacy", "nltk", "matplotlib", "seaborn", "opencv", "flask", "django"],
    "data_tools": ["excel", "tableau", "power bi", "hadoop", "spark", "mysql",
                   "postgresql", "mongodb", "data visualization", "data analysis",
                   "data cleaning", "etl"],
    "devops_cloud": ["git", "github", "docker", "aws", "azure", "gcp", "linux", "ci/cd"],
    "soft_skills": ["communication", "teamwork", "leadership", "problem solving",
                    "time management"],
}
ALL_SKILLS = sorted({s for group in SKILL_DB.values() for s in group})


MATCHER = PhraseMatcher(NLP.vocab, attr="LOWER")
MATCHER.add("SKILLS", [NLP.make_doc(skill) for skill in ALL_SKILLS])


SAMPLE_JD = """Data Scientist (Junior)

Required skills:
Python, SQL, machine learning, pandas, numpy, scikit-learn, statistics, data visualization

Preferred skills:
NLP, git, tableau, communication
"""

SAMPLE_RESUMES = {
    "Aarav_Sharma": """Final year CSE student. Built a spam classifier using Python, pandas and
    scikit-learn. Worked with SQL databases and MySQL. Did data visualization with matplotlib
    and seaborn. Good knowledge of statistics and regression. Uses git and github daily.
    Strong communication and teamwork skills.""",
    "Priya_Nair": """Machine learning intern. Projects in NLP using spaCy and NLTK for text
    classification. Deep learning with tensorflow and keras. Python, numpy, pandas.
    Used docker for model deployment. Participated in Kaggle competitions.""",
    "Rohan_Mehta": """Web developer with 2 years of experience in JavaScript, Flask, Django
    and MongoDB. Deployed apps on AWS using docker and linux. Basic python scripting.
    Leadership experience as team lead.""",
    "Sneha_Kulkarni": """Data analyst. Expert in excel, tableau, power bi and SQL. Did data
    cleaning, data analysis and data visualization for sales reports. Some python and pandas.
    Good problem solving and communication.""",
    "Karan_Verma": """Mechanical engineering graduate interested in data science. Completed an
    online course in python. Knows excel. Looking for opportunities.""",
}

def read_pdf(path):
    """Read text from a PDF file (needs pypdf)."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return " ".join((page.extract_text() or "") for page in reader.pages)


def load_resumes_from_folder(folder):
    resumes = {}
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        name, ext = os.path.splitext(file)
        if ext.lower() == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                resumes[name] = f.read()
        elif ext.lower() == ".pdf":
            resumes[name] = read_pdf(path)
    return resumes


def load_resumes_from_csv(csv_path, category=None, limit=30):
    """Kaggle 'Resume Dataset' has columns: ID, Resume_str, Resume_html, Category."""
    df = pd.read_csv(csv_path)
    if category:
        df = df[df["Category"].str.lower() == category.lower()]
    df = df.head(limit)
    return {f"{row['Category']}_{row['ID']}": row["Resume_str"] for _, row in df.iterrows()}

def clean_text(text):
    """lowercase -> remove urls/emails/symbols -> remove stopwords -> lemmatize"""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      
    text = re.sub(r"\S+@\S+", " ", text)               
    text = re.sub(r"[^a-z0-9\s\+\#\./]", " ", text)    
    tokens = [t.strip("./") for t in text.split()]     
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and (len(t) > 1 or t == "r")]
    return " ".join(tokens)

def extract_skills(text):
    """Return a set of skills found in the text."""
    doc = NLP.make_doc(text.lower())
    found = set()
    for _, start, end in MATCHER(doc):
        found.add(doc[start:end].text)
    return found

def parse_job_description(jd_text):
    """
    Splits JD into required and preferred skills.
    Rule: skills after a line containing 'preferred' / 'nice to have' / 'good to have'
    are 'preferred', everything else is 'required'.
    """
    required_part, preferred_part = jd_text, ""
    match = re.search(r"(preferred|nice to have|good to have|bonus)", jd_text, re.IGNORECASE)
    if match:
        required_part = jd_text[:match.start()]
        preferred_part = jd_text[match.start():]

    required = extract_skills(required_part)
    preferred = extract_skills(preferred_part) - required
    return required, preferred


REQUIRED_WEIGHT = 2  
PREFERRED_WEIGHT = 1


def skill_match_score(candidate_skills, required, preferred):
    """Weighted % of job skills the candidate has."""
    total = REQUIRED_WEIGHT * len(required) + PREFERRED_WEIGHT * len(preferred)
    if total == 0:
        return 0.0
    got = (REQUIRED_WEIGHT * len(candidate_skills & required)
           + PREFERRED_WEIGHT * len(candidate_skills & preferred))
    return got / total


def screen_candidates(resumes, jd_text, w_text=0.5, w_skill=0.5):
    required, preferred = parse_job_description(jd_text)

    names = list(resumes.keys())
    cleaned = [clean_text(resumes[n]) for n in names]
    cleaned_jd = clean_text(jd_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(cleaned + [cleaned_jd])
    text_sim = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()

    rows = []
    for i, name in enumerate(names):
        skills = extract_skills(resumes[name])
        s_score = skill_match_score(skills, required, preferred)
        final = w_text * text_sim[i] + w_skill * s_score
        rows.append({
            "Candidate": name,
            "Final Score (%)": round(final * 100, 1),
            "Text Similarity (%)": round(text_sim[i] * 100, 1),
            "Skill Match (%)": round(s_score * 100, 1),
            "Matched Skills": ", ".join(sorted(skills & (required | preferred))),
            "Missing Required": ", ".join(sorted(required - skills)),
            "Missing Preferred": ", ".join(sorted(preferred - skills)),
            "Extra Skills": ", ".join(sorted(skills - required - preferred)),
        })

    df = pd.DataFrame(rows).sort_values("Final Score (%)", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    return df, required, preferred

def verdict(score):
    if score >= 70:
        return "Strong fit - shortlist"
    if score >= 45:
        return "Moderate fit - consider for interview"
    return "Weak fit - skill gaps too large"


def explain(df):
    """Plain-English explanation for non-technical recruiters."""
    print("\n" + "=" * 70)
    print("CANDIDATE RANKING REPORT")
    print("=" * 70)
    for rank, row in df.iterrows():
        print(f"\n#{rank}  {row['Candidate']}  -  {row['Final Score (%)']}%  "
              f"({verdict(row['Final Score (%)'])})")
        print(f"   Has:     {row['Matched Skills'] or 'none of the listed skills'}")
        print(f"   Missing: {row['Missing Required'] or 'nothing required'}")
        if row["Missing Preferred"]:
            print(f"   Nice-to-have missing: {row['Missing Preferred']}")

    top = df.iloc[0]
    print("\nWhy is the top candidate ranked first?")
    print(f"  {top['Candidate']} matches the most required skills "
          f"(skill match {top['Skill Match (%)']}%) and their resume wording is "
          f"closest to the job description (text similarity {top['Text Similarity (%)']}%).")


def plot_ranking(df, path="ranking_chart.png"):
    plt.figure(figsize=(9, 5))
    plt.barh(df["Candidate"][::-1], df["Final Score (%)"][::-1], color="#4a90d9")
    plt.xlabel("Final Score (%)")
    plt.title("Candidate Ranking for the Role")
    plt.xlim(0, 100)
    for i, v in enumerate(df["Final Score (%)"][::-1]):
        plt.text(v + 1, i, f"{v}%", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nChart saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Resume screening and ranking system")
    parser.add_argument("--resumes", help="folder containing .txt / .pdf resumes")
    parser.add_argument("--csv", help="Kaggle Resume.csv path")
    parser.add_argument("--category", help="category filter for the CSV, e.g. 'Data Science'")
    parser.add_argument("--limit", type=int, default=30, help="max resumes to load from CSV")
    parser.add_argument("--jd", help="path to job description .txt file")
    parser.add_argument("--top", type=int, default=10, help="how many candidates to show")
    args = parser.parse_args()

   
    if args.resumes:
        resumes = load_resumes_from_folder(args.resumes)
    elif args.csv:
        resumes = load_resumes_from_csv(args.csv, args.category, args.limit)
    else:
        print("No dataset given -> using built-in sample resumes.")
        resumes = SAMPLE_RESUMES

    
    if args.jd:
        with open(args.jd, "r", encoding="utf-8") as f:
            jd_text = f.read()
    else:
        jd_text = SAMPLE_JD

    if not resumes:
        print("No resumes found!")
        return

    df, required, preferred = screen_candidates(resumes, jd_text)
    df = df.head(args.top)

    print("Required skills in JD :", ", ".join(sorted(required)))
    print("Preferred skills in JD:", ", ".join(sorted(preferred)))

    explain(df)
    df.to_csv("ranking_results.csv")
    print("\nResults saved to ranking_results.csv")
    plot_ranking(df)


if __name__ == "__main__":
    main()