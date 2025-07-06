import streamlit as st
import PyPDF2
import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load AI models
kw_model = KeyBERT()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Extract skill-like keywords using KeyBERT
def extract_keywords(text, top_n=25):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
    return list(set([kw[0].lower().strip() for kw in keywords if len(kw[0]) > 2]))

# Clean and normalize keywords
def clean_keywords(keywords):
    return list(set([re.sub(r'[^a-zA-Z0-9\s]', '', kw).lower().strip() for kw in keywords]))

# Semantic matcher using cosine similarity
def semantic_match(resume_keywords, jd_keywords, threshold=0.7):
    matched, missing = [], []

    resume_embeddings = embedder.encode(resume_keywords, convert_to_tensor=True)
    jd_embeddings = embedder.encode(jd_keywords, convert_to_tensor=True)

    for i, jd_kw in enumerate(jd_keywords):
        sims = util.cos_sim(jd_embeddings[i], resume_embeddings)
        max_score = sims.max().item()
        if max_score >= threshold:
            matched.append(jd_kw)
        else:
            missing.append(jd_kw)

    return matched, missing

# PDF report generation
def generate_pdf_report(match_score, resume_keywords, jd_keywords, matched_skills, missing_skills):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    def write_line(text, font_size=10):
        nonlocal y
        c.setFont("Helvetica", font_size)
        c.drawString(40, y, text)
        y -= font_size + 4
        if y < 50:
            c.showPage()
            y = height - 40

    write_line("AI Resume Scanner Report", 14)
    write_line(f"Match Score: {match_score:.2f}%", 12)
    write_line("")

    write_line("Skills Found in Resume:", 12)
    for skill in resume_keywords:
        write_line(f"• {skill}")
    write_line("")

    write_line("Skills Required in Job Description:", 12)
    for skill in jd_keywords:
        write_line(f"• {skill}")
    write_line("")

    write_line("Matched Skills:", 12)
    for skill in matched_skills:
        write_line(f"• {skill}")
    write_line("")

    write_line("Missing Skills:", 12)
    for skill in missing_skills:
        write_line(f"• {skill}")

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.title("AI-Powered Resume Scanner")

resume_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")
jd_file = st.file_uploader("📝 Upload Job Description (PDF)", type="pdf")

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    resume_keywords = clean_keywords(extract_keywords(resume_text))
    jd_keywords = clean_keywords(extract_keywords(jd_text))

    matched_skills, missing_skills = semantic_match(resume_keywords, jd_keywords)

    match_score = (len(matched_skills) / len(jd_keywords)) * 100 if jd_keywords else 0

    st.subheader("✅ Skills found in Resume")
    st.write(resume_keywords)

    st.subheader("🎯 Skills Required in JD")
    st.write(jd_keywords)

    st.subheader("📊 Match Score")
    st.metric("Score", f"{match_score:.2f}%")

    st.subheader("✅ Matched Skills")
    st.write(matched_skills)

    st.subheader("❌ Missing Skills")
    st.write(missing_skills)

    # Generate PDF and show download button
    pdf_buffer = generate_pdf_report(match_score, resume_keywords, jd_keywords, matched_skills, missing_skills)
    st.download_button(label="📄 Download PDF Report", data=pdf_buffer, file_name="resume_match_report.pdf", mime="application/pdf")
