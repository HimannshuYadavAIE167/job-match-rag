import sys
from pathlib import Path
import streamlit as st

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import JobMatchPipeline
from src.resume_parser import extract_resume_text, ResumeParseError

st.set_page_config(
    page_title="JobMatch RAG | Intelligent Resume Fit Evaluator",
    page_icon="💼",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return JobMatchPipeline()


pipeline = get_pipeline()

# Sidebar: Controls & Sample Profiles
st.sidebar.title("Settings & Presets")
top_k = st.sidebar.slider("Top Job Matches to Retrieve", min_value=1, max_value=5, value=3)

sample_options = {
    "Custom (Paste or Upload)": "",
    "AI / RAG Engineer": (
        "AI / Machine Learning Engineer with deep focus on Retrieval-Augmented Generation (RAG) pipelines "
        "and agentic workflows. Built systems using LangChain, LangGraph, ChromaDB, and fine-tuned embeddings. "
        "Proficient in Python, PyTorch, Scikit-learn, FastAPI, Docker, and AWS deployments."
    ),
    "Senior Python Developer": (
        "Senior Backend Developer with 4+ years architecting distributed microservices in Python and FastAPI. "
        "Hands-on background with PostgreSQL, Redis task queues, Docker, Kubernetes, and automated CI/CD pipelines."
    ),
    "Frontend React Engineer": (
        "Frontend Developer experienced with React 18, Next.js, TypeScript, Tailwind CSS, and REST/GraphQL integrations. "
        "Focused on responsive UI design, accessible client components, and state management."
    )
}

selected_sample = st.sidebar.selectbox("Load Predefined Resume Persona:", list(sample_options.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**Architecture:**\n- `ChromaDB` Cosine Dense Search\n- Lexical Overlap Re-ranking\n- `Google Gemini 2.5 Flash` Evaluation")

# Main Interface
st.title("💼 JobMatch RAG")
st.caption("A Two-Stage Semantic Retrieval & LLM Gap-Analysis Engine for Technical Roles")

resume_text = ""

# Two input methods: Upload File or Paste Text
tab_upload, tab_paste = st.tabs(["📁 Upload Resume File (.pdf, .docx, .txt)", "✍️ Paste Resume Text"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx", "txt"],
        help="Upload a PDF, Word document, or Plain text resume."
    )
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            resume_text = extract_resume_text(uploaded_file.name, file_bytes)
            st.success(f"Parsed {uploaded_file.name} successfully ({len(resume_text.split())} words extracted).")
            with st.expander("Preview Extracted Text"):
                st.write(resume_text)
        except ResumeParseError as e:
            st.error(f"Error reading file: {e}")

with tab_paste:
    default_text = sample_options[selected_sample]
    pasted_text = st.text_area(
        "Candidate Resume / Profile Summary:",
        value=default_text,
        height=180,
        placeholder="Paste resume text or select a persona from the sidebar..."
    )
    if not resume_text and pasted_text.strip():
        resume_text = pasted_text.strip()

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True)

if analyze_btn:
    if not resume_text.strip():
        st.warning("Please upload a resume file or paste resume text before running analysis.")
    else:
        with st.spinner("Searching vector index & generating Gemini fit assessments..."):
            try:
                results = pipeline.run(
                    resume_text=resume_text,
                    top_k=top_k,
                    generate_explanations=True
                )
            except Exception as e:
                st.error(f"Execution error: {e}")
                results = []

        if not results:
            st.info("No matching job postings found. Check your indexed chunks in ChromaDB.")
        else:
            st.success(f"Retrieved and evaluated top {len(results)} matching opportunities.")

            for item in results:
                score_pct = int(item["match_score"] * 100)
                exp = item.get("explanation", {})

                with st.container():
                    st.markdown("---")
                    head_col1, head_col2 = st.columns([3, 1])
                    with head_col1:
                        st.subheader(f"#{item['rank']} — {item['title']}")
                        st.write(f"🏢 **{item['company']}** | 📍 {item['location']}")
                    with head_col2:
                        st.metric(label="Fit Score", value=f"{score_pct}%")

                    # LLM Rationale
                    rationale = exp.get("match_rationale", "No rationale provided.")
                    st.info(f"**Match Analysis:** {rationale}")

                    # Strengths vs Gaps columns
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("##### ✅ Matching Strengths")
                        strengths = exp.get("matching_strengths", [])
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.write("No direct strengths mapped.")

                    with c2:
                        st.markdown("##### ⚠️ Skill & Experience Gaps")
                        gaps = exp.get("skill_gaps", [])
                        if gaps:
                            for g in gaps:
                                st.markdown(f"- {g}")
                        else:
                            st.write("No major gaps detected.")

                    # Actionable Recommendations
                    recs = exp.get("actionable_recommendations", [])
                    if recs:
                        st.markdown("##### 💡 Actionable Next Steps")
                        for r in recs:
                            st.markdown(f"- {r}")

                    # Retrieved chunk inspect drawer
                    with st.expander("📄 View Retrieved Job Context Chunk"):
                        st.caption(f"Vector Similarity: {item['vector_similarity']} | Lexical Overlap: {item['lexical_overlap']}")
                        st.text(item["matched_context"])
                        if item.get("job_url"):
                            st.markdown(f"[View Job Posting]({item['job_url']})")