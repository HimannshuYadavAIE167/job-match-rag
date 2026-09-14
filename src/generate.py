"""
Generates a grounded, structured explanation of how well a resume matches a
retrieved job chunk, using Gemini. If no API key is configured or the API
call fails for any reason, falls back to a deterministic keyword-overlap
explainer so the app never breaks in front of a user or interviewer.
"""
import json
import logging
import re
from typing import Any

from src.config import GENERATION_MODEL_NAME, get_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are an expert technical talent evaluator and career coach.
Analyze the alignment between a candidate's resume and a retrieved job posting chunk.
Evaluate objectively: do not flatter, do not hallucinate skills not present in the resume, and highlight genuine gaps.

Return your response strictly in valid JSON format with this exact structure:
{
  "match_rationale": "2 concise sentences explaining the fit",
  "matching_strengths": ["list of 2-4 specific strengths matching the requirements"],
  "skill_gaps": ["list of 1-3 missing skills, tools, or experience gaps"],
  "actionable_recommendations": ["1-2 concrete ways to improve the resume or profile for this specific role"]
}
Only output valid JSON. Do not include markdown code fences or conversational commentary."""


class MatchExplainer:
    def __init__(self, model_name: str = GENERATION_MODEL_NAME):
        self.api_key = get_api_key()
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized successfully with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("No valid API key found (checked st.secrets and environment). Falling back to heuristic explainer.")

    def explain_match(
        self,
        resume_text: str,
        job_title: str,
        company: str,
        job_chunk: str,
        match_score: float,
    ) -> dict[str, Any]:
        """Generates a structured gap analysis and fit summary using Gemini."""
        if self.client:
            try:
                user_prompt = f"""Target Role: {job_title} at {company}
Match Score: {match_score:.2f}

[Job Context / Requirement Chunk]:
{job_chunk}

[Candidate Resume / Profile]:
{resume_text}

Generate the JSON evaluation."""

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config={
                        "system_instruction": SYSTEM_INSTRUCTION,
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    },
                )

                content_text = response.text.strip()
                if content_text.startswith("```"):
                    content_text = re.sub(r"^```[a-zA-Z]*\n?", "", content_text)
                    content_text = re.sub(r"\n?```$", "", content_text).strip()

                return json.loads(content_text)
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Falling back to offline generator.")

        return self._generate_fallback_explanation(resume_text, job_title, job_chunk, match_score)

    def _generate_fallback_explanation(
        self,
        resume_text: str,
        job_title: str,
        job_chunk: str,
        match_score: float,
    ) -> dict[str, Any]:
        resume_lower = resume_text.lower()
        chunk_lower = job_chunk.lower()

        tech_keywords = [
            "python", "pytorch", "tensorflow", "rag", "langchain", "langgraph",
            "docker", "kubernetes", "aws", "gcp", "fastapi", "chromadb", "sql",
        ]

        found_strengths = [kw for kw in tech_keywords if kw in chunk_lower and kw in resume_lower]
        missing_skills = [kw for kw in tech_keywords if kw in chunk_lower and kw not in resume_lower]

        strengths = [f"Direct alignment with required competency in {tech.upper()}." for tech in found_strengths[:3]]
        if not strengths:
            strengths = ["General software engineering and foundational technical background match."]

        gaps = [f"Resume does not explicitly emphasize {tech.upper()} as requested in the posting." for tech in missing_skills[:3]]
        if not gaps:
            gaps = ["No critical technology mismatch detected; ensure domain projects are clearly quantified."]

        return {
            "match_rationale": (
                f"Candidate exhibits a {round(match_score * 100, 1)}% composite semantic alignment for the {job_title} role. "
                "Core background overlaps with primary engineering requirements."
            ),
            "matching_strengths": strengths,
            "skill_gaps": gaps,
            "actionable_recommendations": [
                "Quantify business and engineering impact in relevant bullet points.",
                "Explicitly highlight missing tools in your core skills and project descriptions.",
            ],
        }


if __name__ == "__main__":
    sample_resume = (
        "AI / ML Engineer with expertise in building scalable Retrieval-Augmented Generation (RAG) "
        "pipelines using LangChain, LangGraph, and ChromaDB. Proficient in Python, PyTorch, Hugging Face, "
        "and deploying containerized microservices using Docker, Kubernetes, and AWS."
    )

    sample_job_title = "Machine Learning Engineer"
    sample_company = "Cognitive Scale AI"
    sample_chunk = (
        "Requirements:\n"
        "- Strong proficiency in Python, PyTorch, and Scikit-learn.\n"
        "- 3+ years experience designing vector search or semantic retrieval systems.\n"
        "- Hands-on production experience with Docker, Kubernetes, and AWS (SageMaker, S3, ECS).\n"
        "- Experience with Kafka and distributed streaming pipelines."
    )

    explainer = MatchExplainer()
    explanation = explainer.explain_match(
        resume_text=sample_resume,
        job_title=sample_job_title,
        company=sample_company,
        job_chunk=sample_chunk,
        match_score=0.88,
    )

    print("\n" + "=" * 60)
    print("GEMINI MATCH EXPLANATION & GAP ANALYSIS")
    print("=" * 60)
    print(json.dumps(explanation, indent=2))
