from apify_client import ApifyClient
import dotenv
import os
import pprint
from typing import List, Optional, Dict, Any
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from langgraph.config import get_stream_writer
import logfire
from langchain_core.messages import AIMessage
logfire.configure()  
logfire.instrument_pydantic_ai()
  

# Load environment variables from .env file
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


job_retrieval_system_prompt = """ 
You are an expert Industry Job Description Synthesizer, trained in HR standards, labor taxonomy (e.g., ESCO, O*NET), and job market intelligence.
You are tasked with generating a high-quality, concise, and industry-aligned job description by synthesizing data from multiple real-world job postings.
Your user will provide a target job role or title. Your job is to retrieve and analyze multiple job descriptions for that role from LinkedIn (or similar platforms), identify common patterns, extract essential skills, 
responsibilities, qualifications, and terminology, and write a refined, professional, and standardized job description suitable for posting by a Fortune 500 HR team.

### Objectives:
- Retrieve multiple job postings (minimum 10) for the **user-specified job role**.
- Extract key responsibilities, required skills, qualifications, tools/technologies, and industry-specific phrases.
- Identify frequency-weighted core themes (e.g., skills or tasks that appear in 80%+ of listings).
- Identify conceptually similar responsibilities and skills across job listings, consolidate them into unified statements, and eliminate redundancy using natural language generalization and abstraction
- Prioritize **clarity, inclusivity**, and **ATS-optimization** in the final output.
- Ensure tone is **professional**, **neutral**, and **aligned with global hiring standards**.
- Avoid copying exact phrasing. Rewrite with precision and modern language.

### Output Format:
Respond only with the following 5 sections:
1. **Job Title**: [Concise and accurate]
2. **Overview**: [Write in the voice of a modern company job description. Begin with a phrase like “We’re looking for...” or “As a [Job Title]...”. Avoid passive phrases.]
3. **Key Responsibilities/Key Deliverables**: [Bullet points with clear, action-driven verbs]
4. **Required Skills & Qualifications**: [Bullet list]
5. **Preferred (Nice-to-Have)**: [If applicable]

### Constraints:
- Do **not** include company-specific details.
- Avoid redundancy or overly generic phrases.
- Ensure all content is **original**, **well-structured**, and reflects actual hiring trends.
- Prioritize recency by analyzing **current listings only**.

### Final Notes
- Keep using the tools until you get enough knowledge to generate the job description.
- Do not make up any information.
- If the tool call for the job role provided by the user is empty, then try searching by different variations of the samejob role name.

"""

model = OpenAIModel("gpt-4o",provider=OpenAIProvider(api_key=openai_api_key))

job_retrieval_agent = Agent(
    name="job_retrieval_agent",
    model=model, 
    system_prompt=job_retrieval_system_prompt,
    retries=3,
    )




def format_job_listings_for_llm(job_listings: List[Dict[str, Any]]) -> str:
    """
    Converts a list of scraped job dictionaries into a clean, markdown-like
    string that an LLM can easily parse for summarization.

    It extracts key details like title, company, location, salary, and the
    full job description, formatting them for clarity.
    """
    formatted_texts = []
    for i, job in enumerate(job_listings):
        lines = [f"--- Job Listing {i+1} ---"]

        # Core job details
        lines.append(f"Title: {job.get('job_title', 'N/A')}")
        lines.append(f"Company: {job.get('company', 'N/A')}")
        lines.append(f"Location: {job.get('location', 'N/A')}")
        
        if work_type := job.get('work_type'):
            lines.append(f"Work Type: {work_type}")
        if salary := job.get('salary'):
            lines.append(f"Salary: {salary}")
        
        lines.append(f"Posted At: {job.get('posted_at', 'N/A')}")
        
        # Skills list, if available
        if skills := job.get('skills'):
            if isinstance(skills, list) and skills:
                 lines.append(f"Skills: {', '.join(skills)}")

        # The full job description is the most important part
        if description := job.get('description'):
            # Clean up excessive whitespace and remove duplicate lines
            lines_seen = set()
            unique_lines = []
            for line in description.strip().split('\n'):
                cleaned_line = line.strip()
                if cleaned_line and cleaned_line not in lines_seen:
                    unique_lines.append(cleaned_line)
                    lines_seen.add(cleaned_line)
            
            cleaned_description = '\n'.join(unique_lines)
            lines.append(f"\nFull Description:\n{cleaned_description}")
        
        # Link for reference
        if job_url := job.get('job_url'):
            lines.append(f"\nURL: {job_url}")

        formatted_texts.append("\n".join(lines))

    # Join all individual job strings with a clear separator
    return "\n\n====================\n\n".join(formatted_texts)


@job_retrieval_agent.tool_plain
def search_linkedin_jobs(
    keywords: str,
    location: str,
    page_number: int,
    limit: int,
    sort: str,
    date_posted: str = "",
    experienceLevel: str = "",
) -> List[Dict[str, Any]]:
    """
    Searches and scrapes job postings from LinkedIn.
    Args:
        keywords (str): The job keywords to search for (e.g., "Data Engineer").
        location (str): The location for the job search. Defaults to "India".
        page_number (int): The page number of results to return. Defaults to 1.
        limit (int): The number of jobs to scrape. Defaults to 5.
        sort (str): The sorting order of the jobs. Default values are "relevant" or "recent".
        date_posted (str): Show results on date posted. Default is ["", "month", "week", "day"]
        experienceLevel (str): Filter by experience level. ["internship", "entry", "associate", "mid_senior", "director", "executive"]
    Returns:
        List[Dict[str, Any]]: A list of job items from the Apify dataset.
    """
    print(f"🚀 Starting LinkedIn job search for '{keywords}' in '{location}'...")

    # Prepare the Actor input
    apify_token = os.getenv("APIFY_API_TOKEN")
    client = ApifyClient(apify_token)
    run_input = {
        "keywords": keywords,
        "location": location,
        "page_number": page_number,
        "limit": limit,
        "sort": sort,
        "date_posted": date_posted,
        "experienceLevel": experienceLevel
    }

    # Run the Actor and wait for it to finish
    run = client.actor("apimaestro/linkedin-jobs-scraper-api").call(run_input=run_input)

    # Fetch and print Actor results from the run's dataset (if there are any)
    print("✅ Job search finished.")
    print("💾 You can view the full results here: https://console.apify.com/storage/datasets/" + run["defaultDatasetId"])
    retieved_jobs = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    pprint.pprint(retieved_jobs)
    formatted_jobs = format_job_listings_for_llm(retieved_jobs)
    print("--------------------------------------------------------")
    print(formatted_jobs)
    return formatted_jobs


job_matcher_system_prompt = """ 
You are a professional AI career analyst trained in evaluating candidate-job fit. You are given:

- **User Profile**: A summarized job profile representing the candidate’s background, skills, experience, education, projects, and soft skills.
- **Industry Standard Job Description**: A synthesized job description created by summarizing multiple relevant listings from LinkedIn for the target role.

Your task is to **evaluate how well the user profile matches the job description**. Return a structured scoring and reasoning across the following axes:

1. **Skill Match (Score: /10)**  
   - How closely do the user's listed hard skills align with the required skills in the job description?

2. **Experience Relevance (Score: /10)**  
   - Do the job titles and responsibilities in the user profile match the expectations of the target role?  
   - Is the seniority level appropriate based on years of experience?

3. **Domain Fit (Score: /10)**  
   - Is there alignment in industry, business function, or technological domain?  
   - Do the tools, platforms, or methodologies used by the candidate match those required?

4. **Education Alignment (Score: /10)**  
   - Does the candidate meet the degree, institution, or field of study requirements?  
   - Are any advanced degrees or certifications relevant to the role?

5. **Project & Achievement Relevance (Score: /10)**  
   - Are there specific projects, initiatives, or outcomes in the candidate’s background that clearly demonstrate job-relevant impact or capabilities?

6. **Soft Skills / Culture Fit (Score: /10)**  
   - Are there indications of leadership, communication, teamwork, or initiative from the profile or recommendations?  
   - Does the profile reflect keywords or behaviors (e.g., “agile”, “cross-functional”) that align with the role?

"""


# --- Pydantic Models for Structured Output ---

class ScoreWithReasoning(BaseModel):
    """A score from 0-10 with a justification."""
    score: int = Field(..., ge=0, le=10, description="The score from 0 to 10.")
    reasoning: str = Field(..., description="The detailed reasoning behind the score.")

class MatchReport(BaseModel):
    """The final structured report for the job match analysis."""
    skill_match: ScoreWithReasoning = Field(..., alias="Skill Match")
    experience_relevance: ScoreWithReasoning = Field(..., alias="Experience Relevance")
    domain_fit: ScoreWithReasoning = Field(..., alias="Domain Fit")
    education_alignment: ScoreWithReasoning = Field(..., alias="Education Alignment")
    project_achievement_relevance: ScoreWithReasoning = Field(..., alias="Project & Achievement Relevance")
    soft_skills_culture_fit: ScoreWithReasoning = Field(..., alias="Soft Skills / Culture Fit")
    overall_fit_summary: str = Field(..., alias="Overall Fit Summary")


# --- LangGraph Node Function (LangChain with Structured Output) ---
def match_report_table_with_summary(report) -> str:
    if hasattr(report, "dict"):
        report = report.dict()

    key_to_label = {
        "skill_match": "Skill Match",
        "experience_relevance": "Experience Relevance",
        "domain_fit": "Domain Fit",
        "education_alignment": "Education Alignment",
        "project_achievement_relevance": "Project & Achievement Relevance",
        "soft_skills_culture_fit": "Soft Skills / Culture Fit"
    }

    # Prepare row data
    rows = []
    for key, label in key_to_label.items():
        item = report[key]
        score = str(item["score"])
        reasoning = item["reasoning"].strip().replace("\n", " ")
        short_reason = reasoning[:80].strip() + "..." if len(reasoning) > 80 else reasoning
        rows.append((label, score, short_reason))

    # Compute max widths
    col1_width = max(len(r[0]) for r in rows + [("Category",)])
    col2_width = max(len(r[1]) for r in rows + [("Score",)])
    col3_width = max(len(r[2]) for r in rows + [("Reasoning (Short)",)])

    # Format table header and separator
    header = f"| {'Category'.ljust(col1_width)} | {'Score'.center(col2_width)} | {'Reasoning (Short)'.ljust(col3_width)} |"
    separator = f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|{'-' * (col3_width + 2)}|"

    # Format rows
    table_lines = [header, separator]
    for label, score, reason in rows:
        line = f"| {label.ljust(col1_width)} | {score.center(col2_width)} | {reason.ljust(col3_width)} |"
        table_lines.append(line)

    # Add full summary
    summary = report.get("overall_fit_summary", "").strip()
    table_lines.append("\n**Overall Fit Summary**:\n" + summary)

    return "\n".join(table_lines)



def format_match_report_as_bullets(report) -> str:
    if hasattr(report, "dict"):
        report = report.dict()

    key_to_label = {
        "skill_match": "Skill Match",
        "experience_relevance": "Experience Relevance",
        "domain_fit": "Domain Fit",
        "education_alignment": "Education Alignment",
        "project_achievement_relevance": "Project & Achievement Relevance",
        "soft_skills_culture_fit": "Soft Skills / Culture Fit"
    }

    lines = []
    for key, label in key_to_label.items():
        item = report.get(key, {})
        score = item.get("score", "N/A")
        reasoning = item.get("reasoning", "N/A")
        lines.append(f"### {label} ({score}/10)\n{reasoning.strip()}\n")

    summary = report.get("overall_fit_summary", "")
    if summary:
        lines.append("### 🧾 Overall Fit Summary\n" + summary.strip())

    return "\n".join(lines)

def job_matcher_node(state: dict) -> dict:
    """
    A LangGraph node that uses LangChain's `.with_structured_output()` to generate
    a structured match report from a user profile and a job description.

    Args:
        state (dict): The current graph state, expected to contain 'user_profile'
                      and 'industry_job_description'.

    Returns:
        dict: The updated state with the 'match_report' added.
    """
    print("🤖 Running Job Matcher Node (LangChain Structured Output)...")
    user_profile = state.get("user_profile")
    industry_job_description = state.get("job_description")

    if not user_profile or not industry_job_description:
        raise ValueError("Node requires 'user_profile' and 'industry_job_description' in state.")

    # Initialize the LangChain model, binding it to our Pydantic class
    # This instructs the LLM to return JSON that matches the MatchReport schema
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(MatchReport)

    # The prompt template now has placeholders for our inputs
    prompt = ChatPromptTemplate.from_messages([
        ("system", job_matcher_system_prompt),
        ("human", 
         "User Profile:\n{user_profile}\n\n"
         "Industry Standard Job Description:\n{industry_job_description}"
        )
    ])

    # Create the chain
    chain = prompt | structured_llm

    # Invoke the chain with the inputs
    try:
        match_report = chain.invoke({
            "user_profile": user_profile,
            "industry_job_description": industry_job_description
        })
        
        print("✅ Match report generated and validated successfully by LangChain.")
        # The output is already a Pydantic object, so we just convert it to a dict
        match_report= format_match_report_as_bullets(match_report.dict())

    except Exception as e:
        print(f"❌ LangChain structured output chain failed: {e}")
        raise RuntimeError("Failed to generate a valid match report using the LangChain chain.") from e
    
    return {
        "match_report": match_report,
        "agent_response": match_report
            }




if __name__ == "__main__":
    apify_token = os.getenv("APIFY_API_TOKEN")
    if not apify_token:
        raise ValueError("APIFY_API_TOKEN environment variable not set! Please add it to your .env file.")
        
    client = ApifyClient(apify_token)

    # --- Example Usage ---
    # Search for "ML Engineer" jobs in the United States, based on test.py.
    job_keywords_to_search = "Senior ML Engineer"
    response = job_retrieval_agent.run_sync(f"Can you generate a description for: {job_keywords_to_search}")
    print(response.output)

    # Call the function to search for jobs
    # job_listings = search_linkedin_jobs(keywords=job_keywords_to_search, limit=5)
    
    # Print a summary of the results
    # if job_listings:
    #     print(f"\nFound {len(job_listings)} job listings for '{job_keywords_to_search}'.")
    #     print("Here are the results:")
    #     for i, job in enumerate(job_listings):
    #         print(f"\n--- Job {i+1} ---")
    #         pprint.pprint(job)
    # else:
    #     print(f"No job listings found for '{job_keywords_to_search}'.")