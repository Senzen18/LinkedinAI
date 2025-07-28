import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from pydantic_ai.common_tools.tavily import tavily_search_tool


dotenv.load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")

content_gen_system_prompt = """
You are a LinkedIn Profile Optimization Agent designed to rewrite sections of a user's LinkedIn profile to align with current industry best practices and a target job description.

Given:
- The user's scraped LinkedIn profile
- One or more target job descriptions (if provided)
- Access to a Tavily web search tool for querying authoritative sources on LinkedIn optimization and job-specific requirements
- Access to a retreive_linkedin_knowledge_chunks tool for retrieving the most relevant chunks of blogs on how to improve your linkedin profile.

Your task is to:
1. Analyze the job description(s) and extract key role expectations, tone, and essential skills.
2. Identify gaps, redundancies, or outdated language in the user profile in relation to these **expectations**.
3. Perform focused Tavily searches when additional guidance is needed (e.g., “how to write a compelling LinkedIn About section for a Product Manager,” or “top skills for data science roles 2025”).
4. Rewrite profile sections (e.g., Headline, About, Experience) to:
   - Highlight the most relevant skills, achievements, and domain experience
   - Match tone and structure commonly seen in high-performing LinkedIn profiles
   - Integrate role-specific keywords to improve recruiter visibility
5. Output rewritten sections only. Keep each section succinct, human-readable, and aligned with LinkedIn’s best practices and the job’s expectations.


#Final Notes
- Keep using the tools until you get enough knowledge to rewrite the profile.
- Do not shy away from using the tools.

Think like a career branding expert and use copywriting principles to craft impactful, personalized LinkedIn profile content. Be concise, outcome-driven, and prioritize relevance.

 """

model = OpenAIModel("gpt-4o",provider=OpenAIProvider(api_key=open_ai_api_key))
tavily_search = tavily_search_tool(api_key=os.getenv("TAVILY_API_KEY"))
content_gen_agent = Agent(
    name="content_gen_agent",
    model=model,
    system_prompt=content_gen_system_prompt,
    retries=5,
    tools=[tavily_search]
)
@content_gen_agent.tool_plain
def retreive_linkedin_knowledge_chunks(query:str, k:int=3):
    """
    Retrieves the most relevant chunks of blogs on how to improve your linkedin profile.
    Args:
        query (str): The query to search for.
        k (int): The number of chunks to return.
    Returns:
        list: A list of chunks that are most relevant to the query.
    """
    print(f"Retrieving {k} chunks for query: {query}")
    path = "vectordb"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=open_ai_api_key)
    vectordb = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    chunks = vectordb.similarity_search(query, k=k)
    # Extract the relevant text content from the chunks and format as a readable string for the LLM
    if not chunks:
        return "No relevant LinkedIn optimization knowledge found."
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        # Each chunk is likely a Document with .page_content and possibly .metadata
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        # Optionally, include the source or title from metadata for context
        meta = getattr(chunk, "metadata", {})
        title = meta.get("title")
        page = meta.get("page_label") or meta.get("page")
        source = meta.get("source")
        header = f"Chunk {i}:"
        if title:
            header += f" {title}"
        if page:
            header += f" (Page {page})"
        # if source:
        #     header += f"\nSource: {source}"
        formatted_chunks.append(f"{header}\n{text.strip()}")
    # print("\n\n".join(formatted_chunks))
    return "\n\n".join(formatted_chunks)

if __name__ == "__main__":
    from profile_analyzer_agent import *
    from job_matcher_agent import *
    # user_profile = scrape_linkedin_profile("https://www.linkedin.com/in/sendhan-a-38a48811b/")
    # job_description = "ML Engineer"
    # response = job_retrieval_agent.run_sync(job_description)
    # user_profile = scrape_linkedin_profile("https://www.linkedin.com/in/sendhan-a-38a48811b/")

    # Example user profile for demonstration purposes
    example_user_profile = """
    Name: Alex Johnson
    Current Title: Data Scientist at Acme Corp
    Location: San Francisco, CA
    Experience:
      - Data Scientist, Acme Corp (2021-Present)
        - Built and deployed machine learning models for customer segmentation and churn prediction.
        - Led a team of 3 in developing a real-time analytics dashboard using Python and Tableau.
      - Data Analyst, Beta Analytics (2018-2021)
        - Automated ETL pipelines and improved reporting efficiency by 30%.
        - Conducted A/B testing and presented actionable insights to stakeholders.
    Education:
      - M.S. in Data Science, Stanford University (2016-2018)
      - B.S. in Statistics, UCLA (2012-2016)
    Skills: Python, SQL, Machine Learning, Data Visualization, Tableau, Deep Learning, Communication, Leadership
    Certifications: AWS Certified Machine Learning – Specialty
    Projects:
      - Developed a recommendation engine for e-commerce personalization (Python, scikit-learn)
      - Open-source contributor to pandas and scikit-learn
    About:
      Passionate data scientist with 5+ years of experience in analytics and machine learning. Adept at translating business needs into technical solutions and driving measurable impact.
    """

    # Example job description for demonstration purposes
    example_job_description = """
    Job Title: Machine Learning Engineer

    Overview:
    We’re looking for a Machine Learning Engineer to design, build, and deploy scalable machine learning solutions for our core products. As a key member of our data science team, you will collaborate with engineers, analysts, and product managers to deliver impactful models and data-driven features.

    Key Responsibilities/Key Deliverables:
    - Develop, test, and deploy machine learning models for production use.
    - Collaborate with cross-functional teams to define data requirements and deliver insights.
    - Optimize model performance and ensure scalability.
    - Maintain and improve existing ML pipelines and infrastructure.
    - Communicate results and recommendations to technical and non-technical stakeholders.

    Required Skills & Qualifications:
    - Proficiency in Python and experience with ML libraries (scikit-learn, TensorFlow, PyTorch).
    - Strong understanding of data structures, algorithms, and statistics.
    - Experience with cloud platforms (AWS, GCP, or Azure).
    - Bachelor’s or Master’s degree in Computer Science, Data Science, or related field.
    - Excellent problem-solving and communication skills.

    Preferred (Nice-to-Have):
    - Experience with MLOps tools and CI/CD pipelines.
    - Prior work in e-commerce or SaaS environments.
    - Contributions to open-source ML projects.
    """
    # response = retreive_linkedin_knowledge_chunks("how to improve your linkedin profile")
    user_prompt = f"""
    User Profile:
    {example_user_profile}
    Job Description:
    {example_job_description}
    """
    response = content_gen_agent.run_sync(user_prompt)
    print(response.output)