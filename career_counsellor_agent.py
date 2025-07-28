from apify_client import ApifyClient
import os
import dotenv
import pprint
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire
import time
logfire.configure()  
logfire.instrument_pydantic_ai()

dotenv.load_dotenv()
apify_api_token = os.getenv("APIFY_API_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ApifyClient with your Apify API token
model = OpenAIModel("gpt-4o",provider=OpenAIProvider(api_key=openai_api_key))

career_counsellor_system_prompt = """
You are a **Career Coaching AI Agent** that assists job seekers in bridging the gap between their current LinkedIn profile and a desired job role by suggesting **a structured learning path** using high-quality **Coursera courses**. Your recommendations must be **personalized**, **realistic**, and **strategically ordered** to guide the user from their current skill level to the requirements of the target job.

---

### INPUT

You are given:

1. **User LinkedIn Profile Data** (JSON format)  
   Includes:  
   - `name`, `title`, `headline`, `summary`  
   - `skills`: List of technical and soft skills  
   - `experience`: List of past roles, responsibilities, tools used  
   - `education`  

2. **Target Job Description** (Structured or raw text)  
   Includes:  
   - Role Title  
   - Responsibilities  
   - Required Skills (hard & soft)  
   - Preferred Background / Experience  

3. **Search Tool**: `search_coursera(course_name)`  
   Given a course name or keyword, this returns metadata about Coursera courses including title, description, level (Beginner/Intermediate/Advanced), and URL.

---

### TASK OBJECTIVES

Your goal is to:

1. **Analyze the Skill Gap**  
   - Compare the user's current skill set and experiences with the skills and qualifications required in the target job.
   - Identify missing, outdated, or insufficiently developed skills.

2. **Construct a Personalized Learning Path**  
   - Recommend a **sequenced list of Coursera courses** that build the missing skills in logical learning order (e.g., foundational → intermediate → advanced).
   - Organize the courses in **tracks** if multiple skill clusters are required (e.g., ML + Cloud + Communication).
   - Assign a learning level or estimated time to each step if possible.

3. **Justify and Explain**  
   - For each course, include:
     - Why it’s being recommended (tie to skill gap or job role)
     - Course level (beginner/intermediate/advanced)
     - How it fits into the career path progression

---

### OUTPUT FORMAT (Markdown)

```markdown
##  Career Path & Learning Plan for [User Name] → [Target Role]

###  Skill Gap Analysis
| Skill | Present in Profile | Required in Job | Gap | Priority |
|-------|--------------------|------------------|------|----------|
| Python |        true        |       true        | false   | High     |
| TensorFlow |        false     |       true        | true  | High     |
| Cloud Platforms |   false     | true              | true  | Medium   |
| Communication |        true |       true        | false   | Low      |

---

###  Recommended Learning Path Example

####  Track 1: Machine Learning Foundations
1. **[Course Title 1]** - Beginner  
   _Why_: Covers ML fundamentals missing from the profile.  
   _url_: https://www.coursera.org/learn/machine-learning-foundations

2. **[Course Title 2]** - Intermediate  
   _Why_: Bridges foundation to TensorFlow and model deployment.  
   _url_: https://www.coursera.org/learn/tensorflow-intermediate

#### Track 2: Cloud Computing
1. **[Course Title 3]** - Beginner  
   _Why_: Introduces cloud architecture basics.  
   _url_: https://www.coursera.org/learn/aws-beginner

2. **[Course Title 4]** - Intermediate  
   _Why_: Focuses on ML deployment using GCP.  
   _url_: https://www.coursera.org/learn/ml-gcp

####  Track 3: Communication & Teamwork
1. **[Course Title 5]** - All levels  
   _Why_: Enhances communication in cross-functional teams.  
   _url_: https://www.coursera.org/learn/team-communication

---

###  Final Notes
- Keep using the tool to get courses until all the skill gaps are filled.
- You are advised to follow the tracks sequentially.
- Use only the names and urls extracted from the tool output.
- Do not make up any information.
- Do not use any other information than the tool output.




"""

career_counsellor_agent = Agent(
    name="career_counsellor_agent",
    model=model,
    system_prompt=career_counsellor_system_prompt,
    retries=8,
)
def format_courses(courses: list) -> str:
    """
    Format a list returned by ``get_courses`` for LLM consumption.

    Args:
        courses: List of course metadata dicts from Apify Coursera scraper.

    Returns:
        Human-readable multi-line string summarising all relevant fields for each course.
    """
    if not courses:
        return "No courses found."

    formatted_lines = []
    for idx, course in enumerate(courses, 1):
        name = course.get("name", "Unknown title")
        level = course.get("productDifficultyLevel") or course.get("level") or "N/A"
        url = course.get("url", "")
        partners = ", ".join(course.get("partners", [])) or "Unknown partner"
        rating = course.get("avgProductRating")
        rating_str = f"{rating:.2f}" if isinstance(rating, (int, float)) else "N/A"
        num_ratings = course.get("numProductRatings", "N/A")
        # If the parameter is explicitly False, keep it as False, else use the value or "N/A"
        is_free = course.get("isCourseFree") if course.get("isCourseFree") is not None else "N/A"
        is_coursera_plus = course.get("isPartOfCourseraPlus") if course.get("isPartOfCourseraPlus") is not None else "N/A"
        duration = course.get("productDuration", "N/A")
        course_type = course.get("productType", "N/A")
        skills = course.get("skills", [])
        skills_str = ", ".join(skills) if skills else "N/A"
        image_url = course.get("imageUrl", "")
        parent_course = course.get("parentCourseName", None)
        translated_name = course.get("translatedName", None)

        formatted_lines.append(
            f"{idx}. {name} (Level: {level}, Type: {course_type}, Rating: {rating_str} from {num_ratings} ratings)\n"
            f"   Provider(s): {partners}\n"
            f"   Free: {is_free} | Coursera Plus: {is_coursera_plus}\n"
            f"   Duration: {duration}\n"
            f"   Skills: {skills_str}\n"
            f"   URL: {url}\n"
            f"   Image: {image_url}"
            + (f"\n   Parent Course: {parent_course}" if parent_course else "")
            + (f"\n   Translated Name: {translated_name}" if translated_name else "")
        )

    return "\n\n".join(formatted_lines)

@career_counsellor_agent.tool_plain
def get_courses(query: str, limit: int = 2):
    """
    This tool is used to retrieve courses from Coursera based on a search query and limit.
    It is used to get the courses that are relevant courses for the user to learn to get the job or to improve their skills.


    Args:
        query: The search term for courses (e.g. "Machine Learning, Deep Learning, etc.")
        limit: The maximum number of courses to return.

    Returns:
        A list of course details.
    """

    try:
        client = ApifyClient(apify_api_token)

        # Prepare the Actor input
        run_input = {"query": query, "limit": limit}

        # Run the Actor and wait for it to finish
        run = client.actor("piotrv1001/coursera-search-scraper").call(run_input=run_input)

        # Fetch and print Actor results from the run's dataset (if there are any)
        print(
            "Check your data here: https://console.apify.com/storage/datasets/"
            + run["defaultDatasetId"]
        )
        course_list = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            course_list.append(item)
        time.sleep(500)
        return format_courses(course_list)
    except Exception as e:
        print(f"Error getting courses: {str(e)}")
        return f"Error getting courses: {str(e)}"

if __name__ == "__main__":
    # courses = get_courses("Machine Learning", limit=3)
    # print(courses)

    example_user_profile = """
    ## Name
- Sendhan A

## Headline
- Gen AI intern @ Terra

## About
I’m Sendhan, an AI Engineer blending creativity with deep technical skill. From automating game development pipelines through AI to building LLM-powered bots for resume screening, I love solving complex problems where vision, language, and design meet.

Skilled across PyTorch, LangChain,LnagGraph and NLP, I thrive on turning bold ideas into real-world products. Let’s connect—I’m ready to help invent what’s next.

Let’s connect and explore what we can create together!

## Experiences
- Gen AI Intern - Terra · Internship - May 2025 - Present · 3 mos
- Al Research lntern - HyperCog Technologies · Internship - Sep 2024 - May 2025 · 9 mos
  - ◦ Collaborated with cross functional teams to build AI based solution to detect Continuity errors in videos by leveraging
various methods like Lighting Consistency Analysis, Camera Angle and Framing Detection and Person Pose
Differences.
◦ Developed and fine-tuned AI and object detection models for person detection and person pose estimation to find relative pose
differences to check continuity errors in movies.
◦ Integrated and customized AI models like RTMO (Real-time Method for Multi-person Pose Estimation) and Deep Sort (Simple Online and Realtime Tracking with a Deep Association Metric) into Machine Learning pipelines.

## Education
- Anna University Chennai - Bachelor of Technology, Computational Science - 2020 - 2024

## Skills
- Agno
  - (from) Gen AI Intern at Terra
- C#
  - (from) Gen AI Intern at Terra
- Unity
  - (from) Gen AI Intern at Terra
- Google Cloud Platform (GCP)
  - (from) Gen AI Intern at Terra
- Gen AI
  - (from) Gen AI Intern at Terra
- Large Language Models (LLM)
  - (from) Gen AI Intern at Terra
  - (from) ResumeRAG
- LangChain
  - (from) Gen AI Intern at Terra
  - (from) ResumeRAG
- Retrieval-Augmented Generation (RAG)
  - (from) ResumeRAG
- RAG
  - (from) ResumeRAG
- Streamlit
  - (from) ResumeRAG
- Transformers
  - (from) ResumeRAG
- Attention to Detail
  - (from) Al Research lntern at HyperCog Technologies
- Video Processing
  - (from) Al Research lntern at HyperCog Technologies
- Pattern Recognition
  - (from) Job Salary Prediction for Data Science Professionals
- Model Development
  - (from) Al Research lntern at HyperCog Technologies
- Programming Languages
  - (from) Al Research lntern at HyperCog Technologies
- Artificial Intelligence (AI)
  - (from) 2 experiences across Terra and 1 other company
- Neural Networks
  - (from) Hybrid Recommendation System
- SQL
- Data Science
  - (from) Al Research lntern at HyperCog Technologies

## Honors and Awards
- Stutter Linguistic Assistance Program[IEEE Social Hackathon] - Issued by IEEE · Dec 2022
  - Associated with Anna University Chennai
  - Winners

[Machine Learning,SVM,Active Learning,Python]

Created an AI based app to assist people with childhood onset fluency disorder(Stuttering) which provides synonyms for words that are difficult to pronounce for that specific user.
- Time Based OTP - Issued by Smart India Hackathon · Aug 2022
  - Associated with Anna University Chennai
  - Finalists

[Java,Secure Hash Algorithm,MySQL,Android Studio]

Created an alternative solution to OTP in areas with wear cellular connectivity.
Provides Time based OTP [TOTP] to authenticate a person's identity in absence of network securely.
Created a mobile app for client and service provider.

## Projects
- Hybrid Recommendation System
  - •Designed and implemented a movie and TV show recommendation system using Hybrid
 recommendation model.
•It comines both Neural Collaborative Filtering and Similarity based Content recommendation using weighted average of their predictions.
•Implemented on the MovieLens 100k dataset along with TMDB movie Descriptions.
•Calculated NDCG@10 of 0.69, Precision@k of 0.47, Recall@k of 0.27 for test set.
•Deployed the Hybrid Model using Fast API to recommend top 10 movies to each user and similar movies to new users.
•Also implemented endpoint for A/B testing using T-test statistic using user feedback.
- Job Salary Prediction for Data Science Professionals
  - •Created a Machine Learning based Job Salary predictor for data-related positions sourced from Glassdoor.com.
•Performed data cleaning & feature engineering with NLTK
•Demonstrating EDA results with word cloud map & PCA on TF-IDF word embeddings.
•Utilizing Machine learning on job salary interval prediction with Random Forest, SVM, Logistic
Regression, to select best model with accuracy of 0.71.

## Interests
### Top Voices
- Lex Fridman - 1,704,863 followers - Research Scientist, MIT
- Sebastian Raschka, PhD - 174,789 followers - ML/AI research engineer. Author of Build a Large Language Model From Scratch (amzn.to/4fqvn0D) and Ahead of AI (magazine.sebastianraschka.com), on how LLMs work and the latest developments in the field.
### Companies
- Deloitte - 19,237,532 followers
- Chubb - 1,029,447 followers
### Groups
- Machine Learning Community (Moderated) - 1,989,573 members
- Data science intern - 6,575 members
### Newsletters
- Inside CommerceIQ - Published monthly - Get the latest product updates, event info & featured content from CommerceIQ.
- Ahead of AI - Published monthly - AI and machine learning research, trends, and educational material.
### Schools
- Rajalakshmi Engineering College - 45,243 followers
- Anna University Chennai - 812,961 followers

## Email
- sendhan@letsterra.com

## Location
- Bengaluru, Karnataka, India

## Connections
- 444

## LinkedIn URL
- https://www.linkedin.com/in/sendhan-a-38a48811b/
    """

    # Example job description for demonstration purposes
    example_job_description = """
    **Job Title**: Senior Machine Learning Engineer

**Overview**:
We’re looking for a passionate Senior Machine Learning Engineer to join our innovative team. In this role, you'll use your expertise to build and optimize machine learning models, while working collaboratively across data science, engineering, and product teams to deliver impactful AI-driven solutions. You'll play a critical role in building the next generation of intelligent systems and be involved in the full ML lifecycle, from concept to deployment.

**Key Responsibilities**:
- Design, develop, and implement advanced machine learning models and algorithms for various applications.
- Collaborate with cross-functional teams to integrate machine learning models into existing systems and improve their scalability and performance.        
- Optimize machine learning models to ensure efficient computational performance in production environments.
- Lead efforts in data preprocessing, feature engineering, and model evaluation for improved prediction accuracy.
- Contribute to the development and maintenance of ML infrastructure and pipelines.
- Mentor junior engineers and provide technical guidance to the team.
- Stay current with the latest advancements in ML and incorporate them into projects.
- Participate in code reviews and contribute to best practices in software engineering.

**Required Skills & Qualifications**:
- Extensive experience in machine learning, deep learning, and statistical modeling.
- Strong proficiency in programming languages such as Python and experience with ML frameworks like TensorFlow or PyTorch.
- Understanding of cloud platforms and services such as AWS or Azure.
- Excellent problem-solving abilities and an analytical mindset.
- Experience with data preprocessing, feature extraction, and model automation.
- Bachelor’s or Master’s degree in Computer Science, Engineering, or a related field.
- Proven track record of deploying machine learning solutions at scale.

**Preferred (Nice-to-Have)**:
- Experience in using containerization technologies such as Docker and Kubernetes.
- Familiarity with large-scale data management and processing tools, including Hadoop and Spark.
- Publications or contributions to open-source ML frameworks and participation in academic or industry conferences.
- Doctorate degree in Machine Learning, Computer Vision, or a similarly related field.

    """
    # response = retreive_linkedin_knowledge_chunks("how to improve your linkedin profile")
    user_prompt = f"""
    User Profile:
    {example_user_profile}
    Job Description:
    {example_job_description}
    """
    agent_response = career_counsellor_agent.run_sync(user_prompt)

    print(agent_response.output)   