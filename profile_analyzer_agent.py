from apify_client import ApifyClient
from pprint import pprint
from pydantic_ai import Agent, ImageUrl
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pprint
from pydantic import BaseModel, Field
from typing import Optional
import logfire


logfire.configure()  
logfire.instrument_pydantic_ai()  

load_dotenv()
apify_api_token = os.getenv("APIFY_API_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

system_prompt = """
You are an expert HR strategist. Analyze the following LinkedIn profile and perform a **section-by-section critique**. For each profile section (like Profile Picture, Headline, About, Experience, Skills, Education, Certifications), do the following:

1. **Profile Picture**: Analyze the image for professionalism. Check for a clear headshot, appropriate background, and good lighting.
2. For all other sections, check if the section is **missing**, **empty**, **incomplete**, or **poorly written**.
3. If an issue exists, describe it in detail.
4. Provide a **concrete and actionable suggestion** to improve that section. Rewrites are welcome.
5. If the section is strong, briefly state why (do not elaborate too much).
DO NOT perform job matching or compare to a job description.
Focus only on critique, clarity, professionalism, completeness, tone, and consistency.

Input:
{linkedin_profile_json_here}
"""

class ProfileAnalyzerAgent():
    def __init__(self,openai_api_key:str,apify_api_token:str):
        self.openai_api_key = openai_api_key
        self.apify_api_token = apify_api_token
        self.model = OpenAIModel("gpt-4o",provider=OpenAIProvider(api_key=openai_api_key))
        self.profile_analyzer_agent = Agent(
    name="profile_analyzer_agent",
    system_prompt=system_prompt,
    model=self.model,
    retries=3,
    tools=[scrape_linkedin_profile]
)
# model = OpenAIModel("gpt-4o",provider=OpenAIProvider(api_key=openai_api_key))

# profile_analyzer_agent = Agent(
#     name="profile_analyzer_agent",
#     system_prompt=system_prompt,
#     model=model,
#     retries=3,
    
# )

class ProfileOutput(BaseModel):
    """Structured output for scraped profile data."""
    profile_text: str = Field(..., description="The formatted text of the LinkedIn profile.")
    profile_pic: Optional[ImageUrl] = Field(None, description="The URL of the profile picture.")


# Helper: clean nested dictionaries and create a readable string representation

def format_profile_for_llm(profile: dict) -> str:
    """Convert the extracted profile dictionary into a clean, bullet-pointed
    markdown-like string that an LLM can easily parse.

    Nested keys like `subComponents`, `section_components`, and `breakdown` are
    ignored so the final output remains concise and human-readable.
    """
    lines: list[str] = []

    # Basic details
    full_name = profile.get("fullName") or f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
    if full_name:
        lines.append(f"## Name\n- {full_name}")

    headline = profile.get("headline")
    if headline:
        lines.append(f"\n## Headline\n- {headline}")

    about = profile.get("about")
    if about:
        lines.append(f"\n## About\n{about}")

    # Experiences
    experiences = profile.get("experiences") or []
    if experiences:
        lines.append("\n## Experiences")
        for exp in experiences:
            title = exp.get("title", "").strip()
            subtitle = exp.get("subtitle", "").strip()
            caption = exp.get("caption", "").strip()
            bullet = " - ".join(filter(None, [title, subtitle, caption]))
            if bullet:
                lines.append(f"- {bullet}")
            # include descriptions from subComponents
            for sub in exp.get("subComponents", []):
                for desc in sub.get("description", []):
                    text = desc.get("text") if isinstance(desc, dict) else None
                    if text:
                        lines.append(f"  - {text.strip()}")

    # Education
    educations = profile.get("educations") or []
    if educations:
        lines.append("\n## Education")
        for edu in educations:
            title = edu.get("title", "").strip()
            subtitle = edu.get("subtitle", "").strip()
            caption = edu.get("caption", "").strip()
            bullet = " - ".join(filter(None, [title, subtitle, caption]))
            if bullet:
                lines.append(f"- {bullet}")
            for sub in edu.get("subComponents", []):
                for desc in sub.get("description", []):
                    text = desc.get("text") if isinstance(desc, dict) else None
                    if text:
                        lines.append(f"  - {text.strip()}")

    # Skills
    skills = profile.get("skills") or []
    if skills:
        lines.append("\n## Skills")
        for skill in skills:
            skill_name = skill.get("title")
            if skill_name:
                lines.append(f"- {skill_name}")
            for sub in skill.get("subComponents", []):
                for desc in sub.get("description", []):
                    text = desc.get("text") if isinstance(desc, dict) else None
                    if text:
                        lines.append(f"  - (from) {text.strip()}")

    # Honors & Awards
    awards = profile.get("honorsAndAwards") or []
    if awards:
        lines.append("\n## Honors and Awards")
        for award in awards:
            title = award.get("title", "").strip()
            subtitle = award.get("subtitle", "").strip()
            bullet = " - ".join(filter(None, [title, subtitle]))
            if bullet:
                lines.append(f"- {bullet}")
            for sub in award.get("subComponents", []):
                for desc in sub.get("description", []):
                    text = desc.get("text") if isinstance(desc, dict) else None
                    if text:
                        lines.append(f"  - {text.strip()}")

    # Projects
    projects = profile.get("projects") or []
    if projects:
        lines.append("\n## Projects")
        for proj in projects:
            title = proj.get("title", "").strip()
            if title:
                lines.append(f"- {title}")
            for sub in proj.get("subComponents", []):
                for desc in sub.get("description", []):
                    text = desc.get("text") if isinstance(desc, dict) else None
                    if text:
                        lines.append(f"  - {text.strip()}")

    # Interests
    interests = profile.get("interests") or []
    if interests:
        lines.append("\n## Interests")
        for section in interests:
            section_name = section.get("section_name")
            if section_name:
                lines.append(f"### {section_name}")
            for comp in section.get("section_components", []):
                title = comp.get("titleV2") or comp.get("title") or ""
                caption = comp.get("caption", "")
                subtitle = comp.get("subtitle", "")
                bullet = " - ".join(filter(None, [title.strip(), caption.strip(), subtitle.strip()]))
                if bullet:
                    lines.append(f"- {bullet}")

    # Other simple fields
    simple_map = {
        "Email": profile.get("email"),
        "Location": profile.get("addressWithCountry"),
        "Connections": profile.get("connections"),
        "Languages": ", ".join(profile.get("languages", [])) if profile.get("languages") else None,
        "LinkedIn URL": profile.get("linkedinUrl"),
    }
    for heading, value in simple_map.items():
        if value:
            lines.append(f"\n## {heading}\n- {value}")

    return "\n".join(lines)


def scrape_linkedin_profile(profile_url: str) -> ProfileOutput:
    """
    Scrapes a LinkedIn profile and returns a clean, formatted string ready for
    LLM consumption, along with the profile picture URL.
    Args:
        profile_url (str): The LinkedIn profile URL to scrape.
    Returns:
        ProfileOutput: An object containing the formatted profile text and the profile picture URL.
    """
    apify_api_token = os.getenv("APIFY_API_TOKEN")
    client = ApifyClient(apify_api_token)
    run_input = {"profileUrls": [profile_url]}
    try:
        run = client.actor("dev_fusion/linkedin-profile-scraper").call(run_input=run_input)
        dataset_id = run["defaultDatasetId"]
        items = list(client.dataset(dataset_id).iterate_items())
        if not items:
            raise Exception("No data found for the given profile URL.")
        profile = items[0]
        # Extract only the essential fields
        essential_fields = [
            "about", "fullName", "firstName", "lastName", "headline", "experiences",
            "educations", "skills", "email", "addressWithCountry", "connections",
            "honorsAndAwards", "projects", "interests", "languages", "profilePic", 
            "linkedinUrl"
        ]
        extracted = {field: profile.get(field) for field in essential_fields}
        # Convert to readable string
        formatted_profile = format_profile_for_llm(extracted)

        # Prepare combined output including profile picture
        # Use the first URL from profilePicAllDimensions if available, else fallback to profilePic
        profile_pic_url = None
        profile_pic_all = profile.get("profilePicAllDimensions")
        if profile_pic_all and isinstance(profile_pic_all, list) and len(profile_pic_all) > 0:
            profile_pic_url = profile_pic_all[0].get("url")
        else:
            profile_pic_url = extracted.get("profilePic")
        # print("---------------------------------------------------------",profile_pic_url)
        return ProfileOutput(
            profile_text=formatted_profile,
            profile_pic=ImageUrl(url=profile_pic_url) if profile_pic_url else "No profile picture in the linkedin profile"
        )
    except Exception as e:
        # In case of an error, we can return a ProfileOutput with the error message
        return ProfileOutput(
            profile_text=f"Error scraping LinkedIn profile: {e}",
            profile_pic=None
        )
scrape_linkedin_profile_plain = scrape_linkedin_profile

if __name__ == "__main__":
    profile_url = "https://www.linkedin.com/in/sendhan-a-38a48811b/"
    profile_data = scrape_linkedin_profile(profile_url)
    print("Formatted Profile Text:")
    print(profile_data.profile_text)
    # response = profile_analyzer_agent.run_sync(profile_url)
    # print(response.output)
    # profile_url = "https://www.linkedin.com/in/sendhan-a-38a48811b/"
    # profile_data = scrape_linkedin_profile(profile_url)
    # print("Formatted Profile Text:")
    # print(profile_data.profile_text)
    # print("\nProfile Picture URL:")
    # print(profile_data.profile_pic)