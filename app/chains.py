import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
                            model_name="llama-3.1-70b-versatile",
                            groq_api_key= os.getenv("GROQ_API_KEY"),
                            temperature=0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=2,
                            # other params...
                        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WESITE:
                {page_data}
                ### INSTRUCTION
                The scraped text is from the career's page of a website.
                Your job is to extract job postings and return them in JSON format containing 
                following keys: 'role', 'experience', 'skills' and 'description'.
                Only return the valid JSON.
                ### VALID JSON (NO PREAMBLE):
                """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]


    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
             """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            I am a Software Development student at Sheridan College, currently in my third year. I have developed many projects and have experience from two co-op terms. Over my experience, I have empowered numerous enterprises with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency.

            Could you please help me write a cold email to a manager regarding a job opportunity? 
            The email should describe my capabilities in fulfilling their needs and include the most relevant projects from the following links to showcase my portfolio: {link_list}. Remember, my name is Jeenitesh.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))