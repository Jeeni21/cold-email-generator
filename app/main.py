import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-44023?from=job%20search%20funnel")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = loader.load().pop().page_content
            print("Loaded Data:", data)  # Debugging statement
            cleaned_data = clean_text(data)
            print("Cleaned Data:", cleaned_data)  # Debugging statement
            portfolio.load_portfolio()
            jobs = ''
            jobs = llm.extract_jobs(cleaned_data)
            print("Extracted Jobs:", jobs)  # Debugging statement
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
