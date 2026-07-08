from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_helper import get_llm

# 1. Initialize the LLM
model = get_llm()

# 2. Define a template with a variable placeholder {topic}
prompt = ChatPromptTemplate.from_template("Tell me a short, witty joke about {title}.")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({"title": "Cats "})

print(response)