from langchain.llms import OpenAI
from dotenv import load_dotenv

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
load_dotenv()

llm = OpenAI(model="text-davinci-003", temperature=0.9)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))