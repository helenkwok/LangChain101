from langchain.llms import OpenAI

from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="text-davinci-003", temperature=0)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

# initialize tool object for performing Google searches
tools = [
  Tool(
    name = "google-search",
    func=search.run,
    description="useful for when you need to search google to answer questions about current events"
  )
]

# create an agent that uses our Google Search tool
agent = initialize_agent(
  tools,
  llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  max_iterations=6
)

# check out the response
response = agent("What's the latest news about the Mars rover?")
print(response['output'])