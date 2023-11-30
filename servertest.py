
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI,OpenAIChat
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import TextLoader


llm = OpenAI(temperature=0,max_tokens=4096)
loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

 
embeddings = OpenAIEmbeddings()
docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever())



# Tool search
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool


params = {
"engine": "bing",
"gl": "cn",
"hl": "zh-CN",
}

search = SerpAPIWrapper(serpapi_api_key="fe7e2de72185aa0cc1595f5eb784daac2a3497fe405c668efa612e78cbfaff13",params=params)
search_tool = Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )



from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI


llm = OpenAI(temperature=0)


tools = [
    Tool(
        name = "Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question."
    ),
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
   
]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,return_intermediate_steps=True)

agent.agent.llm_chain.prompt.template ="""Answer the following questions as best you can. You have access to the following tools:

Ruff QA System: useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.
Search: useful for when you need to answer questions about current events

Use the following format,and the format needs to be translated into Chinese :

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Ruff QA System, Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


Begin!

Question: {input}
Thought:{agent_scratchpad}"""



 
llm = OpenAI(temperature=0.9)
prompt1 = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that offer {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt1)



app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)


add_routes(
    app,
    agent,
    path="/agent",




)
add_routes(
    app,
    chain | llm |prompt1,
    path="/product",

)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)



model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)