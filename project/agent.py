import json
import os
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import GithubFileLoader
from pydantic import BaseModel , Field
from dotenv import load_dotenv
load_dotenv()

class GithubTool(BaseModel):
    """Load data from repository"""

    file_types: list[str] = Field(..., description="List of file extensions to read")
    repo: str = Field(..., description="Repo name. example: username/reponame")


llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0,
                    api_key = os.getenv("GOOGLE_API_KEY")
                )
class RAGTool:
    def __init__(self, documents_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.setup_vectorstore(documents_path)

    def setup_vectorstore(self, documents_path: str):
        loader = TextLoader(documents_path)
        documents = loader.load()
        text_splitter = SemanticChunker(self.embeddings,breakpoint_threshold_type="gradient")
        texts = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)

    def query(self, question: str) -> str:
        try:
            retriever=self.vectorstore.as_retriever()
            res = retriever.invoke(question)
            print(res)
            return retriever.invoke(question)
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"

def github_loader(input):
  data= json.loads(input)
  repo = data['repo']
  file_types = tuple(data['file_types'])
  if not repo:
      return "Error: Repository path is required"
  loader = GithubFileLoader(
    repo=repo,
    access_token=os.getenv("GITHUB_ACCESS_TOKEN"),
    github_api_url="https://api.github.com/",
    file_filter=lambda file_path: file_path.endswith(
        file_types
    ),
  )
  documents = loader.load()
  text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),breakpoint_threshold_type="gradient")
  texts = [doc for doc in text_splitter.split_documents(documents)]

  return texts

def setup_agent():
    rag_tool = RAGTool("newsdoc.txt")
    prompt = PromptTemplate.from_template("As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\n. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\n.\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you have tool for knowledge base, you MUST use the tool for information:\n\n```\nThought: Do I need to use a RAG tool? Yes\nFinal Answer: [your response here]\n```\n\nUse  conversation history to for context . Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}")

    tools = [
        Tool(
            name="QueryDocuments",
            func=rag_tool.query,
            description="Query the document knowledge base. Input should be a question."
        ),
        Tool(
            name="Github Repository Loader",
            func=github_loader,
            description="Get data from repository. Input should be a dictionary with 'file_types' (list of strings like ['.md', '.ts']) and 'repo' (string like 'username/reponame')",
            arguments_schema = GithubTool
        ),
    ]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def main():
    agent = setup_agent()
    chat_history = ""
    while(True):
      prompt = input("Human:")
      if prompt.lower() == "exit":
        break
      chat_history += f"Human: {prompt}\n"
      print(f"Human: {prompt}")
      res = agent.invoke({"input":prompt,"chat_history":chat_history})
      chat_history += f"AI: {res['output']}\n"
      print("AI: "+res['output'])

main()
