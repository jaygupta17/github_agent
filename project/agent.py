import json
import base64
import os
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import GithubFileLoader
from pydantic import BaseModel , Field
from dotenv import load_dotenv

load_dotenv()


gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

class GithubTool(BaseModel):
    """Load data from repository"""
    file_types: list[str] = Field(..., description="List of file extensions to read")
    repo: str = Field(..., description="Repo name. example: username/reponame")


class RAGTool:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
    def github_loader(self,input):
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

def write_file(input):
   data = json.loads(input)
   try:
    with open(data['file_name'],"w") as f:
        f.write(data['content'])
        print("Done")
        return "written successfully"
   except:
    print("Error writing content")
    return  "Failed to write"

def setup_agent(llm):
    rag_tool = RAGTool()
    prompt = PromptTemplate.from_template("As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\n. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\n.\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nUse action and action input fotmat only when you are using a  tool, otherwise give direct response. When you have a response to say to the Human, or if you have tool for knowledge base, you MUST use the tool for information:\n\n```\nThought: Do I need to use a RAG tool? Yes\nFinal Answer: [your response here]\n```\n\nUse  conversation history to for context . final Output should be in markdown form Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}")

    tools = [
        Tool(
            name="QueryVectorDatabase",
            func=rag_tool.query,
            description="Query the knowledge base. Input should be a string query."
        ),
        Tool(
            name="Github Repository Loader",
            func=rag_tool.github_loader,
            description="Get data from repository and save to database for retreival. Input should be a valid json string with 'file_types' (list of strings like ['.md', '.ts']) and 'repo' (string like 'username/reponame')",
            arguments_schema = GithubTool
        ),
        Tool(
            name="Write file tool",
            func=write_file,
            description="Write the text content to the given file_name. Input should be a valid json string with 'file_name' (string: name of file) and 'content'(string: content to write in file. this string should not break the code because of escape characters , etc. pass the content accordingly)"
        )
    ]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)
    return agent_executor
