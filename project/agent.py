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

class RAGTool:
    def __init__(self , repo:str):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.repo= repo
        self.github_loader()
    def github_loader(self):
        print(self.repo)
        loader = GithubFileLoader(
            repo=self.repo,
            access_token=os.getenv("GITHUB_ACCESS_TOKEN"),
            github_api_url="https://api.github.com/",
            file_filter=lambda file_path: file_path.endswith(
              (".md" , "Dockerfile")
            ),
        )
        # ('.md' , '.py','.jsx' ,'.json' , '.js' , 'Dockerfile','.yaml','.tsx','.ts')
        
        documents = loader.load()
        text_splitter = SemanticChunker(self.embeddings,breakpoint_threshold_type="gradient")
        texts = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
    def query(self, question: str) -> str:
        try:
            retriever=self.vectorstore.as_retriever()
            res = retriever.invoke(question)
            print(res)
            return f"Query result:{res}"
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
    rag_tool = RAGTool("jaygupta17/movies_backend_gdg")
    prompt = PromptTemplate.from_template("As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\n. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\n.\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\n. When you have a response to say to the Human, or if you have tool for knowledge base, you MUST use the tool for information:\n\n```\nThought: Do I need to use a RAG tool? Yes\nFinal Answer: [your response here]\n```\n\nUse  conversation history to for context . Make sure action input is a valid function argument for tool.  Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}")

    tools = [
        Tool(
            name="QueryVectorDatabase",
            func=rag_tool.query,
            description="Query the knowledge base.It returns query results from knowledge base. It includes github repo data. Input should be a string query."
        ),
        Tool(
            name="Write file tool",
            func=write_file,
            description="Write the text content to the given file_name. Input should be a valid dictionary with 'file_name' and 'content' as property with string values. Input should be valid to directly pass into the function argument. pass the content accordingly"
        )
    ]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True, max_execution_time=20)
    return agent_executor
