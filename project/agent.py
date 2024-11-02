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
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

class RAGTool:
    def __init__(self, repo: str):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.repo = repo
        self.github_loader()
        
    def github_loader(self):
        print(f"Loading repository: {self.repo}")
        loader = GithubFileLoader(
            repo=self.repo,
            access_token=os.getenv("GITHUB_ACCESS_TOKEN"),
            github_api_url="https://api.github.com/",
            file_filter=lambda file_path: file_path.endswith(
                (".md", "Dockerfile")
            ),
        )
        
        documents = loader.load()
        text_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="gradient")
        texts = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        
    def query(self, question: str) -> str:
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            results = retriever.invoke(question)
            combined_content = "\n".join([doc.page_content for doc in results])
            return f"Based on the repository content: {combined_content}"
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"

def write_file(input):
    data = json.loads(input)
    try:
        with open(data['file_name'], "w") as f:
            f.write(data['content'])
        return "Written successfully"
    except Exception as e:
        return f"Failed to write: {str(e)}"

def setup_agent(llm):
    rag_tool = RAGTool("jaygupta17/movies_backend_gdg")
    
    prompt = PromptTemplate.from_template("""You are a helpful AI assistant with access to a knowledge base of repository content. Your goal is to provide clear, direct answers based on the repository information.

TOOLS:
------
{tools}

To use a tool, use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have gathered enough information to answer the human's question:
```
Thought: I now have enough information to provide a complete answer
Final Answer: [your response here]
```

Important Guidelines:
1. Use the QueryVectorDatabase tool ONCE to get repository content
2. Only make additional queries if absolutely necessary
3. Synthesize the information into a coherent response
4. Don't repeat queries for the same information

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}""")

    tools = [
        Tool(
            name="QueryVectorDatabase",
            func=rag_tool.query,
            description="Query the repository knowledge base. Use this ONCE to get relevant content. Input should be a specific string query."
        ),
        Tool(
            name="WriteFile",
            func=write_file,
            description="Write content to a file. Input must be a JSON string with 'file_name' and 'content' properties."
        )
    ]
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_execution_time=300,
        max_iterations=5
    )
    return agent_executor
