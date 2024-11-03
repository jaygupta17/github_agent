import json
import os
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import GithubFileLoader
from dotenv import load_dotenv

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    timeout=60,
    max_output_tokens=4000
)

class RAGTool:     
    LANGUAGE_CONFIGS = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".java": Language.JAVA,
        ".cpp": Language.CPP,
        ".c": Language.CPP,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".php": Language.PHP,
        ".rb": Language.RUBY,
        ".cs": Language.CSHARP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".html": Language.HTML,
        ".md": Language.MARKDOWN,
        ".sol": Language.SOL,
    }

    def __init__(self, repo: str,file_types: list = None):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.repo = repo
        self.summary=""
        self.combined_content=""
        self.file_types = file_types or [".md"] 
        self.github_loader()
        
    def get_text_splitter(self, file_extension: str):
        language = self.LANGUAGE_CONFIGS.get(file_extension)
        if language:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )
        
        return RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_document(self, doc):
        """
        Process a single document with appropriate text splitter
        """
        file_extension = os.path.splitext(doc.metadata.get('source', ''))[1].lower()
        splitter = self.get_text_splitter(file_extension)
        
        if isinstance(splitter, MarkdownHeaderTextSplitter):
            splits = splitter.split_text(doc.page_content)
        else:
            splits = splitter.split_text(doc.page_content)
            
        return [
            Document(
                page_content=split if isinstance(split, str) else split.page_content,
                metadata={
                    **doc.metadata,
                    'chunk_index': idx
                }
            )
            for idx, split in enumerate(splits)
        ]
    
    def github_loader(self):
        print(f"Loading repository: {self.repo}")
        loader = GithubFileLoader(
            repo=self.repo,
            access_token=os.getenv("GITHUB_ACCESS_TOKEN"),
            github_api_url="https://api.github.com/",
            file_filter=lambda file_path: file_path.endswith(
                tuple(self.file_types)
            ),
        )
        documents = loader.load()
        processed_docs = []

        for doc in documents:
            if doc.metadata['path'].startswith("main/package-lock.json"):
                pass
            else:
                processed_docs.extend(self.process_document(doc)) 

        self.combined_content = "\n".join([f"""Path:{doc.metadata['path'] or "Unknown file"}; Content:{doc.page_content or ""}; Chunk-Index:{doc.metadata['chunk_index'] or 0}; Source:{doc.metadata['source']}""" for doc in processed_docs])
        self.vectorstore = FAISS.from_documents(processed_docs, self.embeddings)
        self.summary=gemini.invoke(f"You are a RAG assistant. summarize the following repository content for efficient chat with repository use case. Repo Content:\n{self.combined_content}")
        print("Loaded.")
        
    def query(self, question: str) -> str:
        try:
            retriever = self.vectorstore.as_retriever()
            results = retriever.invoke(question)
            for doc in results:
                print(doc)
            combined_content = "\n".join([f"""Path:{doc.metadata['path'] or "Unknown file"}; Content:{doc.page_content or ""}; Chunk-Index:{doc.metadata['chunk_index'] or 0}; Source:{doc.metadata['source']}""" for doc in results])
            
            return f"Based on the repository content: {combined_content}"
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"

    def get_summary(self,input:str="")->str:
            if input:
                    self.summary = gemini.invoke(f"You are a RAG assistant. summarize the following repository content for efficient chat with repository use case. User Input:{input} \n Repo Content:\n{self.combined_content}")
            if not self.summary:
                    self.summary = gemini.invoke(f"You are a RAG assistant. summarize the following repository content for efficient chat with repository use case. Repo Content:\n{self.combined_content}")
            return self.summary.content


def setup_agent(llm,repo_name="jaygupta17/movies_backend_gdg",file_types: list = None):
    rag_tool = RAGTool(repo_name,file_types)
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
[Give final answer as output]
```
                                          
When you dont need tool , use the following format:
```  
Thought: Do I need to use a tool? No
Final Answer: [your response here]
[Give final answer as output]
```
                                                                                                                         
Important Guidelines:
1. Use the QueryVectorDatabase tool only once to get repository content.
2. Only make additional queries if absolutely necessary
3. Synthesize the information into a coherent response
4. Don't repeat queries for the same information
5. Use RepositorySummary tool only once to get repository summary.
6. For RepositorySummary tool
, always provide up to date summary as final answer           


When you have gathered enough information to answer the human's question:
```
Thought: I now have enough information to provide a complete answer
Final Answer: [your response here]
```
                                                                                                           
Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}""")

    tools = [
        Tool(
            name="QueryVectorDatabase",
            func=rag_tool.query,
            description="Query the repository knowledge base. Use this to get relevant content from repository. Input should be a specific string query."
        ),
        Tool(
            name="RepositorySummary",
            func=rag_tool.get_summary,
            description="This tool is helpful when you need to know about repository. this tool is for summarising the whole repository content.",
        )
    ]
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_execution_time=600,
        early_stopping_method="generate",
        max_iterations=8
    )
    return agent_executor
