import streamlit as st
from agent import setup_agent,gemini
import time
from dotenv import load_dotenv
import base64
import requests
from io import BytesIO

load_dotenv()

def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    response = requests.get("https://mermaid.ink/img/" + base64_string)
    return BytesIO(response.content)

def ui():
    st.markdown("""
    ### GitHub Repository Analysis
    Ask questions about the repository content and structure. If you get a timeout:
    - Try breaking your question into smaller parts
    - Be more specific in your questions
    - Avoid asking multiple questions at once
    """)
    st.sidebar.header("Gemini Agent")
    repo_name = st.sidebar.text_input(
        "Enter GitHub Repository",
value="jaygupta17/movies_backend_gdg",
placeholder="username/repository"
    )
    st.sidebar.subheader("Select File Types")
    file_types = {
        "Markdown (.md)": ".md",
        "JavaScript (.js)": ".js",
        "Python (.py)": ".py",
        "TypeScript (.ts)": ".ts",
        "JSON (.json)": ".json",
        "Dockerfile": "Dockerfile",
        "YAML (.yaml, .yml)": ".yaml",
        "TypeScript React (.tsx)": ".tsx",
        "JavaScript React (.jsx)": ".jsx"
    }

    selected_types = []
    for label, extension in file_types.items():
        if st.sidebar.checkbox(label, value=extension == ".md"):
            if extension == ".yaml":
                selected_types.extend([".yaml", ".yml"])
            else:           
                selected_types.append(extension)


    if st.sidebar.button("Apply Changes"):
        st.session_state.selected_types = selected_types
        if "agent" in st.session_state:
            del st.session_state.agent 
        st.rerun()
   

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if repo_name=="jaygupta17/github_agent":
        st.markdown("## Can't access the repo")
    if "agent" not in st.session_state:
        with st.spinner("Initializing agent..."):
            try:
                st.session_state.agent = setup_agent(gemini, repo_name, selected_types)
            except:
                st.markdown("""
                    ## Failed to load repository. Possible reasons:  
                    - Invalid format for repository name
                    - Repository not supported
                    - Repository might be private
                    - Others [try again later]
                    """)

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    user_input = st.chat_input("Enter message..")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        time.sleep(10)        
        with st.spinner("Typing..."):
            try:
                res = st.session_state.agent.invoke({"input":user_input,"chat_history":"".join([x["role"]+" "+x["content"]+"," for x in st.session_state.chat_history])})
                st.session_state.chat_history.append({"role":"user","content":user_input})
                count=0
                while count<3 and res['output'].strip()=="Agent stopped due to iteration limit or time limit.":
                  count=count+1
                  res = st.session_state.agent.invoke({"input":"try again","chat_history":"".join([x["role"]+" "+x["content"]+"," for x in st.session_state.chat_history])})
                  time.sleep(10)        
                if res['output'].strip()=="Agent stopped due to iteration limit or time limit.":
                  with st.chat_message("ai"):
                     st.markdown("Relevant information not found for your query. Include more details or try rephrasing your prompt")
                  st.session_state.chat_history.append({"role":"ai","content":""})
                else:
                  with st.chat_message("ai"):
                     st.markdown(res['output'])
                  st.session_state.chat_history.append({"role":"ai","content":res['output']})
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__=="__main__":
    ui()


