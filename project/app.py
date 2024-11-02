import streamlit as st
from agent import setup_agent,gemini
import time
def ui():
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
    
    current_config = f"{repo_name}_{'-'.join(sorted(selected_types))}"
    if ("agent" not in st.session_state) or (st.session_state.get('current_config') != current_config):
        with st.spinner("Initializing agent..."):
            st.session_state.agent = setup_agent(gemini, repo_name, selected_types)
            st.session_state.current_config = current_config


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])

    user_input = st.chat_input("Enter message..")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        time.sleep(10)        
        with st.spinner("Typing..."):
            res = st.session_state.agent.invoke({"input":user_input,"chat_history":"".join([x["role"]+" "+x["content"]+"," for x in st.session_state.chat_history])})
            st.session_state.chat_history.append({"role":"user","content":user_input})
            with st.chat_message("ai"):
                st.write(res['output'])
            st.session_state.chat_history.append({"role":"ai","content":res['output']})
            

if __name__=="__main__":
    ui()


