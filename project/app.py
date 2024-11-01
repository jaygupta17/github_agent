import streamlit as st
from agent import setup_agent,gemini,groq
import time
def ui():
    st.sidebar.header("Gemini Agent")

    if "agent" not in st.session_state:
        with st.spinner("Thinking"):
            st.session_state.agent = setup_agent(gemini)


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


