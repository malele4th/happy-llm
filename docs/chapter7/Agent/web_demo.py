import streamlit as st

from config import create_agent

st.set_page_config(
    page_title="Happy Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

if "agent" not in st.session_state:
    st.session_state.agent = create_agent(verbose=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent

st.title("🤖 Happy Agent")
st.markdown(
    """欢迎来到 Happy Agent web 界面！

在下方输入您的提示，查看 Agent 的实际操作。
"""
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("我能为您做些什么？"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("思考中..."):
        response = agent.get_completion(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
