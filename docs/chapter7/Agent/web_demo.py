"""Streamlit Web 入口，启动命令：streamlit run web_demo.py"""

import streamlit as st

from config import create_agent

# 页面配置须放在所有 st 组件之前
st.set_page_config(
    page_title="Happy Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Streamlit 每次交互都会从头重跑本脚本，普通变量无法跨次保留，须用 session_state
if "agent" not in st.session_state:
    # Agent 内部还维护发给 LLM 的完整历史（含 system、tool 等）
    st.session_state.agent = create_agent(verbose=True)
    print("[web_demo] Agent 已初始化")

if "messages" not in st.session_state:
    # 仅用于界面展示的用户/助手消息
    st.session_state.messages = []

agent = st.session_state.agent
print(f"[web_demo] 脚本重跑，界面消息数: {len(st.session_state.messages)}，LLM 上下文消息数: {len(agent.messages)}")

st.title("🤖 Happy Agent")
st.markdown(
    """欢迎来到 Happy Agent web 界面！

在下方输入您的提示，查看 Agent 的实际操作。
"""
)

# 每次重跑都全量重绘历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 无新输入时返回 None，不进入 if 块
prompt = st.chat_input("请输入您的问题...")
if prompt:
    print(f"[web_demo] 用户输入: {prompt}")
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("思考中..."):
        # 内部会调 LLM，必要时执行工具后再生成最终回答
        response = agent.get_completion(prompt)

    print(f"[web_demo] 助手回复: {response}")
    print(f"[web_demo] 本轮结束后 LLM 上下文消息数: {len(agent.messages)}")

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
