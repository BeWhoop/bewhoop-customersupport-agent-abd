import streamlit as st
import requests

# ========================
# STREAMLIT CONFIG
# ========================
st.set_page_config(page_title="BeWhoop Support Assistant", page_icon="💬", layout="wide")

st.title("💬 BeWhoop Support Assistant")
st.caption("Ask your questions about BeWhoop services and get instant support!")

API_BASE = "http://127.0.0.1:8000"  # change to deployed FastAPI URL in production

# ========================
# SESSION STATE
# ========================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "history" not in st.session_state:
    st.session_state["history"] = []

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.image("https://www.linkedin.com/company/bewhoopapp?originalSubdomain=pk", use_container_width=True)
    st.markdown("### 🗂️ Chat History")
    if st.session_state["history"]:
        for i, h in enumerate(st.session_state["history"]):
            st.write(f"**{i+1}.** {h}")
    else:
        st.caption("No previous chats yet.")
    st.markdown("---")
    st.markdown("### 💡 Feedback")
    feedback = st.text_area("Your feedback", placeholder="Type your feedback here...")
    if st.button("Submit Feedback"):
        st.success("✅ Thanks! Your feedback has been recorded.")

# ========================
# CHAT INTERFACE
# ========================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your query here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.session_state["history"].append(prompt)

    try:
        # Call FastAPI backend
        response = requests.post(f"{API_BASE}/ask", json={"question": prompt})
        data = response.json()

        if "answer" in data:
            reply = data["answer"]
        else:
            reply = f"⚠️ Error: {data.get('error', 'Unknown error')}"

    except Exception as e:
        reply = f"❌ Failed to connect to API: {e}"

    # Show assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
