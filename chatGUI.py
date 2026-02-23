import streamlit as st

class ChatGUI:
    def __init__(self):
        st.title("Trading Agent")
        # Initialise a chat history.
        if "messages" not in st.session_state:
            st.session_state.messages = []
            greeting = "Hi! Shall we get started with trading stocks?"
            st.session_state.messages.append({"role":"ai", "content": greeting})

        # When the user returns, still display their chat history.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Take the user's prompt.
        # The ':=' is an operator that assigns the input to the prompt variable and checks if its not None at once.
        if prompt := st.chat_input("Enter your prompt..."):
            with st.chat_message("user"):
                st.markdown(prompt)
        
            # Add user's prompt to the chat history too.
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Bot response
            with st.chat_message("assistant"):
                st.markdown("OK.")
            st.session_state.messages.append({"role": "ai", "content": "OK."})

if __name__ == "__main__":
    chatGUI = ChatGUI()