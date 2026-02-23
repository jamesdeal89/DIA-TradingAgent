import streamlit as st


def chatGUI():
    st.title("Trading Agent")
    # Initialise a chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = "Hi! Shall we get started with trading stocks?"
        st.session_state.messages.append({"role":"ai", "content": greeting})

    # Note: this is just the current session, the chat window will be wiped on refresh (backend data / agent not wiped, just chats.)
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
    chatGUI = chatGUI()