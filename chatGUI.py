import streamlit as st
import NLP as nlp
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import json

load_dotenv()

# This decorator makes the function only run once.
# The objects returned are stored in Streamlit's cache.
@st.cache_resource
def initNLP():
    intents = nlp.readIntentsCSV()
    xTrainTf, countVect, tfTransformer = nlp.stemmingVectorisationWeighting(intents)
    indexWithNorms = nlp.genInvertedIndex(countVect, xTrainTf)
    return indexWithNorms, countVect, tfTransformer, intents

def getRunningAgents():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DBHOST"),
            user=os.getenv("DBUSER"),
            passwd=os.getenv("DBPASS")
        )    
        cursor = connection.cursor(dictionary=True)

        # Database and table setup.
        cursor.execute("CREATE DATABASE IF NOT EXISTS agentData")
        cursor.execute("USE agentData")
        cursor.execute("CREATE TABLE IF NOT EXISTS activeAgents (id INT AUTO_INCREMENT PRIMARY KEY, mic VARCHAR(10), prefStrategy VARCHAR(100), banned JSON)")
        connection.commit()

        # Query and return the agent data. 
        cursor.execute("SELECT * FROM activeAgents")
        agents = cursor.fetchall()

        cursor.close()
        connection.close()
        return agents
    except Error as e:
        print(f'Error: {e}')
        return []

def startAgent(mic, strategy, banned):
    # TODO: Start agent as a background process.
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DBHOST"),
            user=os.getenv("DBUSER"),
            passwd=os.getenv("DBPASS")
        )    
        cursor = connection.cursor(dictionary=True)
        cursor.execute("USE agentData")
        cursor.execute("INSERT INTO activeAgents (mic, prefStrategy, banned) VALUES (%s,%s,%s)",
                       (mic, strategy, json.dumps(banned)))
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f'Error: {e}')
        return []


# Runs for any GUI refresh event, e.g. initial load, input, UI interaction.
def chatGUI():
    # Loads cached NLP objects (cached due to decorator on initNLP())
    indexWithNorms, countVect, tfTransformer, intents = initNLP()

    # States for navigation.
    if "activeAgentId" not in st.session_state:
        st.session_state.activeAgentId = None

    if not st.session_state.activeAgentId:
        # Dashboard view.

        # Currently running agents list
        st.title("Agents Dashboard")
        st.subheader("Running Agents")
        agents = getRunningAgents()
        if not agents:
            st.write("No running agents.")
        for agent in agents:
            column1, column2 = st.columns([3,1])
            column1.write(f"Agent {agent['id']} | Market: {agent['mic']}")
            if column2.button(f"Chat with Agent {agent['id']}", key=f"btn{agent['id']}"):
                st.session_state.activeAgentId = agent['id']
                st.rerun()

        # New agent form.
        st.divider()
        st.subheader("Start a New Agent")
        with st.form("newAgent"):
            mic = st.selectbox("Market (MIC)", ["XLON", "XNAS", "XHKG", "XJPX"])
            prefStrategy = st.selectbox("Prefered strategy (optional)", ["None","Strat1", "Strat2", "Strat3"])
            banned = st.multiselect("Banned stategies (optional)", ["strat1","strat2","strat3"])
            if st.form_submit_button("Start Agent"):
                startAgent(mic,prefStrategy,banned)
                st.rerun()

    else:
        # Conversational view for a specific agent.

        aId = st.session_state.activeAgentId
        messagesKey = f"messages{aId}"

        if st.sidebar.button("Back to Dashboard"):
            st.session_state.activeAgentId = None
            st.rerun()

        st.title(f"Trading Agent{st.session_state.activeAgentId} Chat")
        # Initialise a chat history.
        if messagesKey not in st.session_state:
            st.session_state[messagesKey] = []
            greeting = "Hi! Shall we get started with trading stocks?"
            st.session_state[messagesKey].append({"role":"ai", "content": greeting})

        # Note: this is just the current session, the chat window will be wiped on refresh (backend data / agent not wiped, just chats.)
        for message in st.session_state[messagesKey]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Take the user's prompt.
        # The ':=' is an operator that assigns the input to the prompt variable and checks if its not None at once.
        if prompt := st.chat_input("Enter your prompt..."):
            with st.chat_message("user"):
                st.markdown(prompt)
        
            # Add user's prompt to the chat history too.
            st.session_state[messagesKey].append({"role": "user", "content": prompt})

            # Search intents against user prompt.
            match = nlp.searchIntent(indexWithNorms, prompt, countVect, tfTransformer)
            if match:
                docId, score = match
                intentLabel = intents[docId][1]
                # Intents supported include:
                # performance, actions, portfolio.
                if intentLabel == "actions":
                    response = "Here is what I've done while you were away..."
                else:
                    response = "Sorry I didn't understand that."
            else:
                response = "Sorry I didn't understand that."

            # Bot response
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state[messagesKey].append({"role": "ai", "content": response})


if __name__ == "__main__":
    chatGUI()