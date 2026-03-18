import streamlit as st
import NLP as nlp
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import threading
import time
import uuid
from agent import Agent
from stockExchange import StockExchange
from responseFormatter import ResponseFormatter

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
    return st.session_state.get('activeAgents', {})

@st.cache_resource
def initStockExchange():
    return StockExchange()

def runAgentTradeLoop(accountId, agentData, exchange):
    """
    Background trading loop: at each timestep, agent analyzes all stocks in its market.
    All strategy logic handled within agent.runTimestep().
    """
    mic = agentData.get('mic', 'XNAS')
    timestep_count = 0
    
    print(f"DEBUG: Trade loop started for agent {accountId} on {mic}")
    
    while agentData.get('threadRunning', True):
        simDate = agentData.get('simDate')
        simSpeed = agentData.get('simSpeed', 1)
        
        if not agentData.get('threadActive', True):
            time.sleep(0.5)
            continue
        
        try:
            agent = agentData['agent']
            print(f"DEBUG: Agent {accountId} timestep {timestep_count} on {simDate}")
            agent.runTimestep(exchange, simDate)
            timestep_count += 1
        except Exception as e:
            print(f"DEBUG: Agent {accountId} timestep error - {e}")
        
        sleep_seconds = 2.0 / simSpeed
        time.sleep(sleep_seconds)
    
    print(f"DEBUG: Trade loop ended for agent {accountId}")

def startAgent(mic, prefStrategy, bannedList, simDate=None, simSpeed=1):
    exchange = initStockExchange()
    
    try:
        if 'activeAgents' not in st.session_state:
            st.session_state.activeAgents = {}
        
        accountId = int(time.time() * 1000000) % 1000000000
        
        exchange.initialiseAccount(accountId)
        
        preferred = prefStrategy if prefStrategy and prefStrategy != "None" else None
        if simDate is None:
            simDate = datetime.now().strftime('%Y-%m-%d')
        agent = Agent(
            agentId=accountId,
            accountId=accountId,
            mic=mic,
            preferredStrategy=preferred,
            bannedStrategies=bannedList,
            simDate=simDate
        )
        
        st.session_state.activeAgents[accountId] = {
            'agent': agent,
            'mic': mic,
            'prefStrategy': preferred,
            'banned': bannedList,
            'simDate': simDate,
            'simSpeed': simSpeed,
            'threadRunning': True,
            'threadActive': True
        }
        
        agentData = st.session_state.activeAgents[accountId]
        tradeThread = threading.Thread(
            target=runAgentTradeLoop,
            args=(accountId, agentData, exchange),
            daemon=True
        )
        agentData['thread'] = tradeThread
        tradeThread.start()
        
        print(f"DEBUG: Agent {accountId} started with simDate={simDate}, simSpeed={simSpeed}x, thread running")
        return accountId
    except Exception as e:
        print(f'Error starting agent: {e}')
        return None


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
        for accountId, agentData in agents.items():
            column1, column2 = st.columns([3,1])
            column1.write(f"Agent {accountId} | Market: {agentData['mic']}")
            if column2.button(f"Chat with Agent {accountId}", key=f"btn{accountId}"):
                st.session_state.activeAgentId = accountId
                st.rerun()

        # New agent form.
        st.divider()
        st.subheader("Start a New Agent")
        with st.form("newAgent"):
            mic = st.selectbox("Market (MIC)", ["XLON", "XNAS", "XHKG", "XJPX"])
            prefStrategy = st.selectbox("Preferred strategy (optional)", ["None", "Sentiment", "MeanReversion", "Technical", "Fundamental"])
            banned = st.multiselect("Banned strategies (optional)", ["Sentiment", "MeanReversion", "Technical", "Fundamental"])
            simDate = st.date_input("Simulation start date", value=datetime(2011, 1, 1), min_value=datetime(1990, 1, 1), max_value=datetime.now())
            simSpeed = st.selectbox("Simulation speed", [1, 5, 10, 20], format_func=lambda x: f"{x}x speed")
            if st.form_submit_button("Start Agent"):
                simDateStr = simDate.strftime('%Y-%m-%d')
                accountId = startAgent(mic, prefStrategy, banned, simDateStr, simSpeed)
                if accountId:
                    st.success(f"Agent created with account ID: {accountId} (simDate: {simDateStr}, speed: {simSpeed}x)")
                    st.rerun()
                else:
                    st.error("Failed to create agent")

    else:
        # Conversational view for a specific agent.
        exchange = initStockExchange()
        aId = st.session_state.activeAgentId
        messagesKey = f"messages{aId}"

        if st.sidebar.button("Back to Dashboard"):
            st.session_state.activeAgentId = None
            st.rerun()

        agentData = st.session_state.activeAgents.get(aId)
        simDate = agentData.get('simDate', datetime.now().strftime('%Y-%m-%d')) if agentData else datetime.now().strftime('%Y-%m-%d')
        simSpeed = agentData.get('simSpeed', 1) if agentData else 1
        threadActive = agentData.get('threadActive', True) if agentData else False
        
        st.title(f"Trading Agent {st.session_state.activeAgentId} Chat")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sim Date", simDate)
        with col2:
            st.metric("Sim Speed", f"{simSpeed}x")
        with col3:
            statusText = "ACTIVE" if threadActive else "PAUSED"
            st.metric("Status", statusText)
        with col4:
            if st.button("Advance 1 Day"):
                nextDate = (datetime.strptime(simDate, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                st.session_state.activeAgents[aId]['simDate'] = nextDate
                agentData['agent'].setSimDate(nextDate)
                st.rerun()
        
        controlCol1, controlCol2, controlCol3 = st.columns(3)
        with controlCol1:
            if st.button("Pause" if threadActive else "Resume", key="pauseResumeBtn"):
                st.session_state.activeAgents[aId]['threadActive'] = not threadActive
                st.rerun()
        with controlCol2:
            if st.button("Stop Trading"):
                st.session_state.activeAgents[aId]['threadRunning'] = False
                time.sleep(0.5)
                st.info("Agent trading stopped. Back to dashboard to restart.")
        with controlCol3:
            if st.button("Clear Messages"):
                st.session_state[messagesKey] = []
                st.rerun()
        
        st.divider()
        
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
            response = "Sorry I didn't understand that."
            
            if match:
                docId, score = match
                intentLabel = intents[docId][1]
                agentData = st.session_state.activeAgents.get(aId)
                
                if agentData:
                    agent = agentData['agent']
                    
                    if intentLabel == "performance":
                        metrics = agent.performanceTracker.getAllMetrics()
                        response = ResponseFormatter.formatStrategyPerformance(metrics)
                    
                    elif intentLabel == "actions":
                        recentTrades = agent._executionLog
                        response = ResponseFormatter.formatRecentTrades(recentTrades, limit=10)
                    
                    elif intentLabel == "portfolio":
                        balance = exchange.checkBalance(agent.accountId) or 0
                        portfolio = exchange.checkPortfolio(agent.accountId, agent.simDate)
                        portfolioDict = {}
                        if portfolio is not None and not portfolio.empty:
                            for _, row in portfolio.iterrows():
                                ticker = row.get('ticker')
                                tradeType = row.get('tradeType')
                                if ticker:
                                    if ticker not in portfolioDict:
                                        portfolioDict[ticker] = {}
                                    if tradeType == 'long':
                                        portfolioDict[ticker]['long'] = row.get('quantity', 0)
                                        portfolioDict[ticker]['longEntryPrice'] = row.get('entryPrice', 0)
                                    elif tradeType == 'short':
                                        portfolioDict[ticker]['short'] = row.get('quantity', 0)
                                        portfolioDict[ticker]['shortEntryPrice'] = row.get('priceAtShort', 0)
                        response = ResponseFormatter.formatPortfolioSummary(portfolioDict, balance)

            # Bot response
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state[messagesKey].append({"role": "ai", "content": response})


if __name__ == "__main__":
    chatGUI()