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
import logging
from agent import Agent
from stockExchange import StockExchange
from responseFormatter import ResponseFormatter

logger = logging.getLogger(__name__)

load_dotenv()

@st.cache_resource
def initialiseNLP():
    intents = nlp.readIntentsCSV()
    xTrainTf, countVect, tfTransformer = nlp.stemmingVectorisationWeighting(intents)
    indexWithNorms = nlp.genInvertedIndex(countVect, xTrainTf)
    return indexWithNorms, countVect, tfTransformer, intents

def getRunningAgents():
    return st.session_state.get('activeAgents', {})

def initialiseStockExchange():
    return StockExchange()

def logSimulationFinalResults(accountId: int, agent, exchange, agentData=None, closureDate: str = None):
    """
    Log final simulation results to console using responseFormatter summaries.
    First closes all open positions to get true final P&L.
    
    Args:
        accountId: Account ID
        agent: Agent instance
        exchange: StockExchange instance
        agentData: Optional agent metadata
        closureDate: Optional specific date to use for position closure (defaults to agent.simDate)
    """
    try:
        # Check for open positions BEFORE closing
        print(f"\nChecking for open positions before closing...")
        cursor = exchange.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM portfolios WHERE accountId = %s AND closed IS FALSE",
            (accountId,)
        )
        open_count_before = cursor.fetchone()[0]
        print(f"Open positions before: {open_count_before}")
        
        # Get portfolio value BEFORE closing
        cursor.execute(
            "SELECT snapshotDate, cashBalance, portfolioValue, totalValue FROM portfolio_history WHERE accountId = %s ORDER BY snapshotDate DESC LIMIT 1",
            (accountId,)
        )
        portfolio_before = cursor.fetchone()
        if portfolio_before:
            print(f"Portfolio before close: Cash=${portfolio_before[1]:,.2f}, Holdings=${portfolio_before[2]:,.2f}, Total=${portfolio_before[3]:,.2f}")
        cursor.close()
        
        # Close all open positions to get true final P&L
        closureDate = closureDate or agent.simDate
        print(f"\nClosing all open positions at {closureDate}...")
        positions_closed = exchange.closeAllOpenPositions(agent.accountId, closureDate, agent.agentId)
        print(f"Closed {positions_closed} open position(s)")
        
        # Train LSTM strategy at episode end
        if 'LSTM' in agent.strategies:
            try:
                logger.info(f"Agent {agent.agentId}: Starting LSTM training at episode end")
                agent.strategies['LSTM'].train(batchSize=32, epochs=1)
                logger.info(f"Agent {agent.agentId}: LSTM training completed")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
        
        # Check for REMAINING open positions AFTER closing
        cursor = exchange.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM portfolios WHERE accountId = %s AND closed IS FALSE",
            (accountId,)
        )
        open_count_after = cursor.fetchone()[0]
        print(f"Open positions after: {open_count_after}")
        if open_count_after > 0:
            cursor.execute(
                "SELECT ticker, tradeType, quantity FROM portfolios WHERE accountId = %s AND closed IS FALSE",
                (accountId,)
            )
            remaining = cursor.fetchall()
            print(f"WARNING: Still have unclosed positions: {remaining}")
        cursor.close()
        
        print()
        
        print(f"{'='*80}")
        print(f"SIMULATION FINAL RESULTS - Agent {accountId}")
        print(f"{'='*80}")
        start_date = agentData.get('simDate') if agentData else 'Unknown'
        print(f"Simulation Date Range: {start_date} -> {agent.simDate}")
        print(f"Decision Period: {agent.getDecisionPeriod()} days")
        print(f"Total Timesteps: {agent._timestepCounter}")
        print(f"Total Trades: {agent.totalTrades}")
        print()
        
        # Portfolio summary from history
        try:
            portfolio_history = exchange.getPortfolioHistory(agent.accountId)
            if not portfolio_history.empty:
                print("PORTFOLIO SUMMARY:")
                latest = portfolio_history.iloc[-1]
                print(f"Final Cash Balance: ${latest['cashBalance']:,.2f}")
                print(f"Final Portfolio Value: ${latest['portfolioValue']:,.2f}")
                print(f"Final Total Value: ${latest['totalValue']:,.2f}")
                print()
            else:
                print("PORTFOLIO SUMMARY: No portfolio history available.")
                print()
        except Exception as e:
            print(f"Note: Could not retrieve portfolio summary: {e}")
            print()
        
        # Strategy comparison
        try:
            print("STRATEGY PERFORMANCE:")
            print(ResponseFormatter.formatStrategyComparison(agent.performanceTracker, agent._executionLog))
            print()
        except Exception as e:
            print(f"Note: Could not format strategy comparison: {e}")
            print()
        
        # DEBUG: Print detailed trade history from performance tracker
        try:
            print("[DEBUG] DETAILED TRADE HISTORY FROM PERFORMANCE TRACKER:")
            total_trades_recorded = 0
            if hasattr(agent.performanceTracker, '_tradeHistory'):
                for strategy_name in sorted(agent.performanceTracker._tradeHistory.keys()):
                    trades = agent.performanceTracker._tradeHistory[strategy_name]
                    if trades:
                        total_trades_recorded += len(trades)
                        print(f"\n{strategy_name}: {len(trades)} total trades")
                        for i, trade in enumerate(trades):
                            print(f"  {i+1}. {trade['ticker']:6s} {trade['quantity']:4d}@${trade['entryPrice']:8.2f} -> ${trade['exitPrice']:8.2f} = ${trade['pnl']:10.2f}")
                        total_pnl = sum(t['pnl'] for t in trades)
                        print(f"  TOTAL P&L: ${total_pnl:,.2f}")
            print(f"\nTOTAL TRADES RECORDED: {total_trades_recorded}")
            print(f"TOTAL TRADES PLACED (from log): {agent.totalTrades}")
            print(f"MISSING TRADES: {agent.totalTrades - total_trades_recorded}")
            print()
        except Exception as e:
            print(f"[DEBUG] Could not print trade history: {e}\n")
        
        # HOLD recommendation quality analysis
        try:
            print("HOLD RECOMMENDATION ANALYSIS:")
            recommendationMetrics = {}
            for strategyName in agent.strategies.keys():
                recommendationMetrics[strategyName] = agent.performanceTracker.getRecommendationMetrics(strategyName)
            
            hold_analysis = ResponseFormatter.formatRecommendationQuality(recommendationMetrics)
            print(hold_analysis)
            print()
        except Exception as e:
            print(f"Note: Could not retrieve HOLD analysis: {e}")
            print()
        
        # Recent closed trades - with emphasis on recently closed ones
        try:
            print("RECENTLY CLOSED TRADES AT SIMULATION END:")
            closed_trades = exchange.getClosedTradesPnL(agent.accountId)
            if closed_trades:
                # Sort by date and show last 10
                sorted_trades = sorted(closed_trades, key=lambda x: x.get('exitDate', ''), reverse=True)[:10]
                print(f"Total closed trades in DB: {len(closed_trades)}")
                print(f"Showing last 10 closed trades:")
                for trade in sorted_trades:
                    ticker = trade.get('ticker', 'N/A')
                    trade_type = trade.get('tradeType', 'N/A')
                    entry = trade.get('entryPrice', 0)
                    exit_price = trade.get('exitPrice', 0)
                    pnl = trade.get('pnl', 0)
                    pnl_pct = trade.get('pnlPercent', 0)
                    exit_date = trade.get('exitDate', 'N/A')
                    print(f"  {ticker:6s} {trade_type:6s} | Entry ${entry:7.2f} -> Exit ${exit_price:7.2f} | P&L ${pnl:8.2f} ({pnl_pct:6.1f}%) | {exit_date}")
                print()
            else:
                print("No closed trades recorded.")
                print()
        except Exception as e:
            print(f"Note: Could not retrieve closed trades: {e}")
            print()
        
        print(f"{'='*80}")
        print(f"Simulation complete. Results saved to database.")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error logging final results: {e}")

def runAgentTradeLoop(accountId, agentData, exchange):
    """
    Background trading loop: at each timestep, agent analyses all stocks in its market.
    All strategy logic handled within agent.runTimestep().
    """
    mic = agentData.get('mic', 'XNAS')
    timestep_count = 0
    
    print(f"DEBUG: Trade loop started for agent {accountId} on {mic}")
    
    # Get dataset max date once at startup
    maxNewsDate = exchange.getMaxNewsDate()
    todayDate = datetime.now().strftime('%Y-%m-%d')
    
    while agentData.get('threadRunning', True):
        isRealtimeMode = agentData.get('isRealtimeMode', False)
        
        if not agentData.get('threadActive', True):
            time.sleep(0.5)
            continue
        
        try:
            agent = agentData['agent']
            simDate = agent.simDate
            
            # Check simulation boundaries
            if isRealtimeMode:
                # Realtime mode: only simulate on current date
                currentRealDate = datetime.now().strftime('%Y-%m-%d')
                if simDate != currentRealDate:
                    agent.setSimDate(currentRealDate)
                    simDate = currentRealDate
            else:
                # Check if we've reached the configured end date (if set), max news date, or today
                hasReachedLimit = False
                lastValidSimDate = simDate
                endDate = agent.getEndDate()
                currentDate = datetime.strptime(simDate, '%Y-%m-%d')
                decisionPeriod = agent.getDecisionPeriod()
                nextDate = (currentDate + timedelta(days=decisionPeriod)).strftime('%Y-%m-%d')
                nextDateTime = datetime.strptime(nextDate, '%Y-%m-%d')
                
                # Check if NEXT timestep would exceed boundaries - if so, this is the last one
                willExceedBoundary = False
                if endDate:
                    endDateTime = datetime.strptime(endDate, '%Y-%m-%d')
                    if nextDateTime > endDateTime:
                        willExceedBoundary = True
                elif maxNewsDate and nextDate > maxNewsDate:
                    willExceedBoundary = True
                elif nextDate >= todayDate:
                    willExceedBoundary = True
                
                # If already past boundary, stop immediately
                if endDate:
                    endDateTime = datetime.strptime(endDate, '%Y-%m-%d')
                    if currentDate > endDateTime:
                        print(f"DEBUG: Simulation reached configured end date ({endDate}). Stopping agent.")
                        hasReachedLimit = True
                elif maxNewsDate and simDate > maxNewsDate:
                    print(f"DEBUG: Simulation reached max news date ({maxNewsDate}). Stopping agent.")
                    hasReachedLimit = True
                elif simDate > todayDate:
                    print(f"DEBUG: Simulation reached today's date ({todayDate}). Stopping agent.")
                    hasReachedLimit = True
                
                if hasReachedLimit:
                    agentData['threadActive'] = False
                    agentData['threadRunning'] = False
                    print(f"DEBUG: Agent {accountId} simulation ended at {simDate}.")
                    logSimulationFinalResults(accountId, agent, exchange, agentData, lastValidSimDate)
                    break
            
            print(f"DEBUG: Agent {accountId} timestep {timestep_count} on {simDate}")
            agent.runTimestep(exchange, simDate)
            
            # Auto-advance date (skip in realtime mode)
            if not isRealtimeMode:
                # Only advance if next date won't exceed end date
                decisionPeriod = agent.getDecisionPeriod()
                nextDate = (datetime.strptime(simDate, '%Y-%m-%d') + timedelta(days=decisionPeriod)).strftime('%Y-%m-%d')
                
                shouldAdvance = True
                endDate = agent.getEndDate()
                if endDate:
                    endDateTime = datetime.strptime(endDate, '%Y-%m-%d')
                    nextDateTime = datetime.strptime(nextDate, '%Y-%m-%d')
                    if nextDateTime > endDateTime:
                        shouldAdvance = False
                        print(f"DEBUG: Next timestep ({nextDate}) would exceed end date ({endDate}). Stopping after this timestep.")
                        agentData['threadActive'] = False
                        agentData['threadRunning'] = False
                        # Log final results when end date reached
                        logSimulationFinalResults(accountId, agent, exchange, agentData, simDate)
                
                if shouldAdvance:
                    agent.setSimDate(nextDate)
            
            timestep_count += 1
        except Exception as e:
            print(f"DEBUG: Agent {accountId} timestep error - {e}")
        
        # Adjust sleep based on mode
        if isRealtimeMode:
            sleep_seconds = 5.0  # Check every 5 seconds in realtime mode
        else:
            sleep_seconds = 2.0  # Check every 2 seconds in historical mode
        time.sleep(sleep_seconds)
    
    print(f"DEBUG: Trade loop ended for agent {accountId}")

def startAgent(mic, prefStrategy, bannedList, simDate=None, endDate=None, decisionPeriod=1, isRealtimeMode=False):
    exchange = initialiseStockExchange()
    
    try:
        if 'activeAgents' not in st.session_state:
            st.session_state.activeAgents = {}
        
        accountId = int(time.time() * 1000000) % 1000000000
        
        exchange.initialiseAccount(accountId)
        
        preferred = prefStrategy if prefStrategy and prefStrategy != "None" else None
        if simDate is None:
            simDate = datetime.now().strftime('%Y-%m-%d')
        if isRealtimeMode:
            simDate = datetime.now().strftime('%Y-%m-%d')
        
        agent = Agent(
            agentId=accountId,
            accountId=accountId,
            mic=mic,
            preferredStrategy=preferred,
            bannedStrategies=bannedList,
            simDate=simDate,
            endDate=endDate,
            decisionPeriod=decisionPeriod
        )
        
        # Pass agent's performanceTracker to exchange so closed trades get recorded
        exchange.performanceTracker = agent.performanceTracker
        
        st.session_state.activeAgents[accountId] = {
            'agent': agent,
            'mic': mic,
            'prefStrategy': preferred,
            'banned': bannedList,
            'simDate': simDate,
            'endDate': endDate,
            'decisionPeriod': decisionPeriod,
            'isRealtimeMode': isRealtimeMode,
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
        
        print(f"DEBUG: Agent {accountId} started with simDate={simDate}, decision period={decisionPeriod}d, isRealtime={isRealtimeMode}, thread running")
        return accountId
    except Exception as e:
        print(f'Error starting agent: {e}')
        return None


def chatGUI():
    indexWithNorms, countVect, tfTransformer, intents = initialiseNLP()
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
            decisionPeriod = agentData.get('decisionPeriod', 1)
            isRealtimeMode = agentData.get('isRealtimeMode', False)
            modeStr = "realtime" if isRealtimeMode else f"historical"
            column1, column2 = st.columns([3,1])
            column1.write(f"Agent {accountId} | Market: {agentData['mic']} | Mode: {modeStr} | Decision Period: {decisionPeriod}d")
            if column2.button(f"Chat with Agent {accountId}", key=f"btn{accountId}"):
                st.session_state.activeAgentId = accountId
                st.rerun()

        # New agent form.
        st.divider()
        st.subheader("Start a New Agent")
        with st.form("newAgent"):
            mic = st.selectbox("Market (MIC)", ["XLON", "XNAS", "XHKG", "XJPX"], index=1)
            prefStrategy = st.selectbox("Preferred strategy (optional)", ["None", "Sentiment", "MeanReversion", "Technical", "Fundamental", "DeepQL", "LSTM"])
            banned = st.multiselect("Banned strategies (optional)", ["Sentiment", "MeanReversion", "Technical", "Fundamental", "DeepQL", "LSTM"])
            
            # Simulation mode selection
            simMode = st.radio("Simulation mode", ["Historical (backtest)", "Realtime (live)"], horizontal=True)
            isRealtimeMode = (simMode == "Realtime (live)")
            
            if isRealtimeMode:
                st.info("Realtime mode: Agent uses today's date and trades on current stock data. Start date is not applicable.")
                simDate = datetime.now()
                endDate = None
            else:
                with st.container(border=True):
                    st.caption("Historical Simulation Settings")
                    col_start, col_end = st.columns(2)
                    with col_start:
                        simDate = st.date_input("Start date", value=datetime(2011, 1, 1), min_value=datetime(1990, 1, 1), max_value=datetime.now(), key="sim_start")
                    with col_end:
                        endDate = st.date_input("End date (optional)", value=datetime(2011, 12, 31), min_value=datetime(1990, 1, 1), max_value=datetime.now(), key="sim_end")
            
            decisionPeriod = st.slider("Decision period (days)", min_value=1, max_value=60, value=1, step=1, help="Make trading decisions every N days instead of daily. Helps reduce noise by analyzing larger time windows.")
            
            if st.form_submit_button("Start Agent"):
                simDateStr = simDate.strftime('%Y-%m-%d') if not isinstance(simDate, str) else simDate
                endDateStr = endDate.strftime('%Y-%m-%d') if endDate and not isinstance(endDate, str) else (endDate if endDate else None)
                accountId = startAgent(mic, prefStrategy, banned, simDateStr, endDateStr, decisionPeriod, isRealtimeMode)
                if accountId:
                    mode_display = "realtime" if isRealtimeMode else "historical"
                    st.success(f"Agent created with account ID: {accountId} (mode: {mode_display}, decision period: {decisionPeriod}d)")
                    st.rerun()
                else:
                    st.error("Failed to create agent")

    else:
        # Conversational view for a specific agent.
        exchange = initialiseStockExchange()
        aId = st.session_state.activeAgentId
        messagesKey = f"messages{aId}"

        if st.sidebar.button("Back to Dashboard"):
            st.session_state.activeAgentId = None
            st.rerun()

        agentData = st.session_state.activeAgents.get(aId)
        # Read simDate directly from agent object to always stay in sync
        simDate = agentData['agent'].simDate if agentData and 'agent' in agentData else datetime.now().strftime('%Y-%m-%d')
        threadActive = agentData.get('threadActive', True) if agentData else False
        
        st.title(f"Trading Agent {st.session_state.activeAgentId} Chat")

        # Get current decision period info
        currentDecisionPeriod = agentData['agent'].getDecisionPeriod() if agentData and 'agent' in agentData else 1
        isRealtimeMode = agentData.get('isRealtimeMode', False) if agentData else False
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Mode")
            if isRealtimeMode:
                st.write("REALTIME")
            else:
                st.write("HISTORICAL")
        with col2:
            st.caption("Decision Period")
            st.write(f"{currentDecisionPeriod} days")
        with col3:
            st.caption("Status")
            statusText = "ACTIVE" if threadActive else "PAUSED"
            st.write(statusText)
        
        # Adjust decision period slider
        st.write("**Adjust Decision Period:**")
        newDecisionPeriod = st.slider("Decision period (days)", min_value=1, max_value=60, value=currentDecisionPeriod, step=1, key="dp_slider", help="Update the decision period. Agent will make trading decisions every N days.")
        if newDecisionPeriod != currentDecisionPeriod:
            agentData['agent'].setDecisionPeriod(newDecisionPeriod)
            st.session_state.activeAgents[aId]['decisionPeriod'] = newDecisionPeriod
            st.success(f"Decision period updated to {newDecisionPeriod} days")
        
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
            greeting = "Ready to start trading."
            st.session_state[messagesKey].append({"role":"ai", "content": greeting})

        # Note: this is just the current session, the chat window will be wiped on refresh (backend data / agent not wiped, just chats.)
        for message in st.session_state[messagesKey]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Re-render performance charts if this was a performance response
                if message.get("type") == "performance" and message["role"] == "ai":
                    agentData = st.session_state.activeAgents.get(aId)
                    if agentData:
                        agent = agentData['agent']
                        chart_data = ResponseFormatter.formatPerformanceChartData(
                            agent.performanceTracker, exchange, agent.accountId
                        )
                        
                        if chart_data['hasData']:
                            # Portfolio value over time
                            if chart_data['portfolioValue']:
                                import pandas as pd
                                portfolio_df = pd.DataFrame([
                                    {'Date': k, 'Portfolio Value': v}
                                    for k, v in chart_data['portfolioValue'].items()
                                ])
                                st.line_chart(portfolio_df.set_index('Date'))
                                st.caption("Portfolio Value Over Time")
                            
                            # Strategy-specific P&L
                            if chart_data['strategyPnL']:
                                st.markdown("#### Strategy-Specific Cumulative P&L")
                                import pandas as pd
                                for strategy, pnl_dict in chart_data['strategyPnL'].items():
                                    strategy_df = pd.DataFrame([
                                        {'Date': k, 'Cumulative P&L': v}
                                        for k, v in pnl_dict.items()
                                    ])
                                    st.line_chart(strategy_df.set_index('Date'))
                                    st.caption(f"{strategy}")

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
                        # Display portfolio value and performance charts
                        st.markdown("### Performance Metrics")
                        
                        # Get chart data
                        chart_data = ResponseFormatter.formatPerformanceChartData(
                            agent.performanceTracker, exchange, agent.accountId
                        )
                        
                        # Store performance display info in session state
                        performance_summary = f"Performance data retrieved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        st.session_state[messagesKey].append({"role": "ai", "content": performance_summary, "type": "performance"})
                        
                        if chart_data['hasData']:
                            # Portfolio value over time
                            if chart_data['portfolioValue']:
                                import pandas as pd
                                portfolio_df = pd.DataFrame([
                                    {'Date': k, 'Portfolio Value': v}
                                    for k, v in chart_data['portfolioValue'].items()
                                ])
                                st.line_chart(portfolio_df.set_index('Date'))
                                st.caption("Portfolio Value Over Time")
                            
                            # Strategy-specific P&L
                            if chart_data['strategyPnL']:
                                st.markdown("#### Strategy-Specific Cumulative P&L")
                                import pandas as pd
                                for strategy, pnl_dict in chart_data['strategyPnL'].items():
                                    strategy_df = pd.DataFrame([
                                        {'Date': k, 'Cumulative P&L': v}
                                        for k, v in pnl_dict.items()
                                    ])
                                    st.line_chart(strategy_df.set_index('Date'))
                                    st.caption(f"{strategy}")
                        else:
                            st.info("No performance data yet. Run the simulation to generate trading results.")
                    
                    elif intentLabel == "strategy":
                        # Display strategy ranking and comparison
                        response = ResponseFormatter.formatStrategyComparison(agent.performanceTracker, agent._executionLog)
                        
                        # Display in chat
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state[messagesKey].append({"role": "ai", "content": response})
                    
                    elif intentLabel == "actions":
                        recentActions = agent.performanceTracker.getAllRecommendations(limit=20)
                        response = ResponseFormatter.formatRecentActions(recentActions, limit=20)
                        
                        # Display response
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state[messagesKey].append({"role": "ai", "content": response})
                    
                    elif intentLabel == "portfolio":
                        balance = exchange.checkBalance(agent.accountId) or 0
                        portfolio = exchange.checkPortfolio(agent.accountId, agent.simDate)
                        portfolioDict = {}
                        if portfolio is not None and not portfolio.empty:
                            for _, row in portfolio.iterrows():
                                ticker = row.get('ticker')
                                tradeType = row.get('tradeType')
                                mic = row.get('mic', agent.mic)  # Get MIC from portfolio or use agent's MIC
                                if ticker:
                                    if ticker not in portfolioDict:
                                        portfolioDict[ticker] = {'mic': mic}
                                    if tradeType == 'long':
                                        entry_price = row.get('entryPrice', 0)
                                        qty = row.get('quantity', 0)
                                        # Get current price
                                        try:
                                            current_price = exchange.getPrice(ticker, mic, agent.simDate)
                                        except ValueError:
                                            current_price = entry_price  # Fall back to entry price if error
                                        portfolioDict[ticker]['long'] = qty
                                        portfolioDict[ticker]['longEntryPrice'] = entry_price
                                        portfolioDict[ticker]['longCurrentPrice'] = current_price
                                    elif tradeType == 'short':
                                        entry_price = row.get('priceAtShort', 0)
                                        qty = row.get('quantity', 0)
                                        # Get current price
                                        try:
                                            current_price = exchange.getPrice(ticker, mic, agent.simDate)
                                        except ValueError:
                                            current_price = entry_price  # Fall back to entry price if error
                                        portfolioDict[ticker]['short'] = qty
                                        portfolioDict[ticker]['shortEntryPrice'] = entry_price
                                        portfolioDict[ticker]['shortCurrentPrice'] = current_price
                        response = ResponseFormatter.formatPortfolioSummary(portfolioDict, balance)
                        
                        # Display response
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state[messagesKey].append({"role": "ai", "content": response})
                    
                    elif intentLabel == "recommendation":
                        # Display recommendation quality metrics - were holds accurate?
                        recommendationMetrics = {}
                        for strategyName in agent.strategies.keys():
                            recommendationMetrics[strategyName] = agent.performanceTracker.getRecommendationMetrics(strategyName)
                        
                        response = ResponseFormatter.formatRecommendationQuality(recommendationMetrics)
                        
                        # Display in chat
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state[messagesKey].append({"role": "ai", "content": response})
                    
                    else:
                        # Fallback response for unmatched intents
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state[messagesKey].append({"role": "ai", "content": response})
            
            else:
                # No agent found
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state[messagesKey].append({"role": "ai", "content": response})
            
            if not match:
                # No intent match
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state[messagesKey].append({"role": "ai", "content": response})


if __name__ == "__main__":
    chatGUI()