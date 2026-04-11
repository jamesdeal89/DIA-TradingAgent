"""
Response formatter for converting agent metrics and data into GUI-friendly text.
Formats agent performance, portfolio, trades, and strategy data for display.
"""

from typing import Dict, List, Any
from datetime import datetime


class ResponseFormatter:
    """Formats agent data into human-readable responses for the chat interface."""
    
    @staticmethod
    def formatPortfolioSummary(portfolio: Dict[str, Any], balance: float) -> str:
        """
        Format portfolio holdings as markdown table with current prices and P&L.
        
        Args:
            portfolio: Dict with held positions {ticker: {long/short, longEntryPrice, longCurrentPrice, etc}}
            balance: Current account balance
        
        Returns:
            Formatted portfolio summary string (markdown table)
        """
        if not portfolio:
            return f"Your portfolio is empty. Current balance: ${balance:,.2f}"
        
        # Calculate total value of long holdings (excluding shorts)
        total_holdings_value = 0.0
        for ticker, positions in portfolio.items():
            long_qty = positions.get('long', 0)
            if long_qty > 0:
                current_price = float(positions.get('longCurrentPrice', 0))
                total_holdings_value += current_price * float(long_qty)
        
        total_standing = float(balance) + total_holdings_value
        
        lines = []
        lines.append(f"Total Standing: ${total_standing:,.2f}\n")
        lines.append(f"Cash: ${balance:,.2f}\n")
        lines.append(f"Holdings Value: ${total_holdings_value:,.2f}\n")
        lines.append("")
        
        # Markdown table header
        lines.append("| Position | Qty | Entry Price | Current Price | P&L $ | P&L % |")
        lines.append("|---|---|---|---|---|---|")
        
        for ticker, positions in portfolio.items():
            long_qty = positions.get('long', 0)
            short_qty = positions.get('short', 0)
            
            if long_qty > 0:
                entry_price = float(positions.get('longEntryPrice', 0))
                current_price = float(positions.get('longCurrentPrice', entry_price))
                pnl = (current_price - entry_price) * float(long_qty)
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
                lines.append(f"| LONG {ticker} | {long_qty} | ${entry_price:.2f} | ${current_price:.2f} | ${pnl:,.2f} | {pnl_pct:+.2f}% |")
            
            if short_qty > 0:
                entry_price = float(positions.get('shortEntryPrice', 0))
                current_price = float(positions.get('shortCurrentPrice', entry_price))
                pnl = (entry_price - current_price) * float(short_qty)
                pnl_pct = ((entry_price - current_price) / entry_price * 100) if entry_price != 0 else 0
                lines.append(f"| SHORT {ticker} | {short_qty} | ${entry_price:.2f} | ${current_price:.2f} | ${pnl:,.2f} | {pnl_pct:+.2f}% |")
        
        return "\n".join(lines)
    
    @staticmethod
    def formatStrategyPerformance(metrics: Dict[str, Dict[str, Any]]) -> str:
        """
        Format strategy performance metrics.
        
        Args:
            metrics: Dict mapping strategy names -> {profitFactor, winCount, avgPnL, totalPnL, etc}
        
        Returns:
            Formatted performance report
        """
        if not metrics:
            return "No strategy performance data available yet."
        
        lines = ["Strategy Performance Report"]
        lines.append("=" * 60)
        
        for strategy, stats in metrics.items():
            lines.append(f"\n{strategy}:")
            lines.append(f"  Profit Factor: {stats.get('profitFactor', 0):.2f}")
            lines.append(f"  Total Trades: {stats.get('totalTrades', 0)}")
            lines.append(f"  Wins/Losses: {stats.get('winCount', 0)}/{stats.get('lossCount', 0)}")
            lines.append(f"  Avg Win: ${stats.get('avgWin', 0):.2f}")
            lines.append(f"  Avg Loss: ${stats.get('avgLoss', 0):.2f}")
            lines.append(f"  Total P&L: ${stats.get('totalPnL', 0):.2f}")
            lines.append(f"  Avg P&L per Trade: ${stats.get('avgPnl', 0):.2f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def formatRecentTrades(trades: List[Dict[str, Any]], limit: int = 10) -> str:
        """
        Format recent trades as markdown table.
        
        Args:
            trades: List of trade dicts {strategy, ticker, action, quantity, confidence, timestamp}
            limit: Max trades to display
        
        Returns:
            Formatted recent trades (markdown table)
        """
        if not trades:
            return "No recent trades."
        
        lines = []
        lines.append(f"Recent Trades (last {min(limit, len(trades))})")
        lines.append("")
        
        # Markdown table header
        lines.append("| Timestamp | Action | Ticker | Qty | Confidence | Strategy |")
        lines.append("|---|---|---|---|---|---|")
        
        for trade in trades[-limit:]:
            action_str = trade.get('action', 'HOLD').upper()
            ticker = trade.get('ticker', '?')
            qty = trade.get('quantity', 0)
            confidence = trade.get('confidence', 0) * 100
            strategy = trade.get('strategy', 'Unknown')
            timestamp = trade.get('timestamp', '?')
            
            lines.append(f"| {timestamp} | {action_str} | {ticker} | {qty} | {confidence:.0f}% | {strategy} |")
        
        return "\n".join(lines)
    
    @staticmethod
    def formatAgentStatus(agentId: int, isRunning: bool, isPaused: bool, 
                         totalTrades: int, currentDate: str = None) -> str:
        """
        Format agent status.
        
        Args:
            agentId: Agent ID
            isRunning: Whether agent is running
            isPaused: Whether agent is paused
            totalTrades: Total trades executed
            currentDate: Current simulation date (optional)
        
        Returns:
            Formatted status string
        """
        status = "RUNNING"
        if isPaused:
            status = "PAUSED"
        elif not isRunning:
            status = "STOPPED"
        
        date_str = f" at {currentDate}" if currentDate else ""
        return (
            f"Agent {agentId} Status: {status}\n"
            f"Total Trades: {totalTrades}\n"
            f"Simulation{date_str}"
        )
    
    @staticmethod
    def formatStrategyWeights(weights: Dict[str, float]) -> str:
        """
        Format strategy selection weights (from epsilon-greedy).
        
        Args:
            weights: Dict mapping strategy names -> probability weight
        
        Returns:
            Formatted weights
        """
        if not weights:
            return "Strategy weights not available."
        
        lines = ["Strategy Selection Weights"]
        lines.append("-" * 40)
        
        for strategy, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            percentage = weight * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            lines.append(f"{strategy:15} {percentage:5.1f}% {bar}")
        
        return "\n".join(lines)
    
    @staticmethod
    def formatClosedTrades(closedTrades: List[Dict[str, Any]], limit: int = 5) -> str:
        """
        Format recently closed trades with P&L.
        
        Args:
            closedTrades: List of closed trade dicts {strategy, pnl, pnlPercent, tradeType, ticker}
            limit: Max to display
        
        Returns:
            Formatted closed trades
        """
        if not closedTrades:
            return "No closed trades yet."
        
        lines = ["Closed Trades (Recent)"]
        lines.append("-" * 60)
        
        for trade in closedTrades[-limit:]:
            ticker = trade.get('ticker', '?')
            tradeType = trade.get('tradeType', '?').upper()
            pnl = trade.get('pnl', 0)
            pnlPercent = trade.get('pnlPercent', 0)
            strategy = trade.get('strategy', 'Unknown')
            
            pnl_color = "+" if pnl >= 0 else ""
            lines.append(
                f"{tradeType:5} {ticker:6} | P&L: {pnl_color}${pnl:>.2f} "
                f"({pnl_color}{pnlPercent:.1f}%) | {strategy}"
            )
        
        return "\n".join(lines)
    
    @staticmethod
    def formatAggregateStats(stats: Dict[str, Any]) -> str:
        """
        Format aggregate statistics across all agents.
        
        Args:
            stats: Dict with aggregate metrics {totalTrades, strategies: {...}}
        
        Returns:
            Formatted aggregate report
        """
        lines = ["Aggregate Statistics (All Agents)"]
        lines.append("=" * 60)
        
        total_trades = stats.get('totalTrades', 0)
        lines.append(f"Total Trades Executed: {total_trades}")
        
        strategies = stats.get('strategies', {})
        if strategies:
            lines.append("\nStrategy Summary:")
            for strategy, stratStats in strategies.items():
                total_pnl = stratStats.get('totalPnL', 0)
                trades = stratStats.get('totalTrades', 0)
                avg_pf = stratStats.get('avgProfitFactor', 0)
                lines.append(
                    f"  {strategy}: {trades} trades, "
                    f"Avg PF: {avg_pf:.2f}, Total P&L: ${total_pnl:,.2f}"
                )
        
        return "\n".join(lines)
    
    @staticmethod
    def formatPerformanceChartData(performanceTracker, exchange, accountId: int) -> Dict[str, Any]:
        """
        Prepare performance chart data including cumulative P&L and portfolio equity curve.
        
        Args:
            performanceTracker: PerformanceTracker instance
            exchange: StockExchange instance for portfolio history
            accountId: Account ID
        
        Returns:
            Dict with chart data suitable for streamlit charting:
            {
                'cumulativePnL': {date: cumulative_pnl},
                'portfolioValue': {date: total_value},
                'strategyPnL': {strategyName: {date: pnl}},
                'allFalseIfNoData': bool
            }
        """
        data = {
            'cumulativePnL': {},
            'portfolioValue': {},
            'strategyPnL': {},
            'hasData': False
        }
        
        # Get portfolio history
        portfolio_df = exchange.getPortfolioHistory(accountId)
        if not portfolio_df.empty:
            data['hasData'] = True
            for _, row in portfolio_df.iterrows():
                date_str = str(row['snapshotDate'])
                data['portfolioValue'][date_str] = float(row['totalValue'])
        
        # Get strategy-specific P&L from performance tracker
        all_strategies = performanceTracker._tradeHistory.keys()
        for strategy in all_strategies:
            trades = performanceTracker._tradeHistory.get(strategy, [])
            strategy_pnl = {}
            cumulative = 0.0
            
            # Sort trades by exit date
            sorted_trades = sorted(trades, key=lambda t: t.get('exitDate', ''))
            
            for trade in sorted_trades:
                exit_date = trade.get('exitDate', '')
                if exit_date:
                    cumulative += trade.get('pnl', 0)
                    strategy_pnl[exit_date] = cumulative
            
            if strategy_pnl:
                data['strategyPnL'][strategy] = strategy_pnl
                data['hasData'] = True
        
        return data
    
    @staticmethod
    def formatStrategyComparison(performanceTracker: Any, executionLog: List[Dict[str, Any]] = None) -> str:
        """
        Format comprehensive strategy comparison with win rates and loss analysis.
        
        Args:
            performanceTracker: PerformanceTracker instance
            executionLog: List of executed trades for fallback strategy discovery
        
        Returns:
            Formatted markdown table with strategy metrics
        """
        metrics = performanceTracker.getAllMetrics()
        
        # If no closed trades yet, extract strategy names and count from execution log
        if (not metrics or len(metrics) == 0) and executionLog:
            # Count trades per strategy from execution log
            strategy_trade_counts = {}
            for trade in executionLog:
                strategy = trade.get('strategy', 'Unknown')
                if strategy != 'Unknown':
                    strategy_trade_counts[strategy] = strategy_trade_counts.get(strategy, 0) + 1
            
            all_strategies = set(strategy_trade_counts.keys())
            
            # Initialize metrics for all strategies with proper trade counts
            metrics = {strategy: {
                'totalTrades': strategy_trade_counts.get(strategy, 0),
                'winCount': 0,
                'lossCount': 0,
                'winRate': 0,
                'avgWin': 0,
                'avgLoss': 0,
                'profitFactor': 0,
                'totalPnL': 0
            } for strategy in all_strategies}
        
        if not metrics or len(metrics) == 0:
            return "No strategy performance data available yet. Run the simulation to generate trades."
        
        lines = []
        lines.append("Strategy Performance Comparison")
        lines.append("")
        lines.append("| Strategy | Total Trades | Wins | Losses | Win Rate | Avg Win | Avg Loss | Profit Factor | Total P&L |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        
        for strategy in sorted(metrics.keys()):
            stats = metrics[strategy]
            
            total_count = stats.get('totalTrades', 0)
            win_count = stats.get('winCount', 0)
            loss_count = stats.get('lossCount', 0)
            win_rate = stats.get('winRate', 0) * 100 if 'winRate' in stats else (win_count / total_count * 100 if total_count > 0 else 0)
            avg_win = stats.get('avgWin', 0)
            avg_loss = stats.get('avgLoss', 0)
            profit_factor = stats.get('profitFactor', 0)
            total_pnl = stats.get('totalPnL', 0)
            
            lines.append(
                f"| {strategy} | {total_count} | {win_count} | {loss_count} | {win_rate:.1f}% | "
                f"${avg_win:,.2f} | ${avg_loss:,.2f} | {profit_factor:.2f} | ${total_pnl:,.2f} |"
            )
        
        return "\n".join(lines)
    
    @staticmethod
    def formatRecommendationQuality(recommendationMetrics: Dict[str, Dict[str, Any]]) -> str:
        """
        Format strategy recommendation quality metrics.
        Shows all recommendations (including HOLD) and their outcomes.
        
        Args:
            recommendationMetrics: Dict mapping strategy names -> 
                {totalRecommendations, holdCount, holdCorrect, holdMissedLong, holdMissedShort, 
                 holdAccuracy, longCount, shortCount, totalExecuted, pendingCount}
        
        Returns:
            Formatted recommendation quality report
        """
        if not recommendationMetrics:
            return "No recommendation data available yet."
        
        lines = ["## Analysis of HOLD recommendations by strategies:\n"]
        
        for strategy, metrics in recommendationMetrics.items():
            totalRecs = metrics.get('totalRecommendations', 0)
            if totalRecs == 0:
                continue
            
            holdCount = metrics.get('holdCount', 0)
            holdCorrect = metrics.get('holdCorrect', 0)
            holdMissedLong = metrics.get('holdMissedLong', 0)
            holdMissedShort = metrics.get('holdMissedShort', 0)
            holdAccuracy = metrics.get('holdAccuracy', 0.0)
            longCount = metrics.get('longCount', 0)
            shortCount = metrics.get('shortCount', 0)
            totalExecuted = metrics.get('totalExecuted', 0)
            pendingCount = metrics.get('pendingCount', 0)
            
            lines.append(f"**For the {strategy} approach:**\n")
            lines.append(f"It made a total of {totalRecs} recommended actions.\n")
            
            if holdCount > 0:
                holdPct = holdCount/totalRecs*100
                lines.append(f"Of these, {holdCount} ({holdPct:.1f}%) were HOLD recommendations.\n")
                if holdCorrect:
                    lines.append(f"Holding was successful {holdCorrect} times ({holdAccuracy*100:.1f}%) as the price was stable for a period following the recommendation.\n")
                if holdMissedLong:
                    lines.append(f"However, {holdMissedLong} times the price later increased, so a LONG would have been better.\n")
                if holdMissedShort:
                    lines.append(f"Additionally, {holdMissedShort} times the price later decreased, meaning a SHORT would have been more successful.\n")
                if pendingCount > 0:
                    lines.append(f"Note that {pendingCount} recommendations are yet to be evaluated as not enough time has passed.\n\n")
            
            if longCount > 0 or shortCount > 0:
                actionPct = totalExecuted/totalRecs*100
                lines.append(f"There were {totalExecuted} ({actionPct:.1f}%) LONG / SHORT recommendation(s).\n")
                if longCount > 0:
                    lines.append(f"LONG: {longCount}.\n")
                if shortCount > 0:
                    lines.append(f"SHORT: {shortCount}.\n")
            lines.append("---\n")
        
        return "\n".join(lines)
    
    @staticmethod
    def formatSimulationControl(speedMultiplier: float, startDate: str, 
                               endDate: str, currentDate: str) -> str:
        """
        Format simulation control info.
        
        Args:
            speedMultiplier: Simulation speed (e.g., 5.0x)
            startDate: Simulation start date
            endDate: Simulation end date
            currentDate: Current simulation date
        
        Returns:
            Formatted simulator info
        """
        lines = ["Simulation Control"]
        lines.append("-" * 40)
        lines.append(f"Speed: {speedMultiplier}x")
        lines.append(f"Period: {startDate} to {endDate}")
        lines.append(f"Current: {currentDate}")
        
        return "\n".join(lines)
