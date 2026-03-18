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
        Format portfolio holdings into readable text.
        
        Args:
            portfolio: Dict with held positions {ticker: {quantity, entryPrice, tradeType}}
            balance: Current account balance
        
        Returns:
            Formatted portfolio summary string
        """
        if not portfolio:
            return f"Your portfolio is empty. Current balance: ${balance:,.2f}"
        
        lines = [f"Portfolio Summary (Cash Balance: ${balance:,.2f})"]
        lines.append("=" * 50)
        
        for ticker, positions in portfolio.items():
            long_qty = positions.get('long', 0)
            short_qty = positions.get('short', 0)
            
            if long_qty > 0:
                entry_price = positions.get('longEntryPrice', 0)
                lines.append(f"\nLONG: {ticker} x{long_qty} @ ${entry_price:.2f}")
            
            if short_qty > 0:
                entry_price = positions.get('shortEntryPrice', 0)
                lines.append(f"SHORT: {ticker} x{short_qty} @ ${entry_price:.2f}")
            
            lines.append("-" * 50)
        
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
        Format recent trades into readable list.
        
        Args:
            trades: List of trade dicts {strategy, ticker, action, quantity, confidence, timestamp}
            limit: Max trades to display
        
        Returns:
            Formatted recent trades
        """
        if not trades:
            return "No recent trades."
        
        lines = [f"Recent Trades (last {min(limit, len(trades))})"]
        lines.append("-" * 60)
        
        for trade in trades[-limit:]:
            action_str = trade.get('action', 'HOLD').upper()
            ticker = trade.get('ticker', '?')
            qty = trade.get('quantity', 0)
            confidence = trade.get('confidence', 0) * 100
            strategy = trade.get('strategy', 'Unknown')
            timestamp = trade.get('timestamp', '?')
            
            lines.append(
                f"[{timestamp}] {action_str} {ticker} x{qty} "
                f"({confidence:.0f}% conf) via {strategy}"
            )
        
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
