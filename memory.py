#!/usr/bin/env python3
"""
Pocket Options Learning System
Self-reflection, pattern recognition, and persistent learning
"""

import json
import os
from datetime import datetime
from typing import Optional
from collections import defaultdict
import statistics


class TradingMemory:
    """Persistent memory for learning from trades"""
    
    def __init__(self, memory_dir: str = "/home/workspace/pocket-options-agent/memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        self.trade_history_file = os.path.join(memory_dir, "trade_history.json")
        self.learnings_file = os.path.join(memory_dir, "learnings.json")
        self.patterns_file = os.path.join(memory_dir, "patterns.json")
        self.strategy_file = os.path.join(memory_dir, "strategy.json")
        
        # In-memory caches
        self.trade_history = self._load_json(self.trade_history_file, [])
        self.learnings = self._load_json(self.learnings_file, {
            "insights": [],
            "rules_to_follow": [],
            "mistakes_to_avoid": [],
            "successful_patterns": [],
        })
        self.patterns = self._load_json(self.patterns_file, {
            "win_conditions": {},  # Conditions that lead to wins
            "loss_conditions": {},  # Conditions that lead to losses
            "asset_performance": {},  # Per-asset stats
            "time_performance": {},  # Time-of-day stats
        })
        self.strategy = self._load_json(self.strategy_file, {
            "confidence_threshold": 0.6,
            "preferred_assets": [],
            "avoid_assets": [],
            "best_trading_hours": [],
            "worst_trading_hours": [],
            "max_consecutive_losses": 3,
            "cooldown_after_loss_minutes": 5,
        })
    
    def _load_json(self, filepath: str, default):
        """Load JSON file or return default"""
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    return json.load(f)
            except:
                return default
        return default
    
    def _save_json(self, filepath: str, data):
        """Save JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def record_trade(self, trade: dict):
        """Record a completed trade"""
        trade["recorded_at"] = datetime.now().isoformat()
        self.trade_history.append(trade)
        self._save_json(self.trade_history_file, self.trade_history)
        
        # Update patterns
        self._update_patterns(trade)
    
    def _update_patterns(self, trade: dict):
        """Update pattern recognition based on trade result"""
        asset = trade.get("asset", "unknown")
        status = trade.get("status", "unknown")
        direction = trade.get("direction", "unknown")
        confidence = trade.get("confidence", 0)
        profit = trade.get("profit", 0)
        timestamp = trade.get("timestamp", "")
        
        # Extract hour from timestamp
        try:
            hour = datetime.fromisoformat(timestamp).hour
        except:
            hour = -1
        
        # Update asset performance
        if asset not in self.patterns["asset_performance"]:
            self.patterns["asset_performance"][asset] = {
                "total": 0, "wins": 0, "losses": 0, 
                "total_profit": 0, "win_rate": 0
            }
        
        ap = self.patterns["asset_performance"][asset]
        ap["total"] += 1
        ap["total_profit"] += profit
        if status == "win":
            ap["wins"] += 1
        elif status == "lose":
            ap["losses"] += 1
        ap["win_rate"] = ap["wins"] / ap["total"] * 100 if ap["total"] > 0 else 0
        
        # Update time performance
        hour_key = str(hour)
        if hour_key not in self.patterns["time_performance"]:
            self.patterns["time_performance"][hour_key] = {
                "total": 0, "wins": 0, "losses": 0, "profit": 0
            }
        
        tp = self.patterns["time_performance"][hour_key]
        tp["total"] += 1
        tp["profit"] += profit
        if status == "win":
            tp["wins"] += 1
        elif status == "lose":
            tp["losses"] += 1
        
        # Update win/loss conditions
        condition_key = f"{direction}_{asset}"
        if status == "win":
            if condition_key not in self.patterns["win_conditions"]:
                self.patterns["win_conditions"][condition_key] = {"count": 0, "avg_confidence": 0}
            wc = self.patterns["win_conditions"][condition_key]
            wc["count"] += 1
            wc["avg_confidence"] = (wc["avg_confidence"] * (wc["count"] - 1) + confidence) / wc["count"]
        elif status == "lose":
            if condition_key not in self.patterns["loss_conditions"]:
                self.patterns["loss_conditions"][condition_key] = {"count": 0, "avg_confidence": 0}
            lc = self.patterns["loss_conditions"][condition_key]
            lc["count"] += 1
            lc["avg_confidence"] = (lc["avg_confidence"] * (lc["count"] - 1) + confidence) / lc["count"]
        
        self._save_json(self.patterns_file, self.patterns)
    
    def reflect(self) -> dict:
        """
        Analyze all trades and generate insights.
        Returns self-reflection results.
        """
        if len(self.trade_history) < 10:
            return {"status": "insufficient_data", "trades_needed": 10 - len(self.trade_history)}
        
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "total_trades": len(self.trade_history),
            "wins": sum(1 for t in self.trade_history if t.get("status") == "win"),
            "losses": sum(1 for t in self.trade_history if t.get("status") == "lose"),
            "total_profit": sum(t.get("profit", 0) for t in self.trade_history),
            "insights": [],
            "recommendations": [],
        }
        
        reflection["win_rate"] = reflection["wins"] / reflection["total_trades"] * 100 if reflection["total_trades"] > 0 else 0
        
        # Analyze asset performance
        best_assets = []
        worst_assets = []
        for asset, stats in self.patterns["asset_performance"].items():
            if stats["total"] >= 3:  # Need at least 3 trades
                if stats["win_rate"] >= 60:
                    best_assets.append((asset, stats["win_rate"], stats["total_profit"]))
                elif stats["win_rate"] < 40:
                    worst_assets.append((asset, stats["win_rate"], stats["total_profit"]))
        
        # Sort by win rate
        best_assets.sort(key=lambda x: x[1], reverse=True)
        worst_assets.sort(key=lambda x: x[1])
        
        if best_assets:
            reflection["insights"].append(
                f"Best performing assets: {', '.join([f'{a} ({r:.0f}% win)' for a, r, p in best_assets[:3]])}"
            )
            reflection["recommendations"].append({
                "type": "focus_assets",
                "assets": [a for a, r, p in best_assets[:3]],
                "reason": "High win rate"
            })
        
        if worst_assets:
            reflection["insights"].append(
                f"Worst performing assets: {', '.join([f'{a} ({r:.0f}% win)' for a, r, p in worst_assets[:3]])}"
            )
            reflection["recommendations"].append({
                "type": "avoid_assets",
                "assets": [a for a, r, p in worst_assets[:3]],
                "reason": "Low win rate"
            })
        
        # Analyze time performance
        best_hours = []
        worst_hours = []
        for hour, stats in self.patterns["time_performance"].items():
            if stats["total"] >= 3:
                win_rate = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
                if win_rate >= 60:
                    best_hours.append((int(hour), win_rate, stats["profit"]))
                elif win_rate < 40:
                    worst_hours.append((int(hour), win_rate, stats["profit"]))
        
        best_hours.sort(key=lambda x: x[1], reverse=True)
        worst_hours.sort(key=lambda x: x[1])
        
        if best_hours:
            reflection["insights"].append(
                f"Best trading hours: {', '.join([f'{h}:00 ({r:.0f}% win)' for h, r, p in best_hours[:3]])}"
            )
            reflection["recommendations"].append({
                "type": "best_hours",
                "hours": [h for h, r, p in best_hours[:3]],
                "reason": "High win rate during these hours"
            })
        
        if worst_hours:
            reflection["insights"].append(
                f"Worst trading hours: {', '.join([f'{h}:00 ({r:.0f}% win)' for h, r, p in worst_hours[:3]])}"
            )
        
        # Analyze win/loss patterns
        common_win_conditions = sorted(
            self.patterns["win_conditions"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:5]
        
        common_loss_conditions = sorted(
            self.patterns["loss_conditions"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:5]
        
        if common_win_conditions:
            reflection["insights"].append(
                f"Most successful patterns: {', '.join([f'{k} ({v[\"count\"]} wins)' for k, v in common_win_conditions[:3]])}"
            )
        
        if common_loss_conditions:
            reflection["insights"].append(
                f"Most common failures: {', '.join([f'{k} ({v[\"count\"]} losses)' for k, v in common_loss_conditions[:3]])}"
            )
            reflection["recommendations"].append({
                "type": "avoid_patterns",
                "patterns": [k for k, v in common_loss_conditions[:3]],
                "reason": "High failure rate"
            })
        
        # Analyze confidence levels
        win_confidences = [t.get("confidence", 0) for t in self.trade_history if t.get("status") == "win"]
        loss_confidences = [t.get("confidence", 0) for t in self.trade_history if t.get("status") == "lose"]
        
        if win_confidences and loss_confidences:
            avg_win_conf = statistics.mean(win_confidences)
            avg_loss_conf = statistics.mean(loss_confidences)
            
            reflection["insights"].append(
                f"Avg confidence on wins: {avg_win_conf:.0%} | On losses: {avg_loss_conf:.0%}"
            )
            
            # If losses happen at high confidence, AI is overconfident
            if avg_loss_conf > avg_win_conf:
                reflection["insights"].append(
                    "WARNING: AI tends to be overconfident on losing trades"
                )
                reflection["recommendations"].append({
                    "type": "raise_confidence_threshold",
                    "current": self.strategy.get("confidence_threshold", 0.6),
                    "suggested": min(0.8, avg_win_conf + 0.1),
                    "reason": "AI is overconfident on losses"
                })
        
        # Streak analysis
        streaks = self._analyze_streaks()
        if streaks["max_win_streak"] >= 3:
            reflection["insights"].append(f"Best win streak: {streaks['max_win_streak']}")
        if streaks["max_loss_streak"] >= 3:
            reflection["insights"].append(f"Worst loss streak: {streaks['max_loss_streak']}")
            reflection["recommendations"].append({
                "type": "add_loss_cooldown",
                "streak_threshold": 3,
                "cooldown_minutes": 5,
                "reason": "Prevent cascade losses"
            })
        
        # Update learnings
        self._update_learnings(reflection)
        
        return reflection
    
    def _analyze_streaks(self) -> dict:
        """Analyze winning and losing streaks"""
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in self.trade_history:
            status = trade.get("status")
            if status == "win":
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif status == "lose":
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_win_streak": current_win_streak,
            "current_loss_streak": current_loss_streak,
        }
    
    def _update_learnings(self, reflection: dict):
        """Update learnings based on reflection"""
        # Add new insights
        for insight in reflection.get("insights", []):
            if insight not in self.learnings["insights"]:
                self.learnings["insights"].append({
                    "text": insight,
                    "timestamp": reflection["timestamp"],
                    "trades_analyzed": reflection["total_trades"],
                })
        
        # Update rules from recommendations
        for rec in reflection.get("recommendations", []):
            if rec["type"] == "avoid_assets":
                for asset in rec["assets"]:
                    if asset not in self.strategy["avoid_assets"]:
                        self.strategy["avoid_assets"].append(asset)
            
            elif rec["type"] == "focus_assets":
                self.strategy["preferred_assets"] = rec["assets"]
            
            elif rec["type"] == "best_hours":
                self.strategy["best_trading_hours"] = rec["hours"]
            
            elif rec["type"] == "raise_confidence_threshold":
                self.strategy["confidence_threshold"] = rec["suggested"]
            
            elif rec["type"] == "add_loss_cooldown":
                self.strategy["max_consecutive_losses"] = rec.get("streak_threshold", 3)
                self.strategy["cooldown_after_loss_minutes"] = rec.get("cooldown_minutes", 5)
        
        # Keep only last 50 insights
        self.learnings["insights"] = self.learnings["insights"][-50:]
        
        self._save_json(self.learnings_file, self.learnings)
        self._save_json(self.strategy_file, self.strategy)
    
    def get_context_for_ai(self) -> str:
        """Generate context string for AI based on learnings"""
        context_parts = []
        
        # Recent performance
        if len(self.trade_history) >= 5:
            recent = self.trade_history[-20:]
            wins = sum(1 for t in recent if t.get("status") == "win")
            losses = sum(1 for t in recent if t.get("status") == "lose")
            win_rate = wins / len(recent) * 100 if recent else 0
            profit = sum(t.get("profit", 0) for t in recent)
            
            context_parts.append(f"Recent 20 trades: {win_rate:.0f}% win rate, ${profit:.2f} P&L")
        
        # Known good/bad conditions
        if self.patterns["win_conditions"]:
            top_wins = sorted(self.patterns["win_conditions"].items(), key=lambda x: x[1]["count"], reverse=True)[:3]
            context_parts.append(f"Best patterns: {', '.join([k for k, v in top_wins])}")
        
        if self.patterns["loss_conditions"]:
            top_losses = sorted(self.patterns["loss_conditions"].items(), key=lambda x: x[1]["count"], reverse=True)[:3]
            context_parts.append(f"AVOID patterns: {', '.join([k for k, v in top_losses])}")
        
        # Strategy adjustments
        if self.strategy["avoid_assets"]:
            context_parts.append(f"AVOID assets: {', '.join(self.strategy['avoid_assets'])}")
        
        if self.strategy["preferred_assets"]:
            context_parts.append(f"Prefer assets: {', '.join(self.strategy['preferred_assets'])}")
        
        # Current threshold
        threshold = self.strategy.get("confidence_threshold", 0.6)
        context_parts.append(f"Min confidence threshold: {threshold:.0%}")
        
        # Recent insights
        if self.learnings["insights"]:
            recent_insights = self.learnings["insights"][-3:]
            context_parts.append("Recent learnings:")
            for i in recent_insights:
                context_parts.append(f"  - {i['text']}")
        
        return "\n".join(context_parts)
    
    def get_strategy(self) -> dict:
        """Get current strategy parameters"""
        return self.strategy.copy()
    
    def should_avoid_trade(self, asset: str, hour: int) -> tuple:
        """
        Check if trade should be avoided based on learnings.
        Returns (should_avoid: bool, reason: str)
        """
        # Check avoid list
        if asset in self.strategy.get("avoid_assets", []):
            return True, f"Asset {asset} has low win rate historically"
        
        # Check worst hours
        if hour in self.strategy.get("worst_trading_hours", []):
            return True, f"Hour {hour}:00 has poor performance historically"
        
        # Check if in cooldown after losses
        streaks = self._analyze_streaks()
        if streaks["current_loss_streak"] >= self.strategy.get("max_consecutive_losses", 3):
            return True, f"In cooldown after {streaks['current_loss_streak']} consecutive losses"
        
        return False, ""
    
    def clear_memory(self):
        """Clear all memory (fresh start)"""
        self.trade_history = []
        self.learnings = {"insights": [], "rules_to_follow": [], "mistakes_to_avoid": [], "successful_patterns": []}
        self.patterns = {"win_conditions": {}, "loss_conditions": {}, "asset_performance": {}, "time_performance": {}}
        self.strategy = {
            "confidence_threshold": 0.6,
            "preferred_assets": [],
            "avoid_assets": [],
            "best_trading_hours": [],
            "worst_trading_hours": [],
            "max_consecutive_losses": 3,
            "cooldown_after_loss_minutes": 5,
        }
        
        self._save_json(self.trade_history_file, self.trade_history)
        self._save_json(self.learnings_file, self.learnings)
        self._save_json(self.patterns_file, self.patterns)
        self._save_json(self.strategy_file, self.strategy)
