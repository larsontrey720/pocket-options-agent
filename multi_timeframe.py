#!/usr/bin/env python3
"""
Multi-Timeframe Trend Analysis for Pocket Options
Quick win: Only trade when all timeframes align
"""

import logging

logger = logging.getLogger("mtf")


def analyze_multi_timeframe(candles_1m: list, candles_5m: list = None, candles_15m: list = None) -> dict:
    """
    Analyze trend across multiple timeframes.
    
    Returns alignment score and trend direction.
    Higher alignment = higher confidence trade.
    """
    
    def get_trend(candles):
        if not candles or len(candles) < 5:
            return "unknown", 0.0
        
        closes = [c.close for c in candles[-20:]] if len(candles) >= 20 else [c.close for c in candles]
        opens = [c.open for c in candles[-20:]] if len(candles) >= 20 else [c.open for c in candles]
        
        # Price change
        price_change = (closes[-1] - opens[0]) / opens[0] if opens[0] else 0
        
        # Up vs down candles
        up = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down = len(closes) - 1 - up
        
        # Determine trend
        if up > down * 1.3 and price_change > 0.0005:
            return "bullish", (up / (up + down)) if (up + down) > 0 else 0.5
        elif down > up * 1.3 and price_change < -0.0005:
            return "bearish", (down / (up + down)) if (up + down) > 0 else 0.5
        else:
            return "neutral", 0.5
    
    # Get trend for each timeframe
    trend_1m, strength_1m = get_trend(candles_1m)
    trend_5m, strength_5m = get_trend(candles_5m) if candles_5m else ("unknown", 0)
    trend_15m, strength_15m = get_trend(candles_15m) if candles_15m else ("unknown", 0)
    
    # Calculate alignment
    trends = [t for t in [trend_1m, trend_5m, trend_15m] if t != "unknown"]
    
    if not trends:
        return {
            "alignment": 0,
            "direction": "unknown",
            "trade_allowed": False,
            "timeframes": {
                "1m": {"trend": trend_1m, "strength": strength_1m},
                "5m": {"trend": trend_5m, "strength": strength_5m},
                "15m": {"trend": trend_15m, "strength": strength_15m},
            }
        }
    
    # Count matching trends
    bullish_count = sum(1 for t in trends if t == "bullish")
    bearish_count = sum(1 for t in trends if t == "bearish")
    neutral_count = sum(1 for t in trends if t == "neutral")
    
    # Determine overall direction
    if bullish_count >= 2 and bullish_count > bearish_count:
        direction = "bullish"
        alignment = bullish_count / len(trends)
    elif bearish_count >= 2 and bearish_count > bullish_count:
        direction = "bearish"
        alignment = bearish_count / len(trends)
    else:
        direction = "neutral"
        alignment = 0.3  # Low alignment - don't trade
    
    # Only allow trade if strong alignment (2+ timeframes agree)
    trade_allowed = alignment >= 0.66 and direction != "neutral"
    
    # Apply penalty if higher timeframes disagree
    if trend_15m != "unknown" and trend_15m != direction and direction != "neutral":
        trade_allowed = False
        logger.info(f"MTF BLOCKED: 15m trend ({trend_15m}) disagrees with {direction}")
    
    return {
        "alignment": round(alignment, 2),
        "direction": direction,
        "trade_allowed": trade_allowed,
        "timeframes": {
            "1m": {"trend": trend_1m, "strength": round(strength_1m, 2)},
            "5m": {"trend": trend_5m, "strength": round(strength_5m, 2)},
            "15m": {"trend": trend_15m, "strength": round(strength_15m, 2)},
        }
    }


def get_mtf_context_for_ai(mtf_result: dict) -> str:
    """Format MTF result for AI context"""
    
    tf = mtf_result.get("timeframes", {})
    
    return f"""
MULTI-TIMEFRAME ANALYSIS:
1M Trend: {tf.get('1m', {}).get('trend', '?')} (strength: {tf.get('1m', {}).get('strength', 0):.0%})
5M Trend: {tf.get('5m', {}).get('trend', '?')} (strength: {tf.get('5m', {}).get('strength', 0):.0%})
15M Trend: {tf.get('15m', {}).get('trend', '?')} (strength: {tf.get('15m', {}).get('strength', 0):.0%})
Alignment: {mtf_result.get('alignment', 0):.0%} ({mtf_result.get('direction', 'unknown')})
Trade Allowed: {mtf_result.get('trade_allowed', False)}

RULE: If alignment < 66% or 15M trend disagrees -> HOLD (skip trade)
"""
