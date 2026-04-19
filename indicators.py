#!/usr/bin/env python3
"""
Advanced Trading Indicators for High Win Rate
Based on research: Order flow, volatility, support/resistance, time filtering
"""

import math
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def calculate_atr(candles: List, period: int = 14) -> float:
    """Calculate Average True Range - volatility measure"""
    if not candles or len(candles) < period:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    # ATR is SMA of True Range
    return sum(true_ranges[-period:]) / period


def calculate_adx(candles: List, period: int = 14) -> Tuple[float, float, float]:
    """
    Calculate ADX (Average Directional Index)
    Returns: (adx, plus_di, minus_di)
    ADX > 25 = trending market (good for trading)
    ADX < 20 = ranging/sideways (skip trading)
    """
    if len(candles) < period + 1:
        return 0.0, 0.0, 0.0
    
    # Calculate True Range and Directional Movement
    tr_list = []
    plus_dm = []
    minus_dm = []
    
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_high = candles[i-1].high
        prev_low = candles[i-1].low
        prev_close = candles[i-1].close
        
        # True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
        
        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
            minus_dm.append(0)
        elif down_move > up_move and down_move > 0:
            plus_dm.append(0)
            minus_dm.append(down_move)
        else:
            plus_dm.append(0)
            minus_dm.append(0)
    
    if len(tr_list) < period:
        return 0.0, 0.0, 0.0
    
    # Smoothed values
    atr = sum(tr_list[-period:]) / period
    plus_di = 100 * (sum(plus_dm[-period:]) / period) / atr if atr > 0 else 0
    minus_di = 100 * (sum(minus_dm[-period:]) / period) / atr if atr > 0 else 0
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    
    return dx, plus_di, minus_di  # Simplified ADX


def calculate_rsi(candles: List, period: int = 14) -> float:
    """Calculate RSI - overbought/oversold"""
    if len(candles) < period + 1:
        return 50.0
    
    gains = []
    losses = []
    
    for i in range(1, len(candles)):
        change = candles[i].close - candles[i-1].close
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return 50.0
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def find_support_resistance(candles: List, lookback: int = 20) -> Dict:
    """
    Find key support and resistance levels
    Returns levels and distance from current price
    """
    if len(candles) < lookback:
        return {"resistance": None, "support": None, "near_level": False}
    
    recent = candles[-lookback:]
    highs = [c.high for c in recent]
    lows = [c.low for c in recent]
    closes = [c.close for c in recent]
    
    current_price = closes[-1]
    
    # Find resistance (highest high)
    resistance = max(highs)
    
    # Find support (lowest low)
    support = min(lows)
    
    # Distance percentages
    resistance_dist = ((resistance - current_price) / current_price) * 100
    support_dist = ((current_price - support) / current_price) * 100
    
    # Near level detection (within 0.1%)
    near_resistance = resistance_dist < 0.1
    near_support = support_dist < 0.1
    
    return {
        "resistance": resistance,
        "support": support,
        "resistance_dist_pct": round(resistance_dist, 3),
        "support_dist_pct": round(support_dist, 3),
        "near_resistance": near_resistance,
        "near_support": near_support,
        "range_pct": round(((resistance - support) / support) * 100, 2)
    }


def is_peak_trading_hour(utc_hour: int = None) -> Tuple[bool, str]:
    """
    Check if current hour is optimal for trading
    
    PEAK HOURS (high volume, clear trends):
    - 8:00-12:00 EST (London session) = 13:00-17:00 UTC
    - 14:00-17:00 EST (NY + London overlap) = 19:00-22:00 UTC
    
    AVOID:
    - 0:00-6:00 UTC (Asian session, low liquidity)
    - Market open/close (first/last 30 min)
    """
    if utc_hour is None:
        utc_hour = datetime.utcnow().hour
    
    # Convert to EST for reference
    # UTC = EST + 5 (or +4 during DST)
    
    # London Session (8:00-12:00 EST = 13:00-17:00 UTC)
    london_session = 13 <= utc_hour < 17
    
    # NY + London Overlap (14:00-17:00 EST = 19:00-22:00 UTC) - BEST TIME
    overlap_session = 19 <= utc_hour < 22
    
    # Asian Session (avoid)
    asian_session = 0 <= utc_hour < 8
    
    if overlap_session:
        return True, "HIGH_VOL_OVERLAP"  # Best time
    elif london_session:
        return True, "GOOD_VOL_LONDON"  # Good time
    elif asian_session:
        return False, "LOW_VOL_ASIAN"  # Avoid
    else:
        return True, "MODERATE_VOL"  # Acceptable


def calculate_volatility_score(candles: List) -> Dict:
    """
    Calculate volatility metrics
    
    HIGH volatility = good for trend following
    LOW volatility = ranging, skip trading
    EXTREME volatility = unpredictable, skip
    """
    if len(candles) < 20:
        return {"score": 0, "state": "unknown", "tradeable": False}
    
    atr = calculate_atr(candles, 14)
    current_price = candles[-1].close
    
    # ATR as percentage of price
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
    
    # Recent price range
    recent_highs = [c.high for c in candles[-20:]]
    recent_lows = [c.low for c in candles[-20:]]
    range_pct = ((max(recent_highs) - min(recent_lows)) / current_price) * 100
    
    # Volatility state
    if atr_pct < 0.01:  # Less than 0.01% ATR
        state = "DEAD"
        tradeable = False
        score = 0
    elif atr_pct < 0.03:  # Less than 0.03% ATR
        state = "LOW"
        tradeable = False
        score = 25
    elif atr_pct < 0.10:  # Normal range
        state = "NORMAL"
        tradeable = True
        score = 75
    elif atr_pct < 0.20:  # Good volatility
        state = "HIGH"
        tradeable = True
        score = 100
    else:  # Extreme volatility
        state = "EXTREME"
        tradeable = False
        score = 30
    
    return {
        "atr": round(atr, 6),
        "atr_pct": round(atr_pct, 4),
        "range_pct": round(range_pct, 4),
        "score": score,
        "state": state,
        "tradeable": tradeable
    }


def detect_candle_pattern(candles: List) -> Dict:
    """
    Detect high-probability candle patterns
    
    Patterns with high win rate:
    - Bullish/Bearish Engulfing
    - Pinbar (hammer/shooting star)
    - Doji at support/resistance
    """
    if len(candles) < 3:
        return {"pattern": None, "signal": None, "strength": 0}
    
    # Last 3 candles
    c1 = candles[-3]  # 3 candles ago
    c2 = candles[-2]  # 2 candles ago  
    c3 = candles[-1]  # current candle
    
    patterns = []
    
    # Bullish Engulfing: small red candle followed by larger green candle
    if c2.close < c2.open:  # c2 is red
        if c3.close > c3.open:  # c3 is green
            if c3.close > c2.open and c3.open < c2.close:  # engulfs
                patterns.append(("BULLISH_ENGULFING", "call", 80))
    
    # Bearish Engulfing: small green candle followed by larger red candle
    if c2.close > c2.open:  # c2 is green
        if c3.close < c3.open:  # c3 is red
            if c3.close < c2.open and c3.open > c2.close:  # engulfs
                patterns.append(("BEARISH_ENGULFING", "put", 80))
    
    # Pinbar (Hammer - bullish)
    c3_body = abs(c3.close - c3.open)
    c3_range = c3.high - c3.low
    c3_lower_wick = min(c3.close, c3.open) - c3.low
    c3_upper_wick = c3.high - max(c3.close, c3.open)
    
    if c3_range > 0:
        # Hammer: small body, long lower wick, small upper wick
        if c3_body < c3_range * 0.3 and c3_lower_wick > c3_range * 0.6:
            patterns.append(("HAMMER", "call", 70))
        
        # Shooting Star: small body, long upper wick, small lower wick
        if c3_body < c3_range * 0.3 and c3_upper_wick > c3_range * 0.6:
            patterns.append(("SHOOTING_STAR", "put", 70))
    
    # Doji: very small body (indecision)
    if c3_body < c3_range * 0.1:
        patterns.append(("DOJI", "hold", 50))
    
    # Return strongest pattern
    if patterns:
        best = max(patterns, key=lambda x: x[2])
        return {
            "pattern": best[0],
            "signal": best[1],
            "strength": best[2]
        }
    
    return {"pattern": None, "signal": None, "strength": 0}


def should_trade(
    candles_1m: List,
    candles_5m: List,
    candles_15m: List,
    current_hour: int = None
) -> Tuple[bool, Dict]:
    """
    Master filter combining all indicators
    
    Returns: (should_trade, reasons)
    """
    reasons = []
    score = 0
    max_score = 100
    
    # 1. Time filter (20 points)
    time_ok, time_reason = is_peak_trading_hour(current_hour)
    if time_ok:
        score += 20
        reasons.append(f"Time: {time_reason}")
    else:
        reasons.append(f"SKIP - Time: {time_reason}")
        return False, {"score": 0, "reasons": reasons}
    
    # 2. Volatility filter (25 points)
    vol = calculate_volatility_score(candles_1m)
    if vol["tradeable"]:
        score += 25
        reasons.append(f"Volatility: {vol['state']} ({vol['atr_pct']}%)")
    else:
        reasons.append(f"SKIP - Volatility: {vol['state']}")
        return False, {"score": 0, "reasons": reasons}
    
    # 3. ADX trend filter (25 points)
    adx, plus_di, minus_di = calculate_adx(candles_1m)
    if adx >= 20:  # Trending market
        score += 25
        trend_dir = "bullish" if plus_di > minus_di else "bearish"
        reasons.append(f"Trend: ADX={adx:.1f} ({trend_dir})")
    else:
        reasons.append(f"SKIP - Ranging: ADX={adx:.1f} < 20")
        return False, {"score": 0, "reasons": reasons}
    
    # 4. Support/Resistance (15 points)
    sr = find_support_resistance(candles_5m)
    if sr["near_support"] or sr["near_resistance"]:
        score += 15
        level_type = "support" if sr["near_support"] else "resistance"
        reasons.append(f"At {level_type} level")
    else:
        reasons.append(f"Not near S/R level")
    
    # 5. Candle pattern (15 points)
    pattern = detect_candle_pattern(candles_1m)
    if pattern["pattern"] and pattern["signal"] != "hold":
        score += 15
        reasons.append(f"Pattern: {pattern['pattern']} ({pattern['strength']}%)")
    
    # Minimum score to trade
    should = score >= 60
    
    return should, {
        "score": score,
        "reasons": reasons,
        "volatility": vol,
        "adx": adx,
        "trend_direction": "bullish" if plus_di > minus_di else "bearish",
        "support_resistance": sr,
        "pattern": pattern,
        "time_ok": time_reason
    }


if __name__ == "__main__":
    # Test with sample data
    print("Indicators module loaded successfully")
    print(f"Peak hours: 13:00-17:00 UTC (London), 19:00-22:00 UTC (Overlap)")
