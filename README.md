# Pocket Options AI Trading Agent

An AI-powered trading agent for Pocket Option that uses the moonshotai/kimi-k2.5 model via NVIDIA proxy for market analysis and trading decisions.

## Features

- **AI-Powered Analysis**: Uses NVIDIA proxy with moonshotai/kimi-k2.5 for intelligent market analysis
- **Async Trading**: Built on async Pocket Option API for efficient operations
- **Risk Management**: Confidence-based position sizing and trade filtering
- **Multi-Asset Support**: Trade multiple currency pairs simultaneously
- **Demo & Real Mode**: Safe testing with demo account before real trading
- **Session Tracking**: Detailed logging and trade history with win/loss statistics

## Requirements

- Python 3.12+
- Pocket Option account (demo or real)
- Valid SSID from Pocket Option

## Installation

```bash
pip install git+https://github.com/ChipaDevTeam/PocketOptionAPI.git aiohttp
```

## Getting Your SSID

1. Open Pocket Option in your browser
2. Open Developer Tools (F12)
3. Go to **Network** tab
4. Filter by **WS** (WebSocket)
5. Find a message starting with `42["auth"`
6. Copy the **entire** message including `42["auth",{...}]`

Example SSID format:
```
42["auth",{"session":"abcd1234efgh5678","isDemo":1,"uid":12345,"platform":1}]
```

## Configuration

Set environment variables or pass directly:

| Variable | Default | Description |
|----------|---------|-------------|
| `POCKET_OPTION_SSID` | (required) | Your Pocket Option SSID |
| `POCKET_OPTION_DEMO` | `true` | Use demo account (true/false) |
| `POCKET_OPTION_AMOUNT` | `1` | Base trade amount in USD |
| `POCKET_OPTION_DURATION` | `60` | Trade duration in seconds |
| `POCKET_OPTION_INTERVAL` | `120` | Time between analyses in seconds |
| `POCKET_OPTION_MAX_TRADES` | `10` | Maximum trades per session |
| `POCKET_OPTION_MIN_CONFIDENCE` | `0.6` | Minimum AI confidence to trade |
| `POCKET_OPTION_ASSETS` | `EURUSD_otc,GBPUSD_otc,USDJPY_otc` | Comma-separated asset list |

## Usage

### Run with environment variables

```bash
export POCKET_OPTION_SSID='42["auth",{"session":"xxx","isDemo":1,"uid":12345,"platform":1}]'
python agent.py
```

### Run interactively

```bash
python agent.py
# Will prompt for SSID if not set
```

### Run with custom settings

```bash
POCKET_OPTION_SSID="your-ssid" \
POCKET_OPTION_DEMO=true \
POCKET_OPTION_AMOUNT=2 \
POCKET_OPTION_DURATION=60 \
POCKET_OPTION_INTERVAL=90 \
POCKET_OPTION_MAX_TRADES=5 \
POCKET_OPTION_MIN_CONFIDENCE=0.7 \
POCKET_OPTION_ASSETS="EURUSD_otc,GBPUSD_otc" \
python agent.py
```

## AI Decision Process

The agent uses the moonshotai/kimi-k2.5 model to:

1. **Analyze Market Context** - Reviews recent candle data, price trends, momentum
2. **Generate Trade Signal** - Outputs CALL, PUT, or HOLD with confidence score
3. **Calculate Position Size** - Adjusts amount based on confidence level

The AI considers:
- Price trend direction and momentum
- Recent price highs/lows
- Up vs down move counts
- Account balance
- Recent trade history

## Risk Management

- Only trades when AI confidence exceeds minimum threshold (default 60%)
- Position sizing scales with confidence (higher confidence = larger position)
- Maximum position capped at 2x base trade amount
- Session limits prevent runaway trading
- Demo mode by default for safe testing

## Output Example

```
2026-04-17 10:15:00 | INFO | ============================================================
2026-04-17 10:15:00 | INFO | POCKET OPTIONS AI TRADING AGENT
2026-04-17 10:15:00 | INFO | ============================================================
2026-04-17 10:15:00 | INFO | Mode: DEMO
2026-04-17 10:15:00 | INFO | Assets: EURUSD_otc, GBPUSD_otc, USDJPY_otc
2026-04-17 10:15:00 | INFO | Trade interval: 120s
2026-04-17 10:15:00 | INFO | Min confidence: 60%
2026-04-17 10:15:00 | INFO | Max trades per session: 10
2026-04-17 10:15:00 | INFO | Connected to Pocket Option (DEMO)
2026-04-17 10:15:00 | INFO | Analyzing EURUSD_otc...
2026-04-17 10:15:02 | INFO | AI Decision: CALL (confidence: 75%)
2026-04-17 10:15:02 | INFO | Reasoning: Strong bullish momentum with higher highs
2026-04-17 10:15:02 | INFO | Placing CALL order: EURUSD_otc $1.50 for 60s (confidence: 75%)
2026-04-17 10:15:03 | INFO | Order placed: abc123
2026-04-17 10:16:08 | INFO | Trade abc123: WIN (profit: $1.35)
```

## Disclaimer

**WARNING: Binary options trading involves significant risk.**

- This tool is for educational and research purposes only
- Past performance does not guarantee future results
- Trading can result in partial or total loss of invested capital
- Always use demo mode first to test strategies
- Never trade with money you cannot afford to lose
- This project is not affiliated with Pocket Option

## License

MIT License - Use at your own risk.
