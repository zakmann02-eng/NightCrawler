"""
Polymarket Rule-Based Trading Bot — Polymarket US (CFTC Regulated)
- Sports markets only (NFL, NBA, Soccer, MLB, MMA/Boxing)
- Explicitly excludes Politics, Elections, Crypto & World Events
- Evaluates 4 triggers: probability threshold, price move, volume spike, time-to-resolution
- Claude scores each opportunity and assigns position size
- Take profit: +10% | Stop loss: -10% (immediate exit)
- Telegram notifications for all events
- Auth: Ed25519 keypair via Polymarket US developer portal

════════════════════════════════════════════════════════
HARD RULES — DO NOT MODIFY
════════════════════════════════════════════════════════
RULE 1 — NO BANK ACCESS
  The bot does not connect to, read from, or interact with any
  bank account, payment processor, debit card, ACH, wire transfer,
  or any external funding source of any kind. Ever.

RULE 2 — POLYMARKET BALANCE ONLY
  The bot trades exclusively using the USDC balance already present
  in the Polymarket US account. Only the account owner may deposit
  funds manually through the Polymarket US app or website.
  The bot will never request, initiate, trigger, or automate
  a deposit on behalf of anyone — including the account owner.

RULE 3 — NO WALLET MIRRORING
  The bot does not track, monitor, or copy any external wallet or
  trader. All decisions are made independently from live market
  data and Claude's analysis only.

RULE 4 — PERMITTED API CALLS ONLY
  The only endpoints this bot may call are:
    GET  /v1/markets/*         — read market data
    POST /v1/orders            — place trades within existing balance
    GET  /v1/portfolio/positions — read own positions only
  Deposit, withdrawal, transfer, and funding endpoints are
  strictly forbidden and must never be added.
════════════════════════════════════════════════════════
"""

import os
import time
import json
import base64
import logging
import requests
from datetime import datetime, timezone
from typing import Optional
from anthropic import Anthropic
from cryptography.hazmat.primitives.asymmetric import ed25519

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────
# Polymarket US Ed25519 credentials (from developer portal)
POLY_KEY_ID     = os.environ["POLY_KEY_ID"]      # Access Key ID shown in portal
POLY_SECRET_KEY = os.environ["POLY_SECRET_KEY"]  # Base64-encoded Ed25519 private key
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# ── Trading parameters ─────────────────────────────────────────────────────
POLL_INTERVAL_SEC       = 60
BANKROLL_USDC           = float(os.environ.get("BANKROLL_USDC", "100"))
TAKE_PROFIT_PCT         = 0.05
STOP_LOSS_PCT           = -0.05

# Trigger thresholds
PROB_HIGH_THRESHOLD     = 0.80
PROB_LOW_THRESHOLD      = 0.20
PRICE_MOVE_PCT          = 0.05
PRICE_MOVE_WINDOW_MIN   = 15
VOLUME_SPIKE_MULTIPLIER = 2.0
HOURS_TO_RESOLUTION     = 24

# ── Polymarket US API ──────────────────────────────────────────────────────
POLY_BASE_URL = "https://api.polymarket.us/v1"

# Load Ed25519 private key once at startup
_private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
    base64.b64decode(POLY_SECRET_KEY)[:32]
)

def auth_headers(method: str, path: str, body: str = "") -> dict:
    """Generate signed auth headers for every Polymarket US API request."""
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method.upper()}{path}{body}"
    signature = base64.b64encode(_private_key.sign(message.encode())).decode()
    return {
        "X-PM-Access-Key": POLY_KEY_ID,
        "X-PM-Timestamp": timestamp,
        "X-PM-Signature": signature,
        "Content-Type": "application/json",
    }

def poly_get(path: str) -> Optional[dict]:
    """Authenticated GET to Polymarket US API."""
    try:
        r = requests.get(
            f"{POLY_BASE_URL}{path}",
            headers=auth_headers("GET", path),
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"GET {path} failed: {e}")
        return None

def poly_post(path: str, payload: dict) -> Optional[dict]:
    """Authenticated POST to Polymarket US API."""
    try:
        body = json.dumps(payload)
        r = requests.post(
            f"{POLY_BASE_URL}{path}",
            headers=auth_headers("POST", path, body),
            data=body,
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"POST {path} failed: {e}")
        return None

# ── Anthropic client ───────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── In-memory position tracker ─────────────────────────────────────────────
# { market_slug: { "entry_price", "side", "size_usdc", "order_id", ... } }
open_positions: dict = {}

# ── Market filters ─────────────────────────────────────────────────────────
SPORTS_KEYWORDS = [
    "nfl", "football", "super bowl", "touchdown", "quarterback", "ncaa football",
    "nba", "basketball", "ncaa basketball", "march madness", "finals mvp",
    "soccer", "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
    "champions league", "world cup", "mls", "epl", "fifa", "goal", "uefa",
    "mlb", "baseball", "world series", "home run",
    "ufc", "mma", "boxing", "fight", "knockout", "bout", "championship belt",
    "match", "game winner", "playoff", "championship", "season wins",
    "sports", "sport", "league",
]

EXCLUDED_KEYWORDS = [
    # Politics & Elections
    "election", "elect", "vote", "voting", "ballot", "president", "senate",
    "congress", "republican", "democrat", "political", "politics", "policy",
    "governor", "mayor", "referendum", "impeach", "legislation", "bill passes",
    "approval rating", "poll ", "polling", "campaign", "candidate", "white house",
    "parliament", "prime minister", "cabinet", "administration", "inauguration",
    # Crypto & Web3
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency", "token",
    "defi", "nft", "altcoin", "solana", "sol", "binance", "coinbase", "blockchain",
    "stablecoin", "usdc price", "usdt", "xrp", "dogecoin", "doge", "web3",
    "dao", "airdrop", "halving", "memecoin",
    # World Events & Geopolitics
    "war", "conflict", "invasion", "ceasefire", "sanctions", "nato", "united nations",
    "un vote", "geopolit", "treaty", "diplomatic", "missile", "nuclear",
    "earthquake", "hurricane", "disaster", "climate", "gdp", "inflation",
    "federal reserve", "fed rate", "interest rate", "recession", "stock market",
    "s&p", "nasdaq", "dow jones", "oil price", "gold price",
]

def is_sports_market(market: dict) -> bool:
    question = (market.get("question") or "").lower()
    tags = " ".join([
        t.get("label", "") if isinstance(t, dict) else str(t)
        for t in (market.get("tags") or [])
    ]).lower()
    category = (market.get("category") or "").lower()
    combined = f"{question} {tags} {category}"

    for kw in EXCLUDED_KEYWORDS:
        if kw in combined:
            return False
    for kw in SPORTS_KEYWORDS:
        if kw in combined:
            return True
    return False

# ── Telegram ───────────────────────────────────────────────────────────────
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ── Market fetching ────────────────────────────────────────────────────────
def fetch_all_markets() -> list:
    """Fetch active markets from Polymarket US API."""
    markets = []
    cursor = None
    while True:
        path = "/markets?status=active&limit=100"
        if cursor:
            path += f"&cursor={cursor}"
        data = poly_get(path)
        if not data:
            break
        batch = data.get("data") or data if isinstance(data, list) else []
        markets.extend(batch)
        cursor = data.get("next_cursor") if isinstance(data, dict) else None
        if not cursor or not batch:
            break
    return markets

def fetch_market_history(slug: str) -> Optional[dict]:
    """Fetch price history for a market slug."""
    return poly_get(f"/markets/{slug}/prices-history?interval=1h")

# ── Trigger evaluation ─────────────────────────────────────────────────────
def evaluate_triggers(market: dict, price_history: Optional[dict]) -> dict:
    triggers = {
        "prob_threshold": False,
        "price_move": False,
        "volume_spike": False,
        "time_to_resolution": False,
        "details": {}
    }

    # 1. Probability threshold
    try:
        yes_price = float(market.get("yes_price") or market.get("outcomePrices", [0.5])[0])
        triggers["details"]["yes_probability"] = yes_price
        if yes_price <= PROB_LOW_THRESHOLD or yes_price >= PROB_HIGH_THRESHOLD:
            triggers["prob_threshold"] = True
    except (ValueError, TypeError, IndexError):
        pass

    # 2. Price move in window
    if price_history and "history" in price_history:
        history = price_history["history"]
        if len(history) >= 2:
            now_price = history[-1].get("p", 0)
            cutoff_ts = time.time() - (PRICE_MOVE_WINDOW_MIN * 60)
            window_prices = [h["p"] for h in history if h.get("t", 0) >= cutoff_ts]
            if window_prices:
                oldest = window_prices[0]
                if oldest > 0:
                    move = abs(now_price - oldest) / oldest
                    triggers["details"]["price_move_pct"] = round(move, 4)
                    if move >= PRICE_MOVE_PCT:
                        triggers["price_move"] = True

    # 3. Volume spike
    try:
        volume_24h = float(market.get("volume_24h") or market.get("volume24hr") or 0)
        volume_total = float(market.get("volume") or 0)
        start_date_str = market.get("start_date") or market.get("startDate", "")
        if start_date_str:
            start = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
            days_active = max(1, (datetime.now(timezone.utc) - start).days)
            avg = volume_total / days_active
            if avg > 0:
                ratio = volume_24h / avg
                triggers["details"]["volume_spike_ratio"] = round(ratio, 2)
                if ratio >= VOLUME_SPIKE_MULTIPLIER:
                    triggers["volume_spike"] = True
    except (ValueError, TypeError):
        pass

    # 4. Time to resolution
    try:
        end_str = market.get("end_date") or market.get("endDate", "")
        if end_str:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            hours = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
            triggers["details"]["hours_to_resolution"] = round(hours, 1)
            if 0 < hours <= HOURS_TO_RESOLUTION:
                triggers["time_to_resolution"] = True
    except (ValueError, TypeError):
        pass

    triggers["fired_count"] = sum([
        triggers["prob_threshold"],
        triggers["price_move"],
        triggers["volume_spike"],
        triggers["time_to_resolution"]
    ])
    return triggers

# ── Claude confidence scoring ──────────────────────────────────────────────
def claude_evaluate(market: dict, triggers: dict) -> Optional[dict]:
    prompt = f"""You are a prediction market trading analyst evaluating a Polymarket sports opportunity.

Market: {market.get('question', 'Unknown')}
Current YES probability: {triggers['details'].get('yes_probability', 'N/A')}
Volume (24h): {market.get('volume_24h', market.get('volume24hr', 'N/A'))} USDC
Total volume: {market.get('volume', 'N/A')} USDC
Hours to resolution: {triggers['details'].get('hours_to_resolution', 'N/A')}
Price move (recent window): {triggers['details'].get('price_move_pct', 'N/A')}
Volume spike ratio: {triggers['details'].get('volume_spike_ratio', 'N/A')}

Triggers fired: {triggers['fired_count']}/4
- Probability threshold: {triggers['prob_threshold']}
- Price move: {triggers['price_move']}
- Volume spike: {triggers['volume_spike']}
- Time to resolution: {triggers['time_to_resolution']}

Bankroll: {BANKROLL_USDC} USDC

Evaluate this sports market opportunity and respond ONLY with a JSON object (no markdown, no preamble):
{{
  "should_trade": true or false,
  "side": "YES" or "NO",
  "confidence": 1-10,
  "size_pct": 0.01 to 0.05,
  "reasoning": "one sentence explanation"
}}

Only recommend trading if there is genuine edge. Be conservative."""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        log.error(f"Claude evaluation error: {e}")
        return None

# ── Order execution ────────────────────────────────────────────────────────
def place_order(market: dict, side: str, size_usdc: float) -> Optional[str]:
    """Place an order via Polymarket US API."""
    slug = market.get("slug") or market.get("id")
    yes_price = float(market.get("yes_price") or market.get("outcomePrices", [0.5])[0])
    price = yes_price if side == "YES" else (1 - yes_price)

    payload = {
        "market_slug": slug,
        "side": side,
        "amount": round(size_usdc, 2),
        "order_type": "market",
    }
    resp = poly_post("/orders", payload)
    if resp:
        order_id = resp.get("order_id") or resp.get("id")
        log.info(f"Order placed: {order_id} | {side} ${size_usdc:.2f} on '{market.get('question')}'")
        return order_id
    return None

def close_position(market_slug: str, reason: str):
    """Close an open position immediately at market price."""
    position = open_positions.get(market_slug)
    if not position:
        return
    try:
        payload = {
            "market_slug": market_slug,
            "side": "NO" if position["side"] == "YES" else "YES",
            "amount": position["size_usdc"],
            "order_type": "market",
        }
        poly_post("/orders", payload)

        entry = position["entry_price"]
        current = position.get("current_price", entry)
        pnl_pct = round((current - entry) / entry * 100, 2)

        send_telegram(
            f"🔴 *POSITION CLOSED* — {reason}\n"
            f"Market: {position['question']}\n"
            f"Side: {position['side']}\n"
            f"Entry: {entry:.3f} | Exit: {current:.3f}\n"
            f"P&L: {pnl_pct:+.1f}%\n"
            f"Size: ${position['size_usdc']:.2f} USDC"
        )
        del open_positions[market_slug]
        log.info(f"Position closed: {market_slug} ({reason})")
    except Exception as e:
        log.error(f"Failed to close {market_slug}: {e}")
        send_telegram(f"⚠️ *FAILED TO CLOSE POSITION*\nMarket: {market_slug}\nError: {e}\nPlease close manually.")

# ── Position monitor ───────────────────────────────────────────────────────
def monitor_positions(markets_by_slug: dict):
    for slug, position in list(open_positions.items()):
        market = markets_by_slug.get(slug)
        if not market:
            continue
        try:
            yes_price = float(market.get("yes_price") or market.get("outcomePrices", [0.5])[0])
            current = yes_price if position["side"] == "YES" else (1 - yes_price)
            open_positions[slug]["current_price"] = current
            pnl_pct = (current - position["entry_price"]) / position["entry_price"]

            if pnl_pct >= TAKE_PROFIT_PCT:
                close_position(slug, f"✅ Take Profit (+{pnl_pct*100:.1f}%)")
            elif pnl_pct <= STOP_LOSS_PCT:
                close_position(slug, f"🛑 Stop Loss ({pnl_pct*100:.1f}%)")
        except Exception as e:
            log.error(f"Monitor error for {slug}: {e}")

# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    log.info("🤖 Polymarket US trading bot starting...")
    send_telegram(
        "🤖 *Polymarket US Bot Started*\n"
        "⚽🏈🏀⚾🥊 Sports markets only\n"
        "🚫 Excluding: Politics, Elections, Crypto & World Events\n"
        "TP: +5% | SL: -5%"
    )

    while True:
        try:
            log.info("Fetching active markets...")
            markets = fetch_all_markets()
            sports = [m for m in markets if is_sports_market(m)]
            log.info(f"Sports markets: {len(sports)}/{len(markets)}")

            markets_by_slug = {
                (m.get("slug") or m.get("id")): m for m in markets
            }

            # 1. Monitor open positions first
            monitor_positions(markets_by_slug)

            # 2. Scan for new opportunities
            for market in sports:
                slug = market.get("slug") or market.get("id")
                if not slug or slug in open_positions:
                    continue

                price_history = fetch_market_history(slug)
                triggers = evaluate_triggers(market, price_history)

                if triggers["fired_count"] < 2:
                    continue

                log.info(f"Opportunity: {market.get('question')} ({triggers['fired_count']} triggers)")

                evaluation = claude_evaluate(market, triggers)
                if not evaluation or not evaluation.get("should_trade"):
                    continue

                side = evaluation["side"]
                size_pct = float(evaluation["size_pct"])
                size_usdc = round(BANKROLL_USDC * size_pct, 2)
                confidence = evaluation["confidence"]
                reasoning = evaluation["reasoning"]

                order_id = place_order(market, side, size_usdc)
                if not order_id:
                    continue

                yes_price = float(market.get("yes_price") or market.get("outcomePrices", [0.5])[0])
                entry_price = yes_price if side == "YES" else (1 - yes_price)

                open_positions[slug] = {
                    "question": market.get("question"),
                    "side": side,
                    "entry_price": entry_price,
                    "current_price": entry_price,
                    "size_usdc": size_usdc,
                    "order_id": order_id,
                    "confidence": confidence,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                }

                fired = []
                if triggers["prob_threshold"]: fired.append("Prob threshold")
                if triggers["price_move"]: fired.append("Price move")
                if triggers["volume_spike"]: fired.append("Volume spike")
                if triggers["time_to_resolution"]: fired.append("Time to resolution")

                send_telegram(
                    f"🟢 *NEW POSITION OPENED*\n"
                    f"Market: {market.get('question')}\n"
                    f"Side: {side} | Entry: {entry_price:.3f}\n"
                    f"Size: ${size_usdc:.2f} USDC ({size_pct*100:.1f}% bankroll)\n"
                    f"Confidence: {confidence}/10\n"
                    f"Triggers: {', '.join(fired)}\n"
                    f"Reasoning: _{reasoning}_\n"
                    f"TP: +5% | SL: -5%"
                )

        except Exception as e:
            log.error(f"Main loop error: {e}")
            send_telegram(f"⚠️ *Bot Error*\n{str(e)}")

        log.info(f"Sleeping {POLL_INTERVAL_SEC}s | Open positions: {len(open_positions)}")
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    main()
