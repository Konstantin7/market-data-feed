#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборщик рыночных данных для ежедневной сводки.

Запускается на GitHub Actions (бесплатный облачный раннер, ноутбук не нужен),
тянет дневные свечи по трём рынкам, считает индикаторы в pandas и кладёт
результат в data/market_snapshot.json.

Смысл в том, что индикаторы здесь СЧИТАЮТСЯ по реальным свечам, а не
пересказываются моделью со стороннего сайта. Claude потом просто читает готовый JSON:

    curl -s https://raw.githubusercontent.com/<user>/<repo>/main/data/market_snapshot.json

Источники:
  RU     — T-Invest API, если задан секрет TINVEST_TOKEN; иначе MOEX ISS
  US     — Stooq         https://stooq.com
  CRYPTO — Binance       https://api.binance.com

Кроме T-Invest ключи не нужны нигде. Токен T-Invest берётся ТОЛЬКО из переменной
окружения (GitHub Secrets) и никогда не хранится в коде, в репозитории и в снимке.
Достаточно токена «только для чтения».

Зависимости: requests, pandas.
"""
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

MSK = timezone(timedelta(hours=3))
HTTP_TIMEOUT = 30
RETRIES = 3
UA = {"User-Agent": "market-brief-feed/1.0 (+github actions)"}

RU_TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS",
              "MGNT", "PLZL", "CHMF", "ALRS", "YDEX", "VTBR", "AFLT"]
US_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
              "AVGO", "JPM", "XOM", "SPY", "QQQ", "IWM", "TLT"]
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                  "TONUSDT", "DOGEUSDT", "LINKUSDT"]

LOOKBACK_DAYS = 420  # хватает на SMA200 с запасом на выходные и праздники


# ---------------------------------------------------------------- индикаторы

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI по Уайлдеру: сглаживание EMA с alpha = 1/period."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0, 100.0)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def r2(x, nd=4):
    """Округление с честным null для отсутствующих значений."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, nd)


def summarize(df: pd.DataFrame, meta: dict) -> dict:
    """df: колонки date, open, high, low, close, volume — по возрастанию даты."""
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 30:
        return {**meta, "error": "мало свечей: %d" % len(df)}

    close = df["close"]
    macd_line, macd_sig, macd_hist = macd(close)
    last = len(df) - 1

    def at(series):
        return r2(series.iloc[last]) if last < len(series) else None

    sma20, sma50, sma200 = (close.rolling(n).mean() for n in (20, 50, 200))
    ema20 = close.ewm(span=20, adjust=False).mean()
    vol = df["volume"]
    vol_avg20 = vol.rolling(20).mean()

    price = float(close.iloc[last])
    prev = float(close.iloc[last - 1]) if last >= 1 else None

    # положение относительно скользящих — то, что модель чаще всего путает
    above = []
    for name, s in (("SMA20", sma20), ("SMA50", sma50), ("SMA200", sma200)):
        v = s.iloc[last]
        if v == v:
            above.append("%s %s" % ("выше" if price > float(v) else "ниже", name))

    win = df.tail(252)
    win20 = df.tail(20)

    return {
        **meta,
        "last_close": r2(price),
        "last_date": str(df["date"].iloc[last]),
        "prev_close": r2(prev),
        "change_pct": r2((price / prev - 1.0) * 100.0, 2) if prev else None,
        "candles_used": int(len(df)),
        "sma20": at(sma20), "sma50": at(sma50), "sma200": at(sma200),
        "ema20": at(ema20),
        "rsi14": r2(rsi(close).iloc[last], 2),
        "macd": r2(macd_line.iloc[last]),
        "macd_signal": r2(macd_sig.iloc[last]),
        "macd_hist": r2(macd_hist.iloc[last]),
        "atr14": r2(atr(df).iloc[last]),
        "volume": r2(vol.iloc[last], 0),
        "volume_avg20": r2(vol_avg20.iloc[last], 0),
        "high_20d": r2(win20["high"].max()),
        "low_20d": r2(win20["low"].min()),
        "high_52w": r2(win["high"].max()),
        "low_52w": r2(win["low"].min()),
        "position": ", ".join(above) if above else None,
    }


# ------------------------------------------------------------------ загрузка

def http_get(url, params=None, expect="json"):
    err = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json() if expect == "json" else r.text
        except Exception as exc:                      # noqa: BLE001
            err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("не удалось получить %s: %s" % (url, err))


def fetch_moex(ticker: str) -> pd.DataFrame:
    """Дневные свечи с MOEX ISS. Отдаёт максимум 500 строк за запрос."""
    start = (datetime.now(MSK) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    url = ("https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR"
           "/securities/%s/candles.json" % ticker)
    rows, cursor = [], 0
    while True:
        data = http_get(url, {"from": start, "interval": 24,
                              "start": cursor, "iss.meta": "off"})
        block = data.get("candles", {})
        cols = block.get("columns", [])
        chunk = block.get("data", [])
        if not chunk:
            break
        rows.extend(chunk)
        cursor += len(chunk)
        if len(chunk) < 500:
            break
    if not rows:
        raise RuntimeError("MOEX не вернул свечей по %s" % ticker)
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["begin"]).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


TINVEST_TOKEN = os.environ.get("TINVEST_TOKEN", "").strip()
TINVEST_HOSTS = ["https://invest-public-api.tbank.ru",
                 "https://invest-public-api.tinkoff.ru"]
TINVEST_PATH = "/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
_tinvest_host = None


def _quotation(q):
    """{'units': '271', 'nano': 480000000} -> 271.48. Отрицательные — корректно."""
    if q is None:
        return None
    units = int(q.get("units", 0) or 0)
    nano = int(q.get("nano", 0) or 0)
    return units + nano / 1e9


def fetch_tinvest(ticker: str) -> pd.DataFrame:
    """Дневные свечи из T-Invest API. instrumentId принимает 'ТИКЕР_КЛАСС'."""
    global _tinvest_host
    if not TINVEST_TOKEN:
        raise RuntimeError("TINVEST_TOKEN не задан")

    now = datetime.now(timezone.utc)
    body = {
        "instrumentId": "%s_TQBR" % ticker,
        "from": (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": "CANDLE_INTERVAL_DAY",
    }
    headers = {**UA, "Authorization": "Bearer %s" % TINVEST_TOKEN,
               "Content-Type": "application/json"}

    hosts = [_tinvest_host] if _tinvest_host else TINVEST_HOSTS
    last_err = None
    for host in hosts:
        for attempt in range(RETRIES):
            try:
                r = requests.post(host + TINVEST_PATH, json=body,
                                  headers=headers, timeout=HTTP_TIMEOUT)
                if r.status_code in (401, 403):
                    # Токен неверен или отозван — перебирать хосты бессмысленно.
                    raise RuntimeError("T-Invest отклонил токен (HTTP %d). "
                                       "Проверьте секрет TINVEST_TOKEN." % r.status_code)
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                _tinvest_host = host
                candles = r.json().get("candles", [])
                if not candles:
                    raise RuntimeError("T-Invest вернул пустой список свечей")
                rows = []
                for c in candles:
                    if c.get("isComplete") is False:
                        continue      # незакрытая дневная свеча искажает индикаторы
                    rows.append({
                        "date": pd.to_datetime(c["time"]).date(),
                        "open": _quotation(c.get("open")),
                        "high": _quotation(c.get("high")),
                        "low": _quotation(c.get("low")),
                        "close": _quotation(c.get("close")),
                        "volume": float(c.get("volume", 0) or 0),
                    })
                return pd.DataFrame(rows).sort_values("date")
            except RuntimeError:
                raise
            except Exception as exc:                  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("T-Invest недоступен: %s" % last_err)


def fetch_stooq(ticker: str) -> pd.DataFrame:
    text = http_get("https://stooq.com/q/d/l/",
                    {"s": "%s.us" % ticker.lower(), "i": "d"}, expect="text")
    if "Date" not in text[:200]:
        raise RuntimeError("stooq вернул не CSV по %s" % ticker)
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").tail(LOOKBACK_DAYS)
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_binance(symbol: str) -> pd.DataFrame:
    data = http_get("https://api.binance.com/api/v3/klines",
                    {"symbol": symbol, "interval": "1d", "limit": 500})
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qav", "trades", "tbbav", "tbqav", "ignore"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]]


# -------------------------------------------------------------------- сборка

def collect(market, items, fetcher, meta_fn):
    out, failures = [], []
    for item in items:
        try:
            df = fetcher(item)
            out.append(summarize(df, meta_fn(item)))
        except Exception as exc:                      # noqa: BLE001
            failures.append({"ticker": item, "market": market, "error": str(exc)[:300]})
            print("[warn] %s %s: %s" % (market, item, exc), file=sys.stderr)
        time.sleep(0.3)
    return out, failures


def main():
    now = datetime.now(MSK)

    # T-Invest предпочтительнее ISS: официальный источник брокера и не зависит от того,
    # откуда пришёл запрос. Без токена молча работаем на MOEX ISS.
    if TINVEST_TOKEN:
        ru_fetcher, ru_source = fetch_tinvest, "T-Invest API"
    else:
        ru_fetcher, ru_source = fetch_moex, "MOEX ISS"
    print("Источник по России: %s" % ru_source, file=sys.stderr)

    snapshot = {
        "generated_at_msk": now.strftime("%Y-%m-%d %H:%M"),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "timeframe": "1D",
        "note": ("Индикаторы посчитаны по фактическим дневным свечам библиотекой pandas, "
                 "а не взяты со стороннего сайта. RSI по Уайлдеру, MACD(12,26,9), "
                 "ATR(14) по Уайлдеру."),
        "sources": {"RU": ru_source, "US": "Stooq", "CRYPTO": "Binance"},
        "markets": {}, "failures": [],
    }

    ru, f1 = collect("RU", RU_TICKERS, ru_fetcher,
                     lambda t: {"market": "RU", "ticker": t, "exchange": "MOEX",
                                "currency": "RUB", "source": ru_source})
    us, f2 = collect("US", US_TICKERS, fetch_stooq,
                     lambda t: {"market": "US", "ticker": t, "exchange": "US",
                                "currency": "USD", "source": "Stooq"})
    cr, f3 = collect("CRYPTO", CRYPTO_SYMBOLS, fetch_binance,
                     lambda t: {"market": "CRYPTO", "ticker": t, "exchange": "Binance",
                                "currency": "USDT", "source": "Binance"})

    snapshot["markets"] = {"RU": ru, "US": us, "CRYPTO": cr}
    snapshot["failures"] = f1 + f2 + f3

    os.makedirs("data", exist_ok=True)
    with open("data/market_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)

    print(json.dumps({"RU": len(ru), "US": len(us), "CRYPTO": len(cr),
                      "failures": len(snapshot["failures"])}, ensure_ascii=False))

    # Целиком пустой рынок — это не «частичный сбой», а сломанный источник.
    # Роняем сборку, чтобы GitHub прислал письмо: молча отдавать сводке снимок
    # без России хуже, чем громко упасть.
    empty = [name for name, rows in (("RU", ru), ("US", us), ("CRYPTO", cr)) if not rows]
    if empty:
        for f in snapshot["failures"][:5]:
            print("  %s %s: %s" % (f["market"], f["ticker"], f["error"]), file=sys.stderr)
        sys.exit("не загрузился ни один инструмент по рынкам: %s. "
                 "Данные по остальным рынкам записаны и будут закоммичены."
                 % ", ".join(empty))


if __name__ == "__main__":
    main()
