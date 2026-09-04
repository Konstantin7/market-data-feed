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

Источники свечей:
  RU     — T-Invest API, если задан секрет TINVEST_TOKEN; иначе MOEX ISS
  US     — Yahoo Finance, с откатом на Stooq
  CRYPTO — Binance, с откатом на Coinbase и Kraken

Источники живой котировки (поле last_price, отдельно от закрытия свечи):
  RU     — блок marketdata MOEX ISS, вся доска TQBR одним запросом, без токена
  US     — regularMarketPrice из тех же метаданных Yahoo, отдельный запрос не нужен
  CRYPTO — тикер Coinbase

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
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/128.0.0.0 Safari/537.36"),
      "Accept": "*/*"}

RU_TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS",
              "MGNT", "PLZL", "CHMF", "ALRS", "YDEX", "VTBR", "AFLT"]
US_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
              "AVGO", "JPM", "XOM", "SPY", "QQQ", "IWM", "TLT"]
# Базовые активы. Пара под каждый источник собирается автоматически:
# Binance BTCUSDT, Coinbase BTC-USD, Kraken XBTUSD.
CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LINK", "AVAX", "LTC"]

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


# Какой источник фактически отдал данные по каждому инструменту.
# Заполняется каскадами и попадает в снимок, чтобы источник был виден построчно.
LAST_SOURCE = {}

# Живая котировка по инструменту: {"price": float, "ts": "...", "source": "..."}.
# Отдельно от last_close: закрытие свечи нужно индикаторам, живая цена — торговой карточке.
LAST_QUOTE = {}


def fetch_moex_marketdata() -> int:
    """Текущие котировки всей доски TQBR одним запросом. Токен не нужен.

    Блок marketdata отдаёт LAST, BID, OFFER и время обновления по каждой бумаге.
    Вне торговой сессии LAST пуст — тогда берём цену последней сделки дня
    (MARKETPRICETODAY), затем средневзвешенную (WAPRICE).
    """
    try:
        data = http_get("https://iss.moex.com/iss/engines/stock/markets/shares"
                        "/boards/TQBR/securities.json",
                        {"iss.only": "marketdata", "iss.meta": "off"})
    except Exception as exc:                          # noqa: BLE001
        print("[warn] MOEX marketdata недоступен: %s" % str(exc)[:200], file=sys.stderr)
        return 0
    block = (data or {}).get("marketdata") or {}
    cols = block.get("columns") or []
    rows = block.get("data") or []
    if not cols or not rows:
        return 0
    idx = {name: i for i, name in enumerate(cols)}
    need = ("SECID", "LAST", "MARKETPRICETODAY", "WAPRICE", "UPDATETIME", "SYSTIME")
    if "SECID" not in idx:
        return 0
    n = 0
    for row in rows:
        def val(name):
            i = idx.get(name)
            return row[i] if i is not None and i < len(row) else None
        secid = val("SECID")
        price = next((v for v in (val("LAST"), val("MARKETPRICETODAY"), val("WAPRICE"))
                      if isinstance(v, (int, float))), None)
        if not secid or price is None:
            continue
        LAST_QUOTE[secid] = {"price": r2(price),
                             "ts": str(val("SYSTIME") or val("UPDATETIME") or ""),
                             "source": "MOEX ISS marketdata"}
        n += 1
    print("MOEX marketdata: живых котировок %d" % n, file=sys.stderr)
    return n


def fetch_coinbase_ticker(asset: str):
    """Текущая цена спота Coinbase. Ошибка не критична — цена останется от свечи."""
    try:
        d = http_get("https://api.exchange.coinbase.com/products/%s-USD/ticker" % asset)
        price = float(d["price"])
        LAST_QUOTE[asset] = {"price": r2(price), "ts": str(d.get("time") or ""),
                             "source": "Coinbase ticker"}
    except Exception as exc:                          # noqa: BLE001
        print("[warn] Coinbase ticker %s: %s" % (asset, str(exc)[:120]), file=sys.stderr)


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    """Дневные свечи Yahoo Finance. Ключ не нужен, доступен с раннеров GitHub."""
    data = http_get("https://query1.finance.yahoo.com/v8/finance/chart/%s" % ticker,
                    {"range": "2y", "interval": "1d", "includePrePost": "false"})
    chart = (data or {}).get("chart") or {}
    if chart.get("error"):
        raise RuntimeError("Yahoo вернул ошибку: %s" % chart["error"])
    res = (chart.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        raise RuntimeError("Yahoo не вернул свечей по %s" % ticker)

    # Живая котировка лежит в метаданных того же ответа — отдельный запрос не нужен.
    meta = res.get("meta") or {}
    rmp = meta.get("regularMarketPrice")
    if isinstance(rmp, (int, float)):
        rmt = meta.get("regularMarketTime")
        LAST_QUOTE[ticker] = {
            "price": r2(rmp),
            "ts": (datetime.fromtimestamp(rmt, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                   if isinstance(rmt, (int, float)) else ""),
            "source": "Yahoo Finance regularMarketPrice",
        }

    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s", utc=True).date,
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"),
        "volume": q.get("volume"),
    })
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_stooq(ticker: str) -> pd.DataFrame:
    """Запасной источник по США. Stooq режет частые запросы и не любит роботов."""
    text = http_get("https://stooq.com/q/d/l/",
                    {"s": "%s.us" % ticker.lower(), "i": "d"}, expect="text")
    head = text[:200].lower()
    if "date" not in head:
        raise RuntimeError("stooq отдал не CSV, а «%s»" % text[:80].strip().replace("\n", " "))
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").tail(LOOKBACK_DAYS)
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_us(ticker: str) -> pd.DataFrame:
    """Каскад по США: Yahoo, затем Stooq. Первый ответивший выигрывает."""
    errors = []
    for name, fn in (("Yahoo", fetch_yahoo), ("Stooq", fetch_stooq)):
        try:
            df = fn(ticker)
            if len(df) >= 30:
                LAST_SOURCE[ticker] = name
                return df
            errors.append("%s: мало свечей (%d)" % (name, len(df)))
        except Exception as exc:                      # noqa: BLE001
            errors.append("%s: %s" % (name, str(exc)[:150]))
    raise RuntimeError(" | ".join(errors))


def fetch_binance(asset: str) -> pd.DataFrame:
    """Binance отдаёт 451 с американских адресов, а раннеры GitHub — американские."""
    data = http_get("https://api.binance.com/api/v3/klines",
                    {"symbol": "%sUSDT" % asset, "interval": "1d", "limit": 500})
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qav", "trades", "tbbav", "tbqav", "ignore"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_coinbase(asset: str) -> pd.DataFrame:
    """Coinbase Exchange: до 300 дневных свечей за запрос, ключ не нужен."""
    data = http_get("https://api.exchange.coinbase.com/products/%s-USD/candles" % asset,
                    {"granularity": 86400})
    if not isinstance(data, list) or not data:
        raise RuntimeError("Coinbase не вернул свечей по %s" % asset)
    # формат строки: [time, low, high, open, close, volume], по убыванию времени
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


KRAKEN_PAIR = {"BTC": "XBTUSD", "DOGE": "XDGUSD"}


def fetch_kraken(asset: str) -> pd.DataFrame:
    """Kraken: до 720 дневных свечей. У биткоина и доджа собственные тикеры."""
    pair = KRAKEN_PAIR.get(asset, "%sUSD" % asset)
    data = http_get("https://api.kraken.com/0/public/OHLC",
                    {"pair": pair, "interval": 1440})
    if data.get("error"):
        raise RuntimeError("Kraken: %s" % data["error"])
    result = {k: v for k, v in (data.get("result") or {}).items() if k != "last"}
    if not result:
        raise RuntimeError("Kraken не вернул свечей по %s" % asset)
    rows = next(iter(result.values()))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                     "vwap", "volume", "count"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


def fetch_crypto(asset: str) -> pd.DataFrame:
    """Каскад по крипте: Binance, затем Coinbase, затем Kraken."""
    errors = []
    for name, fn in (("Binance", fetch_binance),
                     ("Coinbase", fetch_coinbase),
                     ("Kraken", fetch_kraken)):
        try:
            df = fn(asset)
            if len(df) >= 30:
                LAST_SOURCE[asset] = name
                return df
            errors.append("%s: мало свечей (%d)" % (name, len(df)))
        except Exception as exc:                      # noqa: BLE001
            errors.append("%s: %s" % (name, str(exc)[:120]))
    raise RuntimeError(" | ".join(errors))


# -------------------------------------------------------------------- сборка

def quote_fields(key: str) -> dict:
    """Три поля живой котировки для строки снимка. Нет данных — честные null."""
    q = LAST_QUOTE.get(key) or {}
    return {"last_price": q.get("price"),
            "last_price_ts": q.get("ts") or None,
            "last_price_source": q.get("source") or None}


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
                 "ATR(14) по Уайлдеру. В каждой строке два ценовых поля: last_close — "
                 "закрытие последней дневной свечи, основа индикаторов; last_price — "
                 "живая котировка на момент снимка со своей меткой времени, её и надо "
                 "брать ценой входа в торговую карточку."),
        "sources": {"RU": ru_source,
                    "US": "Yahoo Finance / Stooq",
                    "CRYPTO": "Binance / Coinbase / Kraken"},
        "markets": {}, "failures": [],
    }

    # Живые котировки всей доски MOEX одним запросом, до обхода свечей.
    # Работает без токена и при включённом T-Invest тоже: биржа та же.
    fetch_moex_marketdata()

    def ru_meta(t):
        return {"market": "RU", "ticker": t, "exchange": "MOEX",
                "currency": "RUB", "source": ru_source, **quote_fields(t)}

    def us_meta(t):
        return {"market": "US", "ticker": t, "exchange": "US", "currency": "USD",
                "source": LAST_SOURCE.get(t, "н/д"), **quote_fields(t)}

    def crypto_meta(t):
        if LAST_SOURCE.get(t) in ("Coinbase", "Binance", "Kraken"):
            fetch_coinbase_ticker(t)
        return {"market": "CRYPTO", "ticker": t,
                "exchange": LAST_SOURCE.get(t, "н/д"),
                "currency": "USDT" if LAST_SOURCE.get(t) == "Binance" else "USD",
                "source": LAST_SOURCE.get(t, "н/д"), **quote_fields(t)}

    ru, f1 = collect("RU", RU_TICKERS, ru_fetcher, ru_meta)
    us, f2 = collect("US", US_TICKERS, fetch_us, us_meta)
    cr, f3 = collect("CRYPTO", CRYPTO_ASSETS, fetch_crypto, crypto_meta)

    snapshot["markets"] = {"RU": ru, "US": us, "CRYPTO": cr}
    snapshot["failures"] = f1 + f2 + f3
    used_us = sorted({r["source"] for r in us if r.get("source")})
    used_cr = sorted({r["source"] for r in cr if r.get("source")})
    if used_us:
        snapshot["sources"]["US"] = ", ".join(used_us)
    if used_cr:
        snapshot["sources"]["CRYPTO"] = ", ".join(used_cr)

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
