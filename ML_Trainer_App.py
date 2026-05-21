"""
ML Trainer & Trader App
═══════════════════════════════════════════════════════════════
Sve u jednoj aplikaciji:
  • Trenira XGBoost + RandomForest na MT5 istorijskim podacima
  • Generira live signale svaki novi bar
  • Otvara/zatvara trejdove direktno kroz MT5 Python API
  • Prati otvorene pozicije i P&L u realnom vremenu

Pokretanje:  python ML_Trainer_App.py   ili   ML_Trainer.bat
"""

import sys, os, threading, queue, json, warnings, time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkfont

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score

# ─── Putanje ──────────────────────────────────────────────────────────────────
# Kada radi kao PyInstaller exe, __file__ pokazuje na temp folder.
# sys.executable pokazuje na pravi exe u OmniTrader folderu.
if getattr(sys, 'frozen', False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769,
}
BARS_PER_MONTH = {
    "M1": 43800, "M5": 8760, "M15": 2880, "M30": 1440,
    "H1": 720,   "H4": 180,  "D1": 30,    "W1": 4,
}
# Minimum months of history required per TF to have enough training samples
TF_MIN_MONTHS = {"H1": 24, "H4": 36, "D1": 48, "W1": 60}
# Minimum bars required per TF after cutoff filtering
TF_MIN_BARS   = {"W1": 100, "D1": 200, "H4": 300, "H1": 500}
MAGIC = 771100


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS CALENDAR  (ForexFactory — besplatno, bez API ključa)
# ═══════════════════════════════════════════════════════════════════════════════

_NEWS_URL  = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_NEWS_CACHE: list  = []
_NEWS_LAST_FETCH   = None   # datetime utc
_NEWS_LOCK         = threading.Lock()

# Valute koje prate svaki simbol
_NEWS_CURRENCIES = {
    "EURUSD": ["EUR","USD"], "GBPUSD": ["GBP","USD"],
    "XAUUSD": ["USD"],       "USDJPY": ["USD","JPY"],
    "USDCHF": ["USD","CHF"], "AUDUSD": ["AUD","USD"],
    "NZDUSD": ["NZD","USD"], "USDCAD": ["USD","CAD"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI AI  — Sentiment analiza vijesti
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_KEY   = ""   # postavlja se iz GUI-a
_GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
_SENT_CACHE: dict = {}   # "{title}_{date}" -> {"usd":"bearish","eur":"bullish","str":0.8}
_SENT_LOCK   = threading.Lock()

def gemini_news_sentiment(title: str, country: str, actual: str = "",
                          forecast: str = "", previous: str = "",
                          log_q=None) -> dict:
    """Pošalji vijest Gemini-ju, dobij sentiment po valutama kao dict."""
    if not _GEMINI_KEY:
        return {}
    cache_key = f"{title}_{country}"
    with _SENT_LOCK:
        if cache_key in _SENT_CACHE:
            return _SENT_CACHE[cache_key]
    try:
        prompt = (
            f"Economic news event:\n"
            f"Title: {title}\nCountry: {country}\n"
            f"Actual: {actual or 'N/A'}  Forecast: {forecast or 'N/A'}  Previous: {previous or 'N/A'}\n\n"
            f"Analyze the impact on forex markets. Return ONLY valid JSON, no extra text:\n"
            f"{{\"direction\":\"bullish\"|\"bearish\"|\"neutral\","
            f"\"strength\":0.0-1.0,"
            f"\"reason\":\"one sentence\","
            f"\"currencies\":{{\"USD\":\"bullish\"|\"bearish\"|\"neutral\","
            f"\"EUR\":\"...\",\"GBP\":\"...\",\"XAU\":\"...\"}}}}"
        )
        payload = json.dumps({"contents":[{"parts":[{"text": prompt}]}]}).encode()
        url = _GEMINI_URL.format(key=_GEMINI_KEY)
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read().decode())
        text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Izvadi JSON iz odgovora
        start = text.find("{"); end = text.rfind("}") + 1
        result = json.loads(text[start:end]) if start >= 0 else {}
        with _SENT_LOCK:
            _SENT_CACHE[cache_key] = result
        if log_q and result:
            d = result.get("direction","?"); s = result.get("strength",0)
            log_q.put(("log",
                f"  🤖 Gemini: {title[:35]} → {d} (str={s:.1f}) | {result.get('reason','')}",
                "G" if d=="bullish" else ("R" if d=="bearish" else "M")))
        return result
    except Exception as ex:
        if log_q:
            log_q.put(("log", f"  🤖 Gemini greška: {ex}", "Y"))
        return {}

def get_news_sentiment(symbol: str, direction: str) -> bool:
    """Vrati True ako Gemini sentiment BLOKIRA ovaj smjer za simbol."""
    sym_to_cur = {"EURUSD":"EUR","GBPUSD":"GBP","XAUUSD":"XAU",
                  "USDJPY":"JPY","USDCHF":"CHF","AUDUSD":"AUD"}
    currency = sym_to_cur.get(symbol, "")
    with _SENT_LOCK:
        cached = dict(_SENT_CACHE)
    for sent in cached.values():
        if not isinstance(sent, dict): continue
        curs = sent.get("currencies", {})
        strength = float(sent.get("strength", 0))
        if strength < 0.6: continue   # ignorišemo slabe signale
        # Provjeri USD (uvijek relevantan) i specifičnu valutu simbola
        for cur in (["USD", currency] if currency else ["USD"]):
            cur_sent = curs.get(cur, "neutral")
            if cur == "USD":
                # USD bullish → SELL za EURUSD/GBPUSD/XAUUSD je OK, BUY nije
                if cur_sent == "bullish" and direction == "BUY": return True
                if cur_sent == "bearish" and direction == "SELL": return True
            else:
                if cur_sent == "bearish" and direction == "BUY": return True
                if cur_sent == "bullish" and direction == "SELL": return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM  — Notifikacije na telefon
# ═══════════════════════════════════════════════════════════════════════════════

_TG_TOKEN   = ""   # Bot token (BotFather)
_TG_CHAT_ID = ""   # Chat ID (tvoj user ID ili group ID)

def send_telegram(msg: str, log_q=None):
    """Pošalji poruku na Telegram (ne zahtijeva dodatne biblioteke — koristi urllib)."""
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return
    try:
        url  = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
        data = json.dumps({"chat_id": _TG_CHAT_ID, "text": msg,
                           "parse_mode": "HTML"}).encode()
        req  = urllib.request.Request(url, data=data,
                                      headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=6)
    except Exception as ex:
        if log_q:
            log_q.put(("log", f"  Telegram greška: {ex}", "Y"))


# ═══════════════════════════════════════════════════════════════════════════════
# KORELACIONE GRUPE  — Sprečavaju over-exposure u istom smjeru
# ═══════════════════════════════════════════════════════════════════════════════

# Simboli unutar grupe su visoko korelirani — ne otvaraj 2+ u istom smjeru
CORRELATED_GROUPS = [
    {"EURUSD", "AUDUSD", "NZDUSD"},    # anti-USD trio
    {"EURUSD", "EURGBP", "EURJPY"},    # EUR-bazni
    {"XAUUSD", "EURUSD"},              # oba padaju s USD-om
    {"AUDUSD", "NZDUSD"},              # antipodski par
]


def correlation_allows(symbol: str, direction: str, open_positions) -> tuple:
    """Vrati (True, "") ako korelacioni filter dozvoljava trejd,
    ili (False, "razlog") ako blokira."""
    open_syms_dir = {}  # symbol -> direction
    for p in open_positions:
        d = "BUY" if p.type == 0 else "SELL"
        open_syms_dir[p.symbol] = d
    for group in CORRELATED_GROUPS:
        if symbol not in group:
            continue
        for other_sym, other_dir in open_syms_dir.items():
            if other_sym in group and other_sym != symbol and other_dir == direction:
                return False, f"Korelacija: {other_sym} {other_dir} vec otvoren"
    return True, ""


def fetch_news_calendar(log_q=None) -> bool:
    """Preuzmi ForexFactory kalendar za tekuću sedmicu i keširaj u memoriju."""
    global _NEWS_CACHE, _NEWS_LAST_FETCH
    try:
        req = urllib.request.Request(
            _NEWS_URL,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        with _NEWS_LOCK:
            # Pre-parse dates to avoid repeated datetime.fromisoformat calls in the live loop
            for ev in data:
                ev['_dt_utc'] = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc).replace(tzinfo=None)
            _NEWS_CACHE      = data
            _NEWS_LAST_FETCH = datetime.utcnow()
        high_n = sum(1 for e in data if e.get("impact") == "High")
        if log_q:
            log_q.put(("log",
                f"📰 Vijesti učitane: {len(data)} eventa  |  "
                f"{high_n} × High impact ove sedmice", "M"))
        # Gemini analiza za High-impact vijesti (u pozadini)
        if _GEMINI_KEY:
            def _run_sentiment():
                now_utc = datetime.utcnow()
                for ev in data:
                    if ev.get("impact") != "High": continue
                    try:
                        diff = (ev['_dt_utc'] - now_utc).total_seconds() / 3600
                        if -2 <= diff <= 24:   # analiziraj samo vijesti unutar 24h
                            gemini_news_sentiment(
                                ev.get("title",""), ev.get("country",""),
                                ev.get("actual",""), ev.get("forecast",""),
                                ev.get("previous",""), log_q)
                            time.sleep(0.5)    # rate limit
                    except Exception: pass
            threading.Thread(target=_run_sentiment, daemon=True).start()
        return True
    except Exception as ex:
        if log_q:
            log_q.put(("log", f"📰 Vijesti: greška ({ex})", "Y"))
        return False

def is_news_time(symbol: str, before_min: int = 30, after_min: int = 30):
    """Vrati (True, opis) ako ima High-impakt vijest u prozoru ±min za ovaj simbol."""
    currencies = _NEWS_CURRENCIES.get(symbol, ["USD"])
    now_utc = datetime.utcnow()
    with _NEWS_LOCK:
        events = list(_NEWS_CACHE)
    for ev in events:
        if ev.get("impact") != "High":          # samo crvene vijesti
            continue
        if ev.get("country", "") not in currencies:
            continue
        try:
            diff = (ev['_dt_utc'] - now_utc).total_seconds() / 60
            if -after_min <= diff <= before_min:
                if diff >= 0:
                    desc = f"za {int(diff)}min"
                else:
                    desc = f"prije {int(abs(diff))}min"
                return True, f"{ev['title']} ({ev.get('country','')}, High) — {desc}"
        except Exception:
            continue
    return False, ""

def get_next_news(symbols: list) -> str:
    """Vrati string sa sljedećim High/Medium eventom u narednih 4h za listu simbola."""
    currencies = set()
    for sym in symbols:
        currencies.update(_NEWS_CURRENCIES.get(sym, ["USD"]))
    now_utc  = datetime.utcnow()
    upcoming = []
    with _NEWS_LOCK:
        events = list(_NEWS_CACHE)
    for ev in events:
        if ev.get("impact") not in ("High", "Medium"):
            continue
        if ev.get("country", "") not in currencies:
            continue
        try:
            ev_dt  = datetime.fromisoformat(ev["date"])
            ev_utc = ev_dt.astimezone(timezone.utc).replace(tzinfo=None)
            diff   = (ev_utc - now_utc).total_seconds() / 60
            if 0 < diff <= 240:
                upcoming.append((diff, ev))
        except Exception:
            continue
    if not upcoming:
        return "🟢 Nema vijesti (4h)"
    upcoming.sort(key=lambda x: x[0])
    diff, ev = upcoming[0]
    h, m = int(diff // 60), int(diff % 60)
    t_str = f"{h}h{m:02d}m" if h else f"{m}min"
    impact_icon = "🔴" if ev.get("impact") == "High" else "🟡"
    return f"{impact_icon} {ev['title']} ({ev.get('country','')}) za {t_str}"




# ═══════════════════════════════════════════════════════════════════════════════
# TRADING SESIJE  (UTC sati)
# ═══════════════════════════════════════════════════════════════════════════════

SESSIONS = {
    "Asian":   (0,  9),   # Tokyo/Sydney
    "London":  (7,  16),  # Frankfurt + London
    "NY":      (13, 22),  # New York
    "Overlap": (13, 16),  # London/NY overlap — najveća likvidnost
}

def get_session_name(utc_hour: int) -> str:
    """Vrati ime aktivne sesije za dati UTC sat."""
    if 13 <= utc_hour < 16: return "Overlap"   # London+NY
    if  7 <= utc_hour < 16: return "London"
    if 13 <= utc_hour < 22: return "NY"
    if  0 <= utc_hour <  9: return "Asian"
    return "Dead"

def get_session_icon(name: str) -> str:
    return {"London":"🇬🇧","NY":"🗽","Overlap":"⚡","Asian":"🌏","Dead":"💤"}.get(name,"")


# ═══════════════════════════════════════════════════════════════════════════════
# INDIKATORI
# ═══════════════════════════════════════════════════════════════════════════════

def _ema(s,n):   return s.ewm(span=n,adjust=False).mean()
def _sma(s,n):   return s.rolling(n).mean()
def _rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))
def _atr(h,l,c,n=14):
    h_l = h - l
    h_pc = (h - c.shift(1)).abs()
    l_pc = (l - c.shift(1)).abs()
    tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
    return pd.Series(tr, index=h.index).ewm(alpha=1/n, adjust=False).mean()
def _bb(s,n=20,dev=2.0):
    mid=_sma(s,n); std=s.rolling(n).std()
    up=mid+dev*std; lo=mid-dev*std
    pct=(s-lo)/(up-lo).replace(0,np.nan); wid=(up-lo)/mid.replace(0,np.nan)
    return mid,up,lo,pct.clip(0,1),wid
def _macd(s,fast=12,slow=26,sig=9):
    line=_ema(s,fast)-_ema(s,slow); signal=_ema(line,sig)
    return line,signal,line-signal
def _ao(h,l):
    mid=(h+l)/2; return _sma(mid,5)-_sma(mid,34)
def _adx(h,l,c,n=14):
    up=h-h.shift(1); dn=l.shift(1)-l
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    h_l = h - l
    h_pc = (h - c.shift(1)).abs()
    l_pc = (l - c.shift(1)).abs()
    tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
    atr_n=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=h.index).ewm(alpha=1/n,adjust=False).mean()/atr_n.replace(0,np.nan)
    mdi=100*pd.Series(mdm,index=h.index).ewm(alpha=1/n,adjust=False).mean()/atr_n.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi
def _stoch(h,l,c,k=5,d=3,sl=3):
    lo=l.rolling(k).min(); hi=h.rolling(k).max()
    rk=100*(c-lo)/(hi-lo).replace(0,np.nan)
    sk=_sma(rk,sl); sd=_sma(sk,d); return sk,sd


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES + LABELS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features(df, symbol=""):
    c,h,l,o,v=df["close"],df["high"],df["low"],df["open"],df["volume"]
    e9=_ema(c,9); e20=_ema(c,20); e21=_ema(c,21); e50=_ema(c,50)
    e100=_ema(c,100); e200=_ema(c,200)
    atr14=_atr(h,l,c,14); rsi14=_rsi(c,14); rsi7=_rsi(c,7)
    _,bb_up,bb_lo,bb_pct,bb_wid=_bb(c); macd_l,macd_s,macd_h=_macd(c)
    ao=_ao(h,l); adx,pdi,mdi=_adx(h,l,c); sk,sd=_stoch(h,l,c)
    a=atr14.replace(0,np.nan)
    f=pd.DataFrame(index=df.index)

    for nm,e in [("e9",e9),("e20",e20),("e21",e21),("e50",e50),("e100",e100),("e200",e200)]:
        f[f"{nm}_d"]=(c-e)/a
    f["e9_21"]=(e9-e21)/a; f["e20_50"]=(e20-e50)/a; f["e21_50"]=(e21-e50)/a
    f["e50_100"]=(e50-e100)/a; f["e50_200"]=(e50-e200)/a
    f["e200_sl"]=(e200-e200.shift(20))/a
    f["rsi14"]=rsi14/100; f["rsi7"]=rsi7/100
    f["rsi14p"]=rsi14.shift(1)/100; f["rsi_sl"]=(rsi14-rsi14.shift(3))/30
    _,bb_up2,bb_lo2,bb_pct2,bb_wid2=_bb(c)
    f["bb_pct"]=bb_pct2; f["bb_wid"]=bb_wid2
    f["bb_sqz"]=bb_wid2/bb_wid2.rolling(20).mean().replace(0,np.nan)
    f["macd_h"]=macd_h/a; f["macd_hp"]=macd_h.shift(1)/a
    f["macd_ab"]=(macd_l>macd_s).astype(int)
    f["macd_cu"]=((macd_l>macd_s)&(macd_l.shift(1)<=macd_s.shift(1))).astype(int)
    f["macd_cd"]=((macd_l<macd_s)&(macd_l.shift(1)>=macd_s.shift(1))).astype(int)
    f["ao_n"]=ao/a; f["ao_sg"]=np.sign(ao); f["ao_sl"]=(ao-ao.shift(2))/a
    f["adx"]=adx/100; f["pdi"]=pdi/100; f["mdi"]=mdi/100; f["di_df"]=(pdi-mdi)/100
    f["stk"]=sk/100; f["std_"]=sd/100; f["stk_df"]=(sk-sd)/100
    f["atr_n"]=a/c; f["atr_r"]=a/a.rolling(20).mean()
    f["bar_d"]=np.sign(c-o); f["body"]=(c-o).abs()/a
    f["uwk"]=(h - np.maximum(c, o))/a
    f["lwk"]=(np.minimum(c, o) - l)/a; f["rng"]=(h-l)/a
    f["ib"]=((h<h.shift(1))&(l>l.shift(1))).astype(int)
    f["ibbu"]=((h.shift(1)<h.shift(2))&(l.shift(1)>l.shift(2))&(c>h.shift(2))).astype(int)
    f["ibbd"]=((h.shift(1)<h.shift(2))&(l.shift(1)>l.shift(2))&(c<l.shift(2))).astype(int)
    f["fvg_b"]=(h.shift(2)<l).astype(int); f["fvg_r"]=(l.shift(2)>h).astype(int)
    sh5=h.shift(1).rolling(5).max(); sl5=l.shift(1).rolling(5).min()
    f["mss_b"]=(c>sh5).astype(int); f["mss_r"]=(c<sl5).astype(int)
    f["e50_tb"]=((c>e50)&(l<=e50)).astype(int); f["e50_ts"]=((c<e50)&(h>=e50)).astype(int)
    bwm=bb_wid2.rolling(20).mean(); f["bb_sq"]=(bb_wid2<0.7*bwm).astype(int)
    f["mom3"]=(c-c.shift(3))/a; f["mom10"]=(c-c.shift(10))/a; f["mom20"]=(c-c.shift(20))/a
    f["rng10"]=(h.rolling(10).max()-l.rolling(10).min())/a
    vma=v.rolling(20).mean().replace(0,np.nan)
    f["rvol"]=v/vma   # Relative Volume
    f["vol_ab"]=(v>vma).astype(int); f["vol_12"]=(v>1.2*vma).astype(int)
    atr50=_atr(h,l,c,50)
    f["atr_regime"]=a/atr50.replace(0,np.nan)  # >1=raste vol, <1=pada
    f["h1_trend_sl"]=(e200-e200.shift(96))/a   # EMA200 nagib ~24h (H1 trend proxy)
    try:    hour=pd.Series(df.index.hour,index=df.index)
    except: hour=pd.Series(pd.to_datetime(df.index).hour,index=df.index)
    try:    dow=pd.Series(df.index.dayofweek,index=df.index)
    except: dow=pd.Series(pd.to_datetime(df.index).dayofweek,index=df.index)

    # ── Sesije ────────────────────────────────────────────────────────────────
    f["asian"]   = ((hour>=0)&(hour<9)).astype(int)
    f["lon"]     = ((hour>=7)&(hour<16)).astype(int)
    f["ny"]      = ((hour>=13)&(hour<22)).astype(int)
    f["overlap"] = ((hour>=13)&(hour<16)).astype(int)   # London+NY — max likvidnost
    f["dead"]    = ((hour>=22)|(hour<7)).astype(int)    # mrtva zona

    # Blizina session open-a/close-a (povećana volatilnost)
    f["lon_open"]  = ((hour==7)|(hour==8)).astype(int)   # London open
    f["ny_open"]   = ((hour==13)|(hour==14)).astype(int) # NY open
    f["lon_close"] = ((hour==15)|(hour==16)).astype(int) # London close
    f["ny_close"]  = ((hour==21)|(hour==22)).astype(int) # NY close

    # Dan u sedmici (Pon=0, Pet=4) — Pon/Sri/Čet su statistički bolji
    f["dow_sin"] = np.sin(2*np.pi*dow/5)
    f["dow_cos"] = np.cos(2*np.pi*dow/5)
    f["is_mon"]  = (dow==0).astype(int)
    f["is_fri"]  = (dow==4).astype(int)   # Petak — oprez, reducirana aktivnost

    # Ciklični sat (neprekidan)
    f["h_s"]=np.sin(2*np.pi*hour/24); f["h_c"]=np.cos(2*np.pi*hour/24)

    buy=pd.Series(0,index=df.index); sell=pd.Series(0,index=df.index)
    buy +=((c>e50)&(c<bb_lo2)).astype(int); sell+=((c<e50)&(c>bb_up2)).astype(int)
    sell+=((rsi14>70)&(c<e50)&(c>bb_up2)).astype(int)
    buy +=((c>e50)&(macd_h>0)&(rsi14>40)&(ao>0)).astype(int)
    sell+=((c<e50)&(macd_h<0)&(rsi14<60)&(ao<0)).astype(int)
    buy +=((rsi14.shift(1)<35)&(rsi14>=35)&(e200>e200.shift(20))&(c>e200)&(c>e50)&(v>1.2*vma)&(adx>18)).astype(int)
    sell+=((rsi14.shift(1)>65)&(rsi14<=65)&(e200<e200.shift(20))&(c<e200)&(c<e50)&(v>1.2*vma)&(adx>18)).astype(int)
    buy +=((h.shift(1)<h.shift(2))&(l.shift(1)>l.shift(2))&(c>h.shift(2))&(c>e20)&(rsi14<65)).astype(int)
    sell+=((h.shift(1)<h.shift(2))&(l.shift(1)>l.shift(2))&(c<l.shift(2))&(c<e20)&(rsi14>35)).astype(int)
    buy +=((h.shift(2)<l)&(c>sh5)&(c>e200)).astype(int)
    sell+=((l.shift(2)>h)&(c<sl5)&(c<e200)).astype(int)
    buy +=((c>e50)&(l<=e50)&(v>vma)).astype(int); sell+=((c<e50)&(h>=e50)&(v>vma)).astype(int)
    scbu=(e20>e50)&(e20.shift(1)<=e50.shift(1)); scse=(e20<e50)&(e20.shift(1)>=e50.shift(1))
    buy +=(scbu&(adx>18)&(rsi14>28)&(rsi14<72)&(c>e200)).astype(int)
    sell+=(scse&(adx>18)&(rsi14>28)&(rsi14<72)&(c<e200)).astype(int)
    c21b=(e21>e50)&(e21.shift(1)<=e50.shift(1)); c21s=(e21<e50)&(e21.shift(1)>=e50.shift(1))
    buy +=(c21b&(rsi14>50)&(rsi14<72)&(adx>25)).astype(int)
    sell+=(c21s&(rsi14<50)&(rsi14>28)&(adx>25)).astype(int)
    buy +=((bb_wid2<0.7*bwm)&(c>bb_up2)&(rsi7<70)).astype(int)
    sell+=((bb_wid2<0.7*bwm)&(c<bb_lo2)&(rsi7>30)).astype(int)
    buy +=((c>e50)&(e50>e100)&(sk>sd)&(sk.shift(1)<=sd.shift(1))&(sk<0.5)).astype(int)
    sell+=((c<e50)&(e50<e100)&(sk<sd)&(sk.shift(1)>=sd.shift(1))&(sk>0.5)).astype(int)
    buy +=((e9>e21)&(e21>e50)&(e50>e200)).astype(int); sell+=((e9<e21)&(e21<e50)&(e50<e200)).astype(int)
    buy +=((e9>e21)&(e21>e50)).astype(int); sell+=((e9<e21)&(e21<e50)).astype(int)
    buy +=((e9>e21)&(e9.shift(1)<=e21.shift(1))).astype(int); sell+=((e9<e21)&(e9.shift(1)>=e21.shift(1))).astype(int)
    # Counter-trend RSI/BB votes — suppressed for XAUUSD when ADX > 25
    # Gold trades exclusively on breakout/trend; mean-reversion signals add noise
    if symbol == "XAUUSD":
        _allow_ct = ~(adx > 25)
        buy  += ((rsi14 < 35) & _allow_ct).astype(int)
        sell += ((rsi14 > 65) & _allow_ct).astype(int)
        buy  += ((rsi7  < 25) & _allow_ct).astype(int)
        sell += ((rsi7  > 75) & _allow_ct).astype(int)
        buy  += ((c < bb_lo2) & _allow_ct).astype(int)
        sell += ((c > bb_up2) & _allow_ct).astype(int)
    else:
        buy +=(rsi14<35).astype(int); sell+=(rsi14>65).astype(int)
        buy +=(rsi7<25).astype(int);  sell+=(rsi7>75).astype(int)
        buy +=(c<bb_lo2).astype(int); sell+=(c>bb_up2).astype(int)
    buy +=(macd_h>0).astype(int); sell+=(macd_h<0).astype(int)
    buy +=(ao>0).astype(int); sell+=(ao<0).astype(int)
    buy +=(c>e200).astype(int); sell+=(c<e200).astype(int)
    buy +=((adx>25)&(pdi>mdi)).astype(int); sell+=((adx>25)&(mdi>pdi)).astype(int)
    f["bv"]=buy; f["sv"]=sell; f["vbal"]=buy-sell
    f["vtot"]=buy+sell; f["vconv"]=(buy-sell)/(buy+sell+1)

    # ── Price Action Patterns ──────────────────────────────────────────────────
    body_raw = (c-o).abs()
    top_wick = h - np.maximum(c, o)
    bot_wick = np.minimum(c, o) - l
    rng_raw  = (h-l).replace(0,np.nan)

    # Gdje je close unutar svjeće (0=dno, 1=vrh) — bullish=blizu 1, bearish=blizu 0
    f["close_pos"]   = (c-l)/rng_raw

    # Pin bar (hammer/shooting star) — repovi dominiraju nad tijelom
    f["pin_bull"]    = ((bot_wick>2*body_raw)&(bot_wick>top_wick)).astype(int)
    f["pin_bear"]    = ((top_wick>2*body_raw)&(top_wick>bot_wick)).astype(int)

    # Engulfing — trenutna svjeća guta prethodnu
    f["eng_bull"]    = ((c>o)&(o<=c.shift(1))&(c>=o.shift(1))&(c>o.shift(1))).astype(int)
    f["eng_bear"]    = ((c<o)&(o>=c.shift(1))&(c<=o.shift(1))&(c<o.shift(1))).astype(int)

    # Uzastopni up/down barovi u zadnjih 5 (0.0–1.0)
    bar_up = (c>c.shift(1)).astype(float)
    bar_dn = (c<c.shift(1)).astype(float)
    f["consec_up"]   = bar_up.rolling(5).sum()/5
    f["consec_dn"]   = bar_dn.rolling(5).sum()/5

    # ── Swing High / Low rastojanje ────────────────────────────────────────────
    swh20 = h.shift(1).rolling(20).max()   # swing high zadnjih 20 barova
    swl20 = l.shift(1).rolling(20).min()   # swing low zadnjih 20 barova
    f["dist_swh"]    = (swh20-c)/a.replace(0,np.nan)   # >0 = ispod swing high
    f["dist_swl"]    = (c-swl20)/a.replace(0,np.nan)   # >0 = iznad swing low
    f["near_swh"]    = (f["dist_swh"]<1.0).astype(int) # blizu swing high (unutar 1 ATR)
    f["near_swl"]    = (f["dist_swl"]<1.0).astype(int) # blizu swing low

    # ── Volatility Context ─────────────────────────────────────────────────────
    f["vol_expand"]  = (a>a.shift(1)*1.2).astype(int)  # ATR raste >20%
    f["vol_contract"]= (a<a.shift(1)*0.8).astype(int)  # ATR pada >20%
    f["vol_spike"]   = (a>a.rolling(20).mean()*1.5).astype(int)  # spike volatilnosti

    # ── Multi-bar trend ────────────────────────────────────────────────────────
    f["trend3_up"]   = ((c>c.shift(1))&(c.shift(1)>c.shift(2))&(c.shift(2)>c.shift(3))).astype(int)
    f["trend3_dn"]   = ((c<c.shift(1))&(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))).astype(int)

    # ═══════════════════════════════════════════════════════════════════════════
    # PODRŠKA / OTPOR  +  JAČINA TRENDA
    # ═══════════════════════════════════════════════════════════════════════════

    an = a.replace(0, np.nan)   # ATR bez nula (za dijeljenje)

    # ── Donchian S/R nivoi (50 i 100 barova) ─────────────────────────────────
    sr_h50  = h.shift(1).rolling(50).max()    # 50-bar high  = otpor
    sr_l50  = l.shift(1).rolling(50).min()    # 50-bar low   = podrška
    sr_h100 = h.shift(1).rolling(100).max()   # 100-bar high = jači otpor
    sr_l100 = l.shift(1).rolling(100).min()   # 100-bar low  = jača podrška

    # Rastojanje do S/R u ATR jedinicama (negativno = probijen nivo)
    f["sr_dist_r50"]  = (sr_h50  - c) / an   # >0 ispod otpora, <0 iznad
    f["sr_dist_s50"]  = (c - sr_l50)  / an   # >0 iznad podrške, <0 ispod
    f["sr_dist_r100"] = (sr_h100 - c) / an
    f["sr_dist_s100"] = (c - sr_l100) / an

    # Blizina nivoa (unutar 0.5 ATR = opasna zona)
    f["near_res50"]   = (f["sr_dist_r50"].abs()  < 0.5).astype(int)
    f["near_sup50"]   = (f["sr_dist_s50"].abs()  < 0.5).astype(int)
    f["near_res100"]  = (f["sr_dist_r100"].abs() < 1.0).astype(int)
    f["near_sup100"]  = (f["sr_dist_s100"].abs() < 1.0).astype(int)

    # Proboj nivoa (cijena prešla iznad otpora / ispod podrške)
    f["break_res50"]  = ((c > sr_h50) & (c.shift(1) <= sr_h50.shift(1))).astype(int)
    f["break_sup50"]  = ((c < sr_l50) & (c.shift(1) >= sr_l50.shift(1))).astype(int)

    # Pozicija unutar 50-bar raspona (0=dno/podrška, 1=vrh/otpor)
    sr_range50 = (sr_h50 - sr_l50).replace(0, np.nan)
    f["sr_pos50"]     = (c - sr_l50) / sr_range50

    # Kompresija raspona (uži = konsolidacija, proboj stiže)
    f["sr_compress50"]= sr_range50 / (an * 50)   # mali = komprimovan

    # ── Okrugli brojevi (psihološki nivoi) ───────────────────────────────────
    # FX: svaki 0.0050 je ključan. Gold: svaki 50$
    round_step = np.where(c.values > 100, 50.0, 0.0050)
    dist_rnd   = c.values % round_step
    dist_rnd   = np.minimum(dist_rnd, round_step - dist_rnd)
    f["round_dist"]   = pd.Series(dist_rnd, index=df.index) / an  # mali = blizu okruglog
    f["near_round"]   = (f["round_dist"] < 0.3).astype(int)

    # ── Jačina trenda ─────────────────────────────────────────────────────────
    # Linearni nagib (brzi: end-start normalizovan po ATR)
    f["lr_slope20"]   = (c - c.shift(19)) / (19 * an)   # nagib 20 barova
    f["lr_slope50"]   = (c - c.shift(49)) / (49 * an)   # nagib 50 barova

    # EMA razmak — što su EMA dalje jedna od druge, trend je jači
    f["ema_spread"]   = (e9 - e200).abs() / an           # širok = jak trend

    # Savršen EMA poredak (9>21>50>200 ili obrnuto) = jak trend
    f["ema_bull_ali"] = ((e9>e21)&(e21>e50)&(e50>e200)).astype(int)
    f["ema_bear_ali"] = ((e9<e21)&(e21<e50)&(e50<e200)).astype(int)

    # Kompresija EMA-a (konvergiraju = trend slabi / breakout dolazi)
    f["ema_compress"] = ((e9-e21).abs()+(e21-e50).abs()+(e50-e200).abs()) / an

    # Trajnost trenda: % barova iznad/ispod EMA50 u zadnjih 20
    f["pct_above_e50"]= (c > e50).astype(float).rolling(20).mean()   # 1.0 = jaki uptrend
    f["pct_below_e50"]= (c < e50).astype(float).rolling(20).mean()   # 1.0 = jaki downtrend

    # Trend konzistencija: koliko su HH/LL u zadnjih 10 barova (market structure)
    f["hh_count"]     = (h > h.shift(1)).astype(float).rolling(10).sum() / 10
    f["ll_count"]     = (l < l.shift(1)).astype(float).rolling(10).sum() / 10
    f["trend_str"]    = (f["hh_count"] - f["ll_count"])   # +1=jak uptrend, -1=jak downtrend

    # ── Gold: neutralize RSI/BB features when ADX > 25 ────────────────────────
    # During trending moves XAUUSD ignores mean-reversion levels; pushing these
    # features to neutral prevents the model from learning false counter-trend edges.
    if symbol == "XAUUSD":
        _gold_trend = adx > 25
        if _gold_trend.any():
            f.loc[_gold_trend, "rsi14"]  = 0.5
            f.loc[_gold_trend, "rsi7"]   = 0.5
            f.loc[_gold_trend, "rsi14p"] = 0.5
            f.loc[_gold_trend, "rsi_sl"] = 0.0
            f.loc[_gold_trend, "bb_pct"] = 0.5
            f.loc[_gold_trend, "bb_sqz"] = 0

    return f


def generate_labels(df, n=12):
    """Symmetric Triple Barrier: TP=ATR×2.0 (isti za BUY i SELL), SL=ATR×1.5.
    - lbl=1: cijena dostigla +2ATR (BUY pobijedio) ili pala ispod -1.5ATR SL za SHORT
    - lbl=0: cijena dostigla -2ATR (SELL pobijedio) ili porasla iznad +1.5ATR SL za LONG
    - Simetričan: model uči i BUY i SELL podjednako dobro.
    Breakeven WR = 33.3% (RR ostaje 2:1 jer SL=1.5 → TP=2ATR daje ~1.33:1 ... koristimo live SL).
    Promjena: SL sada 1.5ATR da se poklapa s live ATR×SL=1.5 setingom."""
    c_arr=df["close"].values; h_arr=df["high"].values; l_arr=df["low"].values
    a_arr=_atr(df["high"],df["low"],df["close"],14).values
    lbl=np.full(len(df),np.nan)
    for i in range(len(df)-n):
        if np.isnan(a_arr[i]) or a_arr[i]==0: continue
        entry   = c_arr[i]
        atr     = a_arr[i]
        long_tp = entry + atr * 2.0   # BUY target
        long_sl = entry - atr * 1.5   # BUY stop (= SHORT TP * 0.75, simetričniji)
        shrt_tp = entry - atr * 2.0   # SELL target (simetričan BUY target)
        shrt_sl = entry + atr * 1.5   # SELL stop
        for j in range(i+1, i+n+1):
            h, l = h_arr[j], l_arr[j]
            if h >= long_tp: lbl[i] = 1.0; break   # BUY TP hit
            if l <= shrt_tp: lbl[i] = 0.0; break   # SELL TP hit (simetričan!)
            if l <= long_sl: lbl[i] = 0.0; break   # BUY SL hit
            if h >= shrt_sl: lbl[i] = 1.0; break   # SELL SL hit → BUY zona
    return pd.Series(lbl,index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _save_feedback(symbol, tf_name, feat_names, features, label, profit,
                   ts: str = None, ticket: int = 0):
    """Spremi ishod trejda kao trening uzorak za buduće učenje."""
    fb_path = MODELS_DIR / f"feedback_{symbol}_{tf_name}.csv"
    row = {fn: fv for fn, fv in zip(feat_names, features)}
    row["label"]  = int(label)
    row["profit"] = round(float(profit), 5)
    row["ts"]     = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["ticket"] = int(ticket)
    df_row = pd.DataFrame([row])
    # Provjeri kompatibilnost kolona — ako se promijenio broj featura, počni iznova
    if fb_path.exists():
        try:
            existing_cols = list(pd.read_csv(fb_path, nrows=0).columns)
            if existing_cols != list(df_row.columns):
                fb_path.unlink()  # Stari format — obrisi, počni svježe
        except Exception:
            fb_path.unlink()  # Oštećen fajl — obrisi
    df_row.to_csv(fb_path, mode='a', header=not fb_path.exists(), index=False)


def scan_mt5_history(mt5, log_q, days_back: int = 730):
    """
    Skenira KOMPLETNU MT5 historiju trejdova (MAGIC=771100).
    Za svaki zatvoreni trejd rekonstruiše features na entry baru
    i sprema ih u feedback CSV ako već nisu tamo.
    """
    from_dt = datetime.now() - timedelta(days=days_back)
    deals = mt5.history_deals_get(from_dt, datetime.now())
    if not deals:
        log_q.put(("log", "📚 Historija: nema trejdova s MAGIC=771100.", "M"))
        return 0, 0

    # Grupiši po position_id
    pos_map: dict = {}
    for d in deals:
        if d.magic != MAGIC:
            continue
        pos_map.setdefault(d.position_id, []).append(d)

    saved = skipped = wins = losses = 0
    total_profit = 0.0

    for pid, dl in pos_map.items():
        entry_d = next((d for d in dl if d.entry == 0), None)
        exit_d  = next((d for d in dl if d.entry == 1), None)
        if entry_d is None or exit_d is None:
            skipped += 1; continue

        # Parsuj TF iz komentara "ML:M15+H1:0.72"
        tf_name = None
        comment = entry_d.comment or ""
        if comment.startswith("ML:"):
            parts = comment.split(":")
            if len(parts) >= 2:
                tf_raw = parts[1].split("+")[0]
                if tf_raw in TF_MAP:
                    tf_name = tf_raw
        if tf_name is None:
            skipped += 1; continue

        symbol  = entry_d.symbol
        tf_id   = TF_MAP[tf_name]
        profit  = exit_d.profit + exit_d.commission + exit_d.swap
        label   = 1 if profit > 0 else 0
        entry_ts = datetime.fromtimestamp(entry_d.time).strftime("%Y-%m-%d %H:%M")

        # Provjeri duplikat po ticket-u
        fb_path = MODELS_DIR / f"feedback_{symbol}_{tf_name}.csv"
        if fb_path.exists():
            try:
                existing = pd.read_csv(fb_path, usecols=["ticket"], on_bad_lines='skip')
                if int(pid) in existing["ticket"].values:
                    if profit > 0: wins += 1
                    else:          losses += 1
                    total_profit += profit
                    continue  # već evidentirano
            except: pass

        # Rekonstruiši features na entry baru
        try:
            entry_time = datetime.fromtimestamp(entry_d.time)
            bars = mt5.copy_rates_from(symbol, tf_id, entry_time, 500)
            if bars is None or len(bars) < 50:
                skipped += 1; continue
            df_h = pd.DataFrame(bars)
            df_h["time"] = pd.to_datetime(df_h["time"], unit="s")
            df_h.set_index("time", inplace=True)
            df_h.rename(columns={"tick_volume": "volume"}, inplace=True)
            df_h = df_h[["open", "high", "low", "close", "volume"]]
            feats = compute_features(df_h, symbol=symbol)
            row   = feats.iloc[-1]
            if row.isnull().any():
                skipped += 1; continue
            _save_feedback(symbol, tf_name, feats.columns.tolist(),
                           row.values.tolist(), label, profit,
                           ts=entry_ts, ticket=int(pid))
            saved += 1
            if profit > 0: wins += 1
            else:          losses += 1
            total_profit += profit
        except Exception:
            skipped += 1

    total = wins + losses
    wr    = wins / total * 100 if total > 0 else 0.0
    log_q.put(("log",
        f"📚 Historija: +{saved} novih uzoraka | ukupno {total} pozicija | "
        f"WR={wr:.1f}% ({wins}W/{losses}L) | "
        f"Total P&L: {total_profit:+.2f} | preskočeno={skipped}", "G"))
    return saved, total


def mt5_bars(mt5, symbol, tf_id, count):
    rates=mt5.copy_rates_from_pos(symbol,tf_id,0,count)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df=pd.DataFrame(rates)
    df["time"]=pd.to_datetime(df["time"],unit="s")
    df.set_index("time",inplace=True)
    df.rename(columns={"tick_volume":"volume"},inplace=True)
    return df[["open","high","low","close","volume"]]


def calc_lot(mt5, symbol, sl_dist, risk_pct, max_lot=2.0):
    balance  = mt5.account_info().balance
    risk_amt = balance * risk_pct / 100.0
    tick_val = mt5.symbol_info(symbol).trade_tick_value
    tick_sz  = mt5.symbol_info(symbol).trade_tick_size
    if sl_dist<=0 or tick_val<=0: return mt5.symbol_info(symbol).volume_min
    lot = risk_amt / (sl_dist / tick_sz * tick_val)
    step= mt5.symbol_info(symbol).volume_step
    mn  = mt5.symbol_info(symbol).volume_min
    mx  = min(max_lot, mt5.symbol_info(symbol).volume_max)
    lot = round(lot // step * step, 8)
    return max(mn, min(mx, lot))


def open_trade(mt5, symbol, direction, sl_dist, tp_dist, risk_pct, max_lot, comment="ML"):
    import MetaTrader5 as _mt5
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return None, "Nema tick podataka"

    sym_info  = mt5.symbol_info(symbol)
    digits    = sym_info.digits
    point     = sym_info.point
    spread    = tick.ask - tick.bid  # realni spread

    # Minimalni stop: max(broker stops_level, 3×spread, 10×point)
    stops_lvl = sym_info.trade_stops_level
    min_dist  = max((stops_lvl + 5) * point, spread * 3.0, point * 10)

    rr_ratio  = tp_dist / max(sl_dist, 1e-10)  # zadrži RR
    if sl_dist < min_dist:
        sl_dist = min_dist * 1.5
    tp_dist = max(sl_dist * rr_ratio, min_dist * 1.5)

    lot   = calc_lot(mt5, symbol, sl_dist, risk_pct, max_lot)
    price = tick.ask if direction == "BUY" else tick.bid
    otype = _mt5.ORDER_TYPE_BUY if direction == "BUY" else _mt5.ORDER_TYPE_SELL

    # Provjeri slobodan margin — smanji lot ako nema dovoljno
    acc = mt5.account_info()
    if acc:
        req_margin = mt5.order_calc_margin(otype, symbol, lot, price) or 0.0
        if req_margin > acc.margin_free * 0.95:
            mn_lot = sym_info.volume_min
            lot    = mn_lot  # pokušaj s minimalnim lotom
            req_margin = mt5.order_calc_margin(otype, symbol, lot, price) or 0.0
            if req_margin > acc.margin_free * 0.95:
                return None, f"Nema margina (free={acc.margin_free:.2f}, req≈{req_margin:.2f})"

    if direction == "BUY":
        sl = round(price - sl_dist, digits)
        tp = round(price + tp_dist, digits)
    else:
        sl = round(price + sl_dist, digits)
        tp = round(price - tp_dist, digits)

    req = {
        "action":   _mt5.TRADE_ACTION_DEAL,
        "symbol":   symbol,
        "volume":   lot,
        "type":     otype,
        "price":    price,
        "sl":       sl,
        "tp":       tp,
        "deviation":20,
        "magic":    MAGIC,
        "comment":  comment,
        "type_filling": _mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None:
        return None, f"order_send None: {mt5.last_error()}"
    if res.retcode != _mt5.TRADE_RETCODE_DONE:
        return None, f"retcode={res.retcode}"
    return res.order, None


def close_position(mt5, ticket):
    import MetaTrader5 as _mt5
    pos = None
    for p in (mt5.positions_get() or []):
        if p.ticket == ticket:
            pos = p; break
    if pos is None: return False, "Pozicija ne postoji"

    tick = mt5.symbol_info_tick(pos.symbol)
    price= tick.bid if pos.type==0 else tick.ask
    otype= _mt5.ORDER_TYPE_SELL if pos.type==0 else _mt5.ORDER_TYPE_BUY

    req = {
        "action":   _mt5.TRADE_ACTION_DEAL,
        "symbol":   pos.symbol,
        "volume":   pos.volume,
        "type":     otype,
        "position": ticket,
        "price":    price,
        "deviation":20,
        "magic":    MAGIC,
        "comment":  "ML:close",
        "type_filling": _mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == _mt5.TRADE_RETCODE_DONE:
        return True, None
    return False, f"retcode={res.retcode if res else 'None'}"


def get_positions(mt5):
    all_pos = mt5.positions_get() or []
    return [p for p in all_pos if p.magic == MAGIC]


_partial_closed: set = set()   # ticketi gdje je već urađen partial close

def manage_positions(mt5, positions, atr_sl_mult: float, log_q,
                     be_enabled=True, trail_enabled=True, partial_enabled=True):
    """
    Upravljanje otvorenim pozicijama:
      • Break-even: kad profit >= 1×ATR, pomjeri SL na entry (nulti rizik)
      • Trailing SL: kad profit >= 2×ATR, prati cijenu s razmakom 1×ATR
      • Partial close: kad profit >= 1×ATR, zatvori 50% (samo jednom po trejdu)
    """
    import MetaTrader5 as _mt5
    for p in positions:
        try:
            # Izračunaj trenutni ATR za simbol
            bars = mt5.copy_rates_from_pos(p.symbol, 16385, 0, 20)  # H1
            if bars is None or len(bars) < 14:
                continue
            df_b  = pd.DataFrame(bars)
            atr_v = float(_atr(pd.Series(df_b["high"]), pd.Series(df_b["low"]),
                               pd.Series(df_b["close"]), 14).iloc[-1])
            if atr_v <= 0:
                continue

            is_buy    = (p.type == 0)
            entry     = p.price_open
            tick      = mt5.symbol_info_tick(p.symbol)
            if tick is None: continue
            cur_price = tick.bid if is_buy else tick.ask
            digits    = mt5.symbol_info(p.symbol).digits
            profit_pts = (cur_price - entry) if is_buy else (entry - cur_price)
            atr_step   = atr_v * atr_sl_mult   # 1 ATR korak

            new_sl = p.sl
            action_log = []

            # ── Partial close: zatvori 50% na +1 ATR profit ─────────────────
            if partial_enabled and p.ticket not in _partial_closed:
                if profit_pts >= atr_v * 1.0:
                    min_vol = mt5.symbol_info(p.symbol).volume_min
                    close_vol = round(p.volume / 2.0, 2)
                    if close_vol >= min_vol:
                        price_c = tick.bid if is_buy else tick.ask
                        otype_c = _mt5.ORDER_TYPE_SELL if is_buy else _mt5.ORDER_TYPE_BUY
                        req_c = {"action": _mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                                 "volume": close_vol, "type": otype_c, "position": p.ticket,
                                 "price": price_c, "deviation": 20, "magic": MAGIC,
                                 "comment": "ML:partial", "type_filling": _mt5.ORDER_FILLING_IOC}
                        res_c = mt5.order_send(req_c)
                        if res_c and res_c.retcode == _mt5.TRADE_RETCODE_DONE:
                            _partial_closed.add(p.ticket)
                            action_log.append(f"Partial close {close_vol:.2f} lot")
                            send_telegram(
                                f"<b>PARTIAL CLOSE</b> {p.symbol}\n"
                                f"Zatvoreno {close_vol:.2f} lot @ {price_c}\n"
                                f"Profit do sad: {p.profit:+.2f} EUR", log_q)

            # ── Break-even: pomjeri SL na entry kad profit >= 1×ATR ─────────
            if be_enabled and profit_pts >= atr_v * 1.0:
                be_sl = round(entry, digits)
                if is_buy and (p.sl < be_sl - atr_v * 0.1):
                    new_sl = be_sl
                    action_log.append(f"Break-even SL → {new_sl}")
                elif not is_buy and (p.sl > be_sl + atr_v * 0.1 or p.sl == 0):
                    new_sl = be_sl
                    action_log.append(f"Break-even SL → {new_sl}")

            # ── Trailing SL: prati cijenu kad profit >= 2×ATR ───────────────
            if trail_enabled and profit_pts >= atr_v * 2.0:
                if is_buy:
                    trail_sl = round(cur_price - atr_step, digits)
                    if trail_sl > new_sl + atr_v * 0.1:
                        new_sl = trail_sl
                        action_log.append(f"Trail SL → {new_sl}")
                else:
                    trail_sl = round(cur_price + atr_step, digits)
                    if trail_sl < new_sl - atr_v * 0.1 or new_sl == 0:
                        new_sl = trail_sl
                        action_log.append(f"Trail SL → {new_sl}")

            # ── Pošalji SL izmjenu MT5-u ─────────────────────────────────────
            if action_log and new_sl != p.sl and new_sl > 0:
                req_sl = {"action": _mt5.TRADE_ACTION_SLTP, "position": p.ticket,
                          "symbol": p.symbol, "sl": new_sl, "tp": p.tp}
                res_sl = mt5.order_send(req_sl)
                ok = res_sl and res_sl.retcode == _mt5.TRADE_RETCODE_DONE
                log_q.put(("log",
                    f"  SL mgmt {p.symbol} #{p.ticket}: "
                    f"{' | '.join(action_log)} {'OK' if ok else 'GREŠKA'}",
                    "G" if ok else "Y"))

        except Exception as mgmt_e:
            log_q.put(("log", f"  manage_positions greška: {mgmt_e}", "Y"))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE — učitaj jednom, osvježi samo kad se fajl promijeni
# ═══════════════════════════════════════════════════════════════════════════════

_model_cache: dict = {}  # key -> {xgb, rf, lgb, mlp, feats, mtime}

def _load_models(key):
    xp = MODELS_DIR/f"xgb_{key}.pkl"
    rp = MODELS_DIR/f"rf_{key}.pkl"
    fp = MODELS_DIR/f"features_{key}.json"
    if not (xp.exists() and rp.exists() and fp.exists()):
        return None, None, None, None, None
    lp  = MODELS_DIR/f"lgb_{key}.pkl"
    mp  = MODELS_DIR/f"mlp_{key}.pkl"
    mtime = max(xp.stat().st_mtime, rp.stat().st_mtime)
    cached = _model_cache.get(key)
    if cached and cached["mtime"] >= mtime:
        return cached["xgb"], cached["rf"], cached.get("lgb"), cached.get("mlp"), cached["feats"]
    xgb_m = joblib.load(xp); rf_m = joblib.load(rp)
    lgb_m = joblib.load(lp) if lp.exists() else None
    mlp_m = joblib.load(mp) if mp.exists() else None
    with open(fp) as f: feat_names = json.load(f)
    _model_cache[key] = {"xgb": xgb_m, "rf": rf_m, "lgb": lgb_m,
                         "mlp": mlp_m, "feats": feat_names, "mtime": mtime}
    return xgb_m, rf_m, lgb_m, mlp_m, feat_names


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL + TRADE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def dynamic_confidence(adx_value: float) -> float:
    """Dinamički prag confidence-a baziran na ADX:
    - Jak trend (ADX>30): prag 0.60 — model je sigurniji u trendu
    - Umjeren trend (ADX 20-30): prag 0.65
    - Ranging (ADX<20): prag 0.72 — strogi filter, malo signala ali kvalitetniji
    """
    if adx_value > 30.0: return 0.60
    if adx_value > 20.0: return 0.65
    return 0.72


def get_signal(mt5, symbol, tf_name, tf_id, conf_thr, last_bars, log_q):
    """Čisti signal detektor — vraća signal dict (uključuje feat_names i features za feedback)."""
    key = f"{symbol}_{tf_name}"

    bars = mt5.copy_rates_from_pos(symbol, tf_id, 0, 2)
    if bars is None or len(bars) < 2: return None
    bar_t = int(bars[-2]["time"])
    if last_bars.get(key) == bar_t: return None
    last_bars[key] = bar_t

    xgb_m, rf_m, lgb_m, mlp_m, feat_names = _load_models(key)
    if xgb_m is None: return None

    df = mt5_bars(mt5, symbol, tf_id, 500)
    if df.empty or len(df) < 100: return None

    feats = compute_features(df, symbol=symbol)
    miss = [c for c in feat_names if c not in feats.columns]
    if miss: return None
    row = feats[feat_names].iloc[-2]
    if row.isnull().any(): return None

    X  = row.values.reshape(1,-1)
    adx_raw = float(row.get("adx", 0.0)) * 100  # feature je /100

    # ── Pokušaj koristiti regime-specifičan XGB model ────────────────────────
    xgb_for_signal = xgb_m
    rp_trend = MODELS_DIR/f"xgb_trend_{key}.pkl"
    rp_range = MODELS_DIR/f"xgb_range_{key}.pkl"
    try:
        if adx_raw > 25 and rp_trend.exists():
            _rm = joblib.load(rp_trend)
            if _rm is not None: xgb_for_signal = _rm
        elif adx_raw <= 20 and rp_range.exists():
            _rm = joblib.load(rp_range)
            if _rm is not None: xgb_for_signal = _rm
    except Exception: pass

    # ── Ensemble: XGB + RF + LGB (ako postoji) + MLP (ako postoji) ──────────
    probs = [xgb_for_signal.predict_proba(X)[0], rf_m.predict_proba(X)[0]]
    if lgb_m is not None:
        probs.append(lgb_m.predict_proba(X)[0])
    if mlp_m is not None:
        probs.append(mlp_m.predict_proba(X)[0])
    pe = np.mean(probs, axis=0)

    # ── Meta-learner override (ako je dostupan i bolji) ──────────────────────
    meta_path = MODELS_DIR/f"meta_lr_{key}.pkl"
    if meta_path.exists():
        try:
            meta_lr, scaler_meta = joblib.load(meta_path)
            stack = np.array([[probs[i][1] for i in range(len(probs))]])
            # Dopuni ako meta očekuje više kolona
            while stack.shape[1] < scaler_meta.n_features_in_:
                stack = np.hstack([stack, stack[:, -1:]])
            stack = stack[:, :scaler_meta.n_features_in_]
            stack_s = scaler_meta.transform(stack)
            meta_prob = meta_lr.predict_proba(stack_s)[0]
            pe = meta_prob   # [P(0), P(1)]
        except Exception: pass

    pb, ps = float(pe[1]), float(pe[0])

    bv    = int(feats["bv"].iloc[-2]); sv = int(feats["sv"].iloc[-2])
    rsi_v = round(float(feats["rsi14"].iloc[-2]*100),1)
    adx_v = round(float(feats["adx"].iloc[-2]*100),1)
    atr_v = float(_atr(df["high"],df["low"],df["close"],14).iloc[-2])

    # Dinamički prag: GUI conf_thr je minimalni floor, dynamic_confidence je baza
    # max() osigurava da GUI slider uvijek bude poštovan
    eff_thr = max(dynamic_confidence(adx_v), conf_thr)  # GUI 0.80 → mora biti ≥0.80
    if pb > eff_thr:    sig, conf = "BUY",  pb
    elif ps > eff_thr:  sig, conf = "SELL", ps
    else:               sig, conf = "HOLD", max(pb,ps)

    # ── News filter — blokiraj signal ako je High-impakt vijest unutar ±30min ──
    news_note = ""
    if sig != "HOLD" and _NEWS_CACHE:
        news_hit, news_desc = is_news_time(symbol, before_min=30, after_min=30)
        if news_hit:
            sig, conf = "HOLD", max(pb, ps)
            news_note = f"  📰 {news_desc}"

    # ── Gemini sentiment filter — blokiraj samo kontra-sentiment smjer ──────
    if sig != "HOLD" and _GEMINI_KEY and _SENT_CACHE:
        if get_news_sentiment(symbol, sig):
            news_note += "  🤖 Gemini: kontra-sentiment"
            sig, conf = "HOLD", max(pb, ps)

    icon = "▲" if sig=="BUY" else ("▼" if sig=="SELL" else "─")
    clr  = "G" if sig=="BUY" else ("R" if sig=="SELL" else ("Y" if news_note else "M"))
    log_q.put(("log",
        f"{icon} {symbol:8s} {tf_name:4s}  {sig:4s}  conf={conf:.1%}  "
        f"thr={eff_thr:.0%}  bv={bv} sv={sv}  rsi={rsi_v}  adx={adx_v}{news_note}", clr))

    vconv_v = round(float(feats["vconv"].iloc[-2]), 4) if "vconv" in feats.columns else 0.0

    return {
        "symbol": symbol, "tf": tf_name, "signal": sig,
        "confidence": round(conf,4), "p_buy": round(pb,4), "p_sell": round(ps,4),
        "bv": bv, "sv": sv, "rsi": rsi_v, "adx": adx_v, "atr": atr_v,
        "vconv": vconv_v,
        "ts": datetime.utcnow().strftime("%H:%M:%S"),
        "feat_names": feat_names,
        "features": row.values.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════════

class App:
    BG="#0d1117"; PNL="#161b22"; BDR="#30363d"
    ACC="#58a6ff"; GRN="#3fb950"; RED="#f85149"
    YLW="#d29922"; TXT="#c9d1d9"; MUT="#8b949e"
    FN=("Consolas",10); FB=("Consolas",11,"bold")
    FS=("Consolas",9);  FL=("Consolas",14,"bold")

    def __init__(self):
        self.root=tk.Tk()
        self.root.title("ML Trader  —  XGBoost + RandomForest  |  Live Trading")
        self.root.geometry("1380x900"); self.root.minsize(1100,700)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW",self._on_close)

        self._q=queue.Queue()
        self._mt5=None
        self._training=False
        self._live_run=False
        self._sym_vars={}; self._tf_vars={}
        self._results={}
        self._last_bars={}
        self._live_thread=None
        self._daily_trades=0
        self._day_start_bal=0.0
        self._last_day=""
        self._pending_feedback={}  # ticket -> {symbol, tf, direction, feat_names, features}
        self._prev_tickets=set()   # otvoreni tiketi prethodne iteracije
        self._live_warmup=False    # prvi scan samo promatra, ne trguje
        self._session_start=""
        self._session_bal_start=0.0
        self._sess_vars={}   # {"London": BooleanVar, ...}

        self._build_ui()
        self._poll()
        self._connect_mt5()

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_header()
        nb=ttk.Notebook(self.root)
        nb.pack(fill="both",expand=True,padx=10,pady=(4,0))

        style=ttk.Style(); style.theme_use("clam")
        style.configure("TNotebook",background=self.BG,borderwidth=0)
        style.configure("TNotebook.Tab",background=self.PNL,foreground=self.MUT,
                        font=self.FN,padding=(14,6))
        style.map("TNotebook.Tab",
                  background=[("selected",self.BG)],
                  foreground=[("selected",self.ACC)])

        self._tab_train  = tk.Frame(nb,bg=self.BG)
        self._tab_live   = tk.Frame(nb,bg=self.BG)
        self._tab_pos    = tk.Frame(nb,bg=self.BG)
        self._tab_log    = tk.Frame(nb,bg=self.BG)

        nb.add(self._tab_train, text="  ▶  TRENING  ")
        nb.add(self._tab_live,  text="  ⚡  LIVE TRADING  ")
        nb.add(self._tab_pos,   text="  📊  POZICIJE  ")
        nb.add(self._tab_log,   text="  📋  LOG  ")
        self._nb=nb

        self._build_train_tab()
        self._build_live_tab()
        self._build_pos_tab()
        self._build_log_tab()
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        h=tk.Frame(self.root,bg=self.BG); h.pack(fill="x",padx=12,pady=(10,4))
        tk.Label(h,text="ML",fg=self.ACC,bg=self.BG,
                 font=("Consolas",20,"bold")).pack(side="left")
        tk.Label(h,text=" TRADER",fg=self.GRN,bg=self.BG,
                 font=("Consolas",20,"bold")).pack(side="left")
        tk.Label(h,text="  XGBoost + RandomForest  |  Sve strategije  |  Live Trading",
                 fg=self.MUT,bg=self.BG,font=self.FN).pack(side="left",padx=10)

        self._lbl_mt5=tk.Label(h,text="● OFFLINE",fg=self.RED,bg=self.BG,font=self.FB)
        self._lbl_mt5.pack(side="right",padx=8)
        self._lbl_bal=tk.Label(h,text="",fg=self.TXT,bg=self.BG,font=self.FN)
        self._lbl_bal.pack(side="right",padx=12)
        self._lbl_clk=tk.Label(h,text="",fg=self.MUT,bg=self.BG,font=self.FS)
        self._lbl_clk.pack(side="right",padx=8)
        self._tick_clock()
        ttk.Separator(self.root).pack(fill="x",padx=10)

    # ── TAB: TRENING ──────────────────────────────────────────────────────────
    def _build_train_tab(self):
        t=self._tab_train
        t.columnconfigure(1,weight=1); t.rowconfigure(0,weight=1)

        left=tk.Frame(t,bg=self.PNL,width=290)
        left.grid(row=0,column=0,sticky="nsew",padx=(8,6),pady=8)
        left.pack_propagate(False)

        # Simboli
        self._sec(left,"SIMBOLI")
        sg=tk.Frame(left,bg=self.PNL); sg.pack(fill="x",padx=10,pady=(0,4))
        syms=["EURUSD","GBPUSD","XAUUSD","USDJPY","USDCHF",
              "AUDUSD","NZDUSD","USDCAD","EURJPY","GBPJPY",
              "EURGBP","GBPCHF","XAGUSD","US30","NAS100"]
        # GBPUSD i USDJPY su defaultno isključeni (live analiza: WR 24%/22%)
        _GOOD_SYMS = {"EURUSD","XAUUSD","AUDUSD","EURGBP","NZDUSD","EURJPY"}
        for i,s in enumerate(syms):
            v=tk.BooleanVar(value=s in _GOOD_SYMS)
            self._sym_vars[s]=v
            tk.Checkbutton(sg,text=s,variable=v,bg=self.PNL,fg=self.TXT,
                           selectcolor=self.BG,activebackground=self.PNL,
                           font=self.FS,cursor="hand2").grid(
                row=i//2,column=i%2,sticky="w",padx=4,pady=1)
        bf=tk.Frame(left,bg=self.PNL); bf.pack(fill="x",padx=10,pady=(0,6))
        self._sbtn(bf,"Sve",lambda:self._sym_all(True)).pack(side="left",padx=2)
        self._sbtn(bf,"Nijedna",lambda:self._sym_all(False)).pack(side="left",padx=2)

        ttk.Separator(left).pack(fill="x",padx=8,pady=4)

        # Timeframeovi
        self._sec(left,"TIMEFRAMEOVI")
        tf=tk.Frame(left,bg=self.PNL); tf.pack(fill="x",padx=10,pady=(0,4))
        for i,nm in enumerate(TF_MAP.keys()):
            v=tk.BooleanVar(value=nm in ["M5","M15","M30","H1"])
            self._tf_vars[nm]=v
            tk.Checkbutton(tf,text=nm,variable=v,bg=self.PNL,fg=self.TXT,
                           selectcolor=self.BG,activebackground=self.PNL,
                           font=self.FS,cursor="hand2").grid(
                row=i//4,column=i%4,sticky="w",padx=6,pady=2)

        ttk.Separator(left).pack(fill="x",padx=8,pady=4)

        # Parametri
        self._sec(left,"PARAMETRI TRENINGA")
        pm=tk.Frame(left,bg=self.PNL); pm.pack(fill="x",padx=10,pady=(0,6))
        self._months_v=tk.IntVar(value=36)
        self._fwd_v=tk.IntVar(value=12)
        self._optuna_en=tk.BooleanVar(value=_HAS_OPTUNA)
        self._meta_en=tk.BooleanVar(value=True)
        self._regime_en=tk.BooleanVar(value=True)
        for r,(lbl,var,mn,mx) in enumerate([
            ("Meseci istorije:",self._months_v,1,120),
            ("Forward bars (label):",self._fwd_v,5,50),
        ]):
            tk.Label(pm,text=lbl,fg=self.MUT,bg=self.PNL,font=self.FS).grid(
                row=r,column=0,sticky="w",pady=2)
            tk.Spinbox(pm,textvariable=var,from_=mn,to=mx,width=6,
                       bg=self.BG,fg=self.TXT,font=self.FS,
                       insertbackground=self.TXT,relief="flat").grid(
                row=r,column=1,sticky="w",padx=6,pady=2)

        # Opcije treninga
        op=tk.Frame(pm,bg=self.PNL)
        op.grid(row=2,column=0,columnspan=2,sticky="w",pady=(6,0))
        for txt,var,clr in [
            (f"Optuna tuning {'(instaliran)' if _HAS_OPTUNA else '(nije instaliran)'}",
             self._optuna_en, self.GRN if _HAS_OPTUNA else self.MUT),
            ("Meta-learner (stacking)", self._meta_en, self.ACC),
            ("Regime detection (trend/ranging)", self._regime_en, self.ACC),
        ]:
            tk.Checkbutton(op,text=txt,variable=var,bg=self.PNL,fg=clr,
                           selectcolor=self.BG,activebackground=self.PNL,
                           font=self.FS,cursor="hand2").pack(anchor="w",pady=1)

        ttk.Separator(left).pack(fill="x",padx=8,pady=4)

        # Progress
        self._sec(left,"NAPREDAK")
        self._prog_lbl=tk.Label(left,text="Čekam...",fg=self.MUT,bg=self.PNL,font=self.FS)
        self._prog_lbl.pack(padx=10,anchor="w")
        s2=ttk.Style(); s2.configure("P.Horizontal.TProgressbar",
            troughcolor=self.BG,background=self.ACC,
            darkcolor=self.ACC,lightcolor=self.ACC,bordercolor=self.BG)
        self._prog=ttk.Progressbar(left,style="P.Horizontal.TProgressbar",
                                   orient="horizontal",mode="determinate")
        self._prog.pack(padx=10,pady=4,fill="x")
        self._prog_d=tk.Label(left,text="",fg=self.MUT,bg=self.PNL,font=self.FS)
        self._prog_d.pack(padx=10,anchor="w")

        ttk.Separator(left).pack(fill="x",padx=8,pady=6)

        # Dugmad
        self._btn_train=tk.Button(left,text="▶  POKRENI TRENING",
            fg=self.BG,bg=self.GRN,font=self.FB,relief="flat",cursor="hand2",pady=9,
            activebackground="#2ea043",command=self._start_training)
        self._btn_train.pack(fill="x",padx=8,pady=3)
        self._btn_tstop=tk.Button(left,text="■  ZAUSTAVI TRENING",
            fg=self.TXT,bg=self.BDR,font=self.FN,relief="flat",cursor="hand2",
            pady=6,state="disabled",command=self._stop_training)
        self._btn_tstop.pack(fill="x",padx=8,pady=2)

        # Tabela rezultata (desno)
        right=tk.Frame(t,bg=self.PNL)
        right.grid(row=0,column=1,sticky="nsew",padx=(0,8),pady=8)
        right.rowconfigure(1,weight=1); right.columnconfigure(0,weight=1)
        tk.Label(right,text="REZULTATI MODELA",fg=self.MUT,bg=self.PNL,
                 font=self.FS).grid(row=0,column=0,sticky="w",padx=10,pady=(8,4))

        cols=("Simbol","TF","Bars","Primerci","XGB","RF","Ensemble",
              "Prec@65%","Hi-sigs","Top Feature","Trenirano")
        self._rtree=ttk.Treeview(right,columns=cols,show="headings",selectmode="browse")
        s3=ttk.Style()
        s3.configure("R.Treeview",background=self.PNL,fieldbackground=self.PNL,
                     foreground=self.TXT,font=("Consolas",9),rowheight=22)
        s3.configure("R.Treeview.Heading",background=self.BG,foreground=self.MUT,
                     font=("Consolas",9,"bold"),relief="flat")
        s3.map("R.Treeview",background=[("selected","#1f6feb")])
        self._rtree.configure(style="R.Treeview")
        ws=[70,50,65,70,65,65,75,75,70,160,120]
        for col,w in zip(cols,ws):
            self._rtree.heading(col,text=col)
            self._rtree.column(col,width=w,minwidth=40,anchor="center")
        self._rtree.column("Top Feature",anchor="w")
        sy=ttk.Scrollbar(right,orient="vertical",command=self._rtree.yview)
        sx=ttk.Scrollbar(right,orient="horizontal",command=self._rtree.xview)
        self._rtree.configure(yscroll=sy.set,xscroll=sx.set)
        self._rtree.grid(row=1,column=0,sticky="nsew")
        sy.grid(row=1,column=1,sticky="ns"); sx.grid(row=2,column=0,sticky="ew")
        self._rtree.tag_configure("good",background="#0d2818",foreground="#3fb950")
        self._rtree.tag_configure("ok",  background="#1c2128",foreground="#c9d1d9")
        self._rtree.tag_configure("warn",background="#2d1f00",foreground="#d29922")
        self._rtree.tag_configure("bad", background="#2d0a0a",foreground="#f85149")

    # ── TAB: LIVE TRADING ─────────────────────────────────────────────────────
    def _build_live_tab(self):
        t=self._tab_live
        # Grid layout za t: red 0=kontrole, red 1=TF+Gemini, red 2=separator, red 3=signali
        t.columnconfigure(0,weight=1)
        t.rowconfigure(3,weight=1)

        # ── Red 0: Risk Management + Sesije + POKRENI ─────────────────────────
        top=tk.Frame(t,bg=self.PNL)
        top.grid(row=0,column=0,sticky="ew",padx=8,pady=(8,2))

        # Risk postavke
        self._sec2(top,"RISK MANAGEMENT")
        rg=tk.Frame(top,bg=self.PNL); rg.pack(side="left",padx=14)
        self._risk_v=tk.DoubleVar(value=1.0)
        self._atr_sl_v=tk.DoubleVar(value=1.5)   # povećano: manje SL hitova
        self._rr_v=tk.DoubleVar(value=2.0)
        self._maxlot_v=tk.DoubleVar(value=2.0)
        self._maxopen_v=tk.IntVar(value=3)
        self._maxday_v=tk.IntVar(value=6)
        self._maxdd_v=tk.DoubleVar(value=4.0)
        self._conf_v=tk.DoubleVar(value=0.65)     # povećano: manje loših signala
        self._min_prec_v=tk.DoubleVar(value=0.55) # NOVO: min preciznost modela
        self._min_adx_v=tk.DoubleVar(value=20.0)  # NOVO: min ADX (izbjegaj ranging)
        self._int_v=tk.IntVar(value=30)

        params=[
            ("Rizik %",self._risk_v,0.1,5.0,0.1,"%.1f"),
            ("ATR × SL",self._atr_sl_v,0.5,4.0,0.1,"%.1f"),
            ("RR ratio",self._rr_v,1.0,5.0,0.1,"%.1f"),
            ("Max lot",self._maxlot_v,0.01,10.0,0.01,"%.2f"),
        ]
        for i,(lbl,var,mn,mx,inc,fmt) in enumerate(params):
            f2=tk.Frame(rg,bg=self.PNL); f2.grid(row=0,column=i,padx=10)
            tk.Label(f2,text=lbl,fg=self.MUT,bg=self.PNL,font=self.FS).pack(anchor="w")
            tk.Spinbox(f2,textvariable=var,from_=mn,to=mx,increment=inc,
                       format=fmt,width=7,bg=self.BG,fg=self.TXT,font=self.FB,
                       insertbackground=self.TXT,relief="flat").pack()

        ttk.Separator(top,orient="vertical").pack(side="left",fill="y",padx=8,pady=4)

        rg2=tk.Frame(top,bg=self.PNL); rg2.pack(side="left",padx=4)
        params2=[
            ("Max otvorenih",self._maxopen_v,1,20,1,"%.0f"),
            ("Max trejdova/dan",self._maxday_v,1,50,1,"%.0f"),
            ("Max DD % dan",self._maxdd_v,0.5,20.0,0.5,"%.1f"),
            ("Confidence min",self._conf_v,0.50,0.95,0.01,"%.2f"),
            ("Min Prec model",self._min_prec_v,0.0,1.0,0.05,"%.2f"),
            ("Min ADX",self._min_adx_v,0.0,50.0,5.0,"%.0f"),
            ("Interval (s)",self._int_v,5,300,5,"%.0f"),
        ]
        for i,(lbl,var,mn,mx,inc,fmt) in enumerate(params2):
            f3=tk.Frame(rg2,bg=self.PNL); f3.grid(row=0,column=i,padx=8)
            tk.Label(f3,text=lbl,fg=self.MUT,bg=self.PNL,font=self.FS).pack(anchor="w")
            tk.Spinbox(f3,textvariable=var,from_=mn,to=mx,increment=inc,
                       format=fmt,width=7,bg=self.BG,fg=self.TXT,font=self.FB,
                       insertbackground=self.TXT,relief="flat").pack()

        ttk.Separator(top,orient="vertical").pack(side="left",fill="y",padx=8,pady=4)

        # Session filter
        sf2=tk.Frame(top,bg=self.PNL); sf2.pack(side="left",padx=10)
        tk.Label(sf2,text="SESIJE ZA TRADING",fg=self.MUT,bg=self.PNL,
                 font=("Consolas",8,"bold")).pack(anchor="w",pady=(4,2))
        sess_colors={"Overlap":self.YLW,"London":self.ACC,"NY":self.GRN,"Asian":self.MUT}
        for sname,sclr in sess_colors.items():
            v=tk.BooleanVar(value=sname in ("Overlap","London","NY"))
            self._sess_vars[sname]=v
            icon=get_session_icon(sname)
            tk.Checkbutton(sf2,text=f"{icon} {sname}",variable=v,
                           bg=self.PNL,fg=sclr,selectcolor=self.BG,
                           activebackground=self.PNL,font=self.FS,
                           cursor="hand2").pack(anchor="w",pady=1)
        self._lbl_sess=tk.Label(sf2,text="",fg=self.MUT,bg=self.PNL,
                                font=("Consolas",9,"bold"))
        self._lbl_sess.pack(anchor="w",pady=(4,0))
        self._update_session_label()

        ttk.Separator(top,orient="vertical").pack(side="left",fill="y",padx=8,pady=4)

        # Automatski trejdovi + zaštite
        sw=tk.Frame(top,bg=self.PNL); sw.pack(side="left",padx=10)
        tk.Label(sw,text="AUTOMATSKI\nTREJDOVI",fg=self.MUT,bg=self.PNL,
                 font=("Consolas",8,"bold")).pack(anchor="w")
        self._trade_en=tk.BooleanVar(value=True)
        self._be_en=tk.BooleanVar(value=True)
        self._trail_en=tk.BooleanVar(value=True)
        self._partial_en=tk.BooleanVar(value=True)
        self._corr_en=tk.BooleanVar(value=True)
        tk.Checkbutton(sw,text="UKLJUCI",variable=self._trade_en,
                       bg=self.PNL,fg=self.GRN,selectcolor=self.BG,
                       activebackground=self.PNL,font=self.FB,cursor="hand2").pack(anchor="w")
        for txt,var in [("Break-even SL",self._be_en),
                        ("Trailing SL",  self._trail_en),
                        ("Partial close",self._partial_en),
                        ("Korel. filter", self._corr_en)]:
            tk.Checkbutton(sw,text=txt,variable=var,bg=self.PNL,fg=self.ACC,
                           selectcolor=self.BG,activebackground=self.PNL,
                           font=self.FS,cursor="hand2").pack(anchor="w",pady=1)

        # Telegram postavke
        ttk.Separator(top,orient="vertical").pack(side="left",fill="y",padx=8,pady=4)
        tgf=tk.Frame(top,bg=self.PNL); tgf.pack(side="left",padx=8)
        tk.Label(tgf,text="TELEGRAM",fg=self.MUT,bg=self.PNL,
                 font=("Consolas",8,"bold")).pack(anchor="w",pady=(4,2))
        self._tg_token_v=tk.StringVar(value=_TG_TOKEN)
        self._tg_chat_v=tk.StringVar(value=_TG_CHAT_ID)
        for lbl2,var2,show2 in [("Bot Token:",self._tg_token_v,"*"),
                                  ("Chat ID:",  self._tg_chat_v,"")]:
            tf2=tk.Frame(tgf,bg=self.PNL); tf2.pack(anchor="w",pady=1)
            tk.Label(tf2,text=lbl2,fg=self.MUT,bg=self.PNL,
                     font=self.FS,width=10).pack(side="left")
            tk.Entry(tf2,textvariable=var2,width=22,bg=self.BG,fg=self.ACC,
                     font=self.FS,insertbackground=self.TXT,
                     relief="flat",show=show2).pack(side="left")
        tk.Button(tgf,text="Test / Snimi",bg=self.BDR,fg=self.TXT,font=self.FS,
                  relief="flat",cursor="hand2",
                  command=self._save_telegram).pack(anchor="w",pady=2)

        bd=tk.Frame(top,bg=self.PNL); bd.pack(side="right",padx=12)
        self._btn_live_start=tk.Button(bd,text="  POKRENI LIVE",
            fg=self.BG,bg=self.ACC,font=self.FB,relief="flat",cursor="hand2",
            pady=10,padx=20,activebackground="#388bfd",command=self._start_live)
        self._btn_live_start.pack(pady=3)
        self._btn_live_stop=tk.Button(bd,text="  ZAUSTAVI",
            fg=self.TXT,bg=self.BDR,font=self.FN,relief="flat",cursor="hand2",
            pady=6,padx=20,state="disabled",command=self._stop_live)
        self._btn_live_stop.pack(pady=2)
        self._btn_close_all=tk.Button(bd,text="X  ZATVORI SVE POZICIJE",
            fg=self.RED,bg=self.PNL,font=self.FS,relief="flat",cursor="hand2",
            pady=5,padx=8,bd=1,command=self._close_all)
        self._btn_close_all.pack(pady=2)

        # ── Red 1: TF selektor + Gemini (uvijek vidljivi) ──────────────────────
        row1=tk.Frame(t,bg=self.BDR)   # BDR boja = vidljivo odvojen red
        row1.grid(row=1,column=0,sticky="ew",padx=8,pady=(2,0))

        # TF checkboxes
        tf_lf=tk.Frame(row1,bg=self.BDR); tf_lf.pack(side="left",padx=(10,0),pady=4)
        tk.Label(tf_lf,text="TF za trading:",fg=self.ACC,bg=self.BDR,
                 font=("Consolas",8,"bold")).pack(side="left",padx=(0,6))
        self._live_tf_vars={}
        tf_clr={"M1":"#8b949e","M5":"#8b949e","M15":"#58a6ff","M30":"#58a6ff",
                "H1":"#3fb950","H4":"#3fb950","D1":"#d29922","W1":"#d29922"}
        for nm in TF_MAP.keys():
            v=tk.BooleanVar(value=nm=="M15")
            self._live_tf_vars[nm]=v
            tk.Checkbutton(tf_lf,text=nm,variable=v,bg=self.BDR,
                           fg=tf_clr.get(nm,self.TXT),selectcolor=self.BG,
                           activebackground=self.BDR,font=self.FS,
                           cursor="hand2").pack(side="left",padx=2)

        ttk.Separator(row1,orient="vertical").pack(side="left",fill="y",padx=8,pady=4)
        tk.Button(row1,text="Preporuc TF",bg="#1f6feb",fg="white",font=self.FS,
                  relief="flat",cursor="hand2",padx=8,pady=2,
                  command=self._suggest_best_tf).pack(side="left",padx=(0,6))
        self._lbl_best_tf=tk.Label(row1,text="(klikni za sugestiju)",
                                   fg=self.MUT,bg=self.BDR,font=self.FS)
        self._lbl_best_tf.pack(side="left",padx=4)

        # Gemini key (desna strana istog reda)
        ttk.Separator(row1,orient="vertical").pack(side="left",fill="y",padx=8,pady=4)
        tk.Label(row1,text="Gemini Key:",fg=self.MUT,bg=self.BDR,
                 font=self.FS).pack(side="left",padx=(0,4))
        self._gemini_key_v=tk.StringVar(value=_GEMINI_KEY)
        ge=tk.Entry(row1,textvariable=self._gemini_key_v,width=30,bg=self.BG,fg=self.ACC,
                    font=self.FS,insertbackground=self.TXT,relief="flat",show="*")
        ge.pack(side="left",padx=(0,4))
        tk.Button(row1,text="Snimi",bg=self.BDR,fg=self.TXT,font=self.FS,relief="flat",
                  cursor="hand2",command=self._save_gemini_key).pack(side="left")
        self._lbl_gemini_status=tk.Label(row1,text="(bez Gemini)",
                                         fg=self.MUT,bg=self.BDR,font=self.FS)
        self._lbl_gemini_status.pack(side="left",padx=6)

        # ── Red 2: Separator + naslov signala ──────────────────────────────────
        ttk.Separator(t).grid(row=2,column=0,sticky="ew",padx=8,pady=2)

        # ── Red 3: Tabela signala (expand) ─────────────────────────────────────
        sf=tk.Frame(t,bg=self.PNL)
        sf.grid(row=3,column=0,sticky="nsew",padx=8,pady=(0,8))
        sf.rowconfigure(1,weight=1); sf.columnconfigure(0,weight=1)
        tk.Label(sf,text="LIVE SIGNALI  (zadnjih 100)",fg=self.MUT,bg=self.PNL,
                 font=self.FS).grid(row=0,column=0,sticky="w",padx=10,pady=(6,2))

        scols=("Vreme","Simbol","TF","Signal","Conf","P-Buy","P-Sell","bv","sv","RSI","ADX","Akcija")
        self._stree=ttk.Treeview(sf,columns=scols,show="headings",selectmode="browse")
        s4=ttk.Style()
        s4.configure("S.Treeview",background=self.PNL,fieldbackground=self.PNL,
                     foreground=self.TXT,font=("Consolas",9),rowheight=22)
        s4.configure("S.Treeview.Heading",background=self.BG,foreground=self.MUT,
                     font=("Consolas",9,"bold"),relief="flat")
        s4.map("S.Treeview",background=[("selected","#1f6feb")])
        self._stree.configure(style="S.Treeview")
        sws=[72,72,48,60,60,60,60,36,36,50,50,80]
        for col,w in zip(scols,sws):
            self._stree.heading(col,text=col)
            self._stree.column(col,width=w,minwidth=30,anchor="center")
        ssy=ttk.Scrollbar(sf,orient="vertical",command=self._stree.yview)
        self._stree.configure(yscroll=ssy.set)
        self._stree.grid(row=1,column=0,sticky="nsew")
        ssy.grid(row=1,column=1,sticky="ns")
        self._stree.tag_configure("buy", background="#0d2818",foreground="#3fb950")
        self._stree.tag_configure("sell",background="#2d0a0a",foreground="#f85149")
        self._stree.tag_configure("hold",background="#1c2128",foreground="#8b949e")

    # ── TAB: POZICIJE ─────────────────────────────────────────────────────────
    def _build_pos_tab(self):
        t=self._tab_pos
        t.columnconfigure(0,weight=1); t.rowconfigure(1,weight=1)

        # Statistika (gornji red)
        sf=tk.Frame(t,bg=self.PNL); sf.pack(fill="x",padx=8,pady=(8,4))
        self._cards={}
        for nm,lbl in [("open","OTVORENE"),("profit","PROFIT"),
                       ("today","TREJDOVI DANAS"),("dd","DNEVNI DD")]:
            c=tk.Frame(sf,bg=self.BDR,padx=16,pady=10); c.pack(side="left",padx=4)
            tk.Label(c,text=lbl,fg=self.MUT,bg=self.BDR,font=self.FS).pack(anchor="w")
            lv=tk.Label(c,text="─",fg=self.TXT,bg=self.BDR,font=self.FL)
            lv.pack(anchor="w"); self._cards[nm]=lv

        tk.Button(sf,text="↻  OSVEŽI",fg=self.ACC,bg=self.PNL,
                  font=self.FS,relief="flat",cursor="hand2",
                  pady=4,padx=8,command=self._refresh_positions).pack(side="right",padx=8)

        # Tabela pozicija
        pf=tk.Frame(t,bg=self.PNL); pf.pack(fill="both",expand=True,padx=8,pady=(0,8))
        pf.rowconfigure(1,weight=1); pf.columnconfigure(0,weight=1)
        tk.Label(pf,text="OTVORENE POZICIJE  (Magic=771100)",
                 fg=self.MUT,bg=self.PNL,font=self.FS).grid(
            row=0,column=0,sticky="w",padx=10,pady=(6,2))

        pcols=("Ticket","Simbol","Tip","Lot","Entry","Trenutna","SL","TP","Profit","Komentar")
        self._ptree=ttk.Treeview(pf,columns=pcols,show="headings",selectmode="browse")
        s5=ttk.Style()
        s5.configure("PT.Treeview",background=self.PNL,fieldbackground=self.PNL,
                     foreground=self.TXT,font=("Consolas",9),rowheight=22)
        s5.configure("PT.Treeview.Heading",background=self.BG,foreground=self.MUT,
                     font=("Consolas",9,"bold"),relief="flat")
        s5.map("PT.Treeview",background=[("selected","#1f6feb")])
        self._ptree.configure(style="PT.Treeview")
        pws=[80,80,50,50,90,90,90,90,80,160]
        for col,w in zip(pcols,pws):
            self._ptree.heading(col,text=col)
            self._ptree.column(col,width=w,minwidth=40,anchor="center")
        psy=ttk.Scrollbar(pf,orient="vertical",command=self._ptree.yview)
        self._ptree.configure(yscroll=psy.set)
        self._ptree.grid(row=1,column=0,sticky="nsew"); psy.grid(row=1,column=1,sticky="ns")
        self._ptree.tag_configure("buy", background="#0d2818",foreground="#3fb950")
        self._ptree.tag_configure("sell",background="#2d0a0a",foreground="#f85149")

        # Dugme zatvori selektovanu
        tk.Button(pf,text="✕  ZATVORI SELEKTOVANU",fg=self.RED,bg=self.PNL,
                  font=self.FS,relief="flat",cursor="hand2",pady=5,padx=10,bd=1,
                  command=self._close_selected).grid(row=2,column=0,sticky="e",padx=8,pady=4)

    # ── TAB: LOG ─────────────────────────────────────────────────────────────
    def _build_log_tab(self):
        t=self._tab_log; t.rowconfigure(0,weight=1); t.columnconfigure(0,weight=1)
        self._log_box=scrolledtext.ScrolledText(
            t,bg=self.BG,fg=self.TXT,font=("Consolas",9),
            insertbackground=self.TXT,borderwidth=0,relief="flat",
            wrap="word",state="disabled")
        self._log_box.pack(fill="both",expand=True,padx=8,pady=8)
        self._log_box.tag_config("G",foreground=self.GRN)
        self._log_box.tag_config("R",foreground=self.RED)
        self._log_box.tag_config("Y",foreground=self.YLW)
        self._log_box.tag_config("B",foreground=self.ACC)
        self._log_box.tag_config("M",foreground=self.MUT)

        tk.Button(t,text="Obriši log",fg=self.MUT,bg=self.PNL,font=self.FS,
                  relief="flat",cursor="hand2",pady=3,
                  command=lambda:(self._log_box.config(state="normal"),
                                  self._log_box.delete("1.0","end"),
                                  self._log_box.config(state="disabled"))
                  ).pack(side="right",padx=8,pady=4)

    def _build_statusbar(self):
        sb_frame = tk.Frame(self.root, bg="#010409")
        sb_frame.pack(fill="x", side="bottom")
        self._sbar = tk.Label(sb_frame, text="Spreman.", fg=self.MUT, bg="#010409",
                              font=self.FS, anchor="w")
        self._sbar.pack(side="left", fill="x", expand=True, ipady=3, padx=6)
        self._lbl_news = tk.Label(sb_frame, text="📰 Vijesti: učitavam...",
                                  fg=self.MUT, bg="#010409", font=self.FS, anchor="e")
        self._lbl_news.pack(side="right", ipady=3, padx=10)

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _sec(self,p,t):
        tk.Label(p,text=t,fg=self.ACC,bg=self.PNL,
                 font=("Consolas",9,"bold")).pack(anchor="w",padx=10,pady=(8,2))

    def _sec2(self,p,t):
        tk.Label(p,text=t,fg=self.ACC,bg=self.PNL,
                 font=("Consolas",9,"bold")).pack(side="left",padx=(12,4),anchor="n",pady=8)

    def _sbtn(self,p,t,c):
        return tk.Button(p,text=t,fg=self.MUT,bg=self.BG,font=self.FS,
                         relief="flat",cursor="hand2",padx=6,pady=2,command=c)

    def _sym_all(self,v):
        for var in self._sym_vars.values(): var.set(v)

    def _log(self,text,tag=""):
        ts=datetime.now().strftime("%H:%M:%S")
        self._log_box.config(state="normal")
        self._log_box.insert("end",f"[{ts}] {text}\n",tag)
        self._log_box.see("end")
        if int(self._log_box.index("end").split(".")[0])>4000:
            self._log_box.delete("1.0","800.0")
        self._log_box.config(state="disabled")

    def _status(self,msg):
        self.root.after(0,lambda:self._sbar.config(text=f"  {msg}"))

    def _tick_clock(self):
        self._lbl_clk.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self._update_session_label()
        self.root.after(1000,self._tick_clock)

    def _update_session_label(self):
        try:
            h    = datetime.utcnow().hour
            name = get_session_name(h)
            icon = get_session_icon(name)
            clr  = {"Overlap":self.YLW,"London":self.ACC,"NY":self.GRN,
                    "Asian":self.MUT,"Dead":self.RED}.get(name,self.MUT)
            self._lbl_sess.config(text=f"{icon} {name}", fg=clr)
        except: pass

    def _is_session_allowed(self) -> bool:
        """Provjeri da li je trenutna sesija dozvoljena za trading.
        Thread-safe: koristi .get() na tkinter BooleanVar koji su kreirani u GUI threadu."""
        if not self._sess_vars:
            return True  # filter nije inicijalizovan → dozvoli sve
        h    = datetime.utcnow().hour
        name = get_session_name(h)
        var  = self._sess_vars.get(name)
        return var.get() if var is not None else True

    # ══════════════════════════════════════════════════════════════════════════
    # MT5
    # ══════════════════════════════════════════════════════════════════════════

    def _connect_mt5(self):
        def _c():
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    self._mt5=mt5
                    info=mt5.terminal_info(); acc=mt5.account_info()
                    self.root.after(0,lambda:self._lbl_mt5.config(
                        text=f"● {info.name}",fg=self.GRN))
                    if acc:
                        self._day_start_bal=acc.balance
                        self.root.after(0,lambda:self._lbl_bal.config(
                            text=f"{acc.balance:.2f} {acc.currency}"))
                    self._q.put(("log",f"MT5 konektovan: {info.name} | "
                                 f"Nalog: {acc.login if acc else '?'}","G"))
                    self._load_models()
                    self.root.after(0,self._refresh_positions)
                    # Povrati dnevne trejdove iz MT5 historije (restart tokom dana)
                    try:
                        today_start = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
                        deals_today = mt5.history_deals_get(today_start, datetime.now())
                        if deals_today:
                            today_count = sum(1 for d in deals_today
                                              if d.magic==MAGIC and d.entry==1)
                            if today_count > 0:
                                self._daily_trades = today_count
                                self._q.put(("log",
                                    f"  📊 Danas već {today_count} trejdova (iz MT5 historije)","M"))
                    except: pass
                    # Skeniranje historije u pozadini
                    threading.Thread(
                        target=scan_mt5_history,
                        args=(self._mt5, self._q, 730),
                        daemon=True).start()
                    # Preuzmi news kalendar u pozadini
                    threading.Thread(
                        target=fetch_news_calendar,
                        args=(self._q,),
                        daemon=True).start()
                else:
                    self._q.put(("log","MT5 nije dostupan. Pokreni MetaTrader 5.","R"))
            except Exception as e:
                self._q.put(("log",f"MT5 greška: {e}","R"))
        threading.Thread(target=_c,daemon=True).start()

    def _load_models(self):
        for p in sorted(MODELS_DIR.glob("meta_*.json")):
            try:
                with open(p) as f: meta=json.load(f)
                key=(meta["symbol"],meta["tf"])
                self._results[key]=meta
                self.root.after(0,lambda m=meta:self._insert_row(m,True))
            except: pass

    # ══════════════════════════════════════════════════════════════════════════
    # TRENING
    # ══════════════════════════════════════════════════════════════════════════

    def _start_training(self):
        if self._training: return
        if self._mt5 is None:
            messagebox.showwarning("MT5","MT5 nije konektovan.")
            return
        syms=[s for s,v in self._sym_vars.items() if v.get()]
        tfs=[(tf,TF_MAP[tf]) for tf,v in self._tf_vars.items() if v.get()]
        if not syms or not tfs:
            messagebox.showwarning("Odabir","Odaberi bar jedan simbol i TF.")
            return
        total=len(syms)*len(tfs)
        self._prog.config(maximum=total,value=0)
        self._done=0; self._total=total
        self._training=True
        self._btn_train.config(state="disabled")
        self._btn_tstop.config(state="normal",bg=self.RED,fg=self.BG)
        months=self._months_v.get(); fwd=self._fwd_v.get()
        self._q.put(("log",
            f"Trening: {len(syms)} simbola × {len(tfs)} TF = {total} modela | "
            f"{months} mjes | fwd={fwd}","B"))
        self._status(f"Treniram {total} modela...")
        threading.Thread(target=self._train_loop,
                         args=(syms,tfs,months,fwd),daemon=True).start()

    def _train_loop(self,syms,tfs,months,fwd):
        import traceback

        # Osvježi feedback iz kompletne MT5 historije prije treninga
        self._q.put(("log","📚 Skeniranje MT5 historije trejdova...","B"))
        scan_mt5_history(self._mt5, self._q, days_back=730)

        for tf_name,tf_id in tfs:
            for symbol in syms:
                if not self._training: break
                self._q.put(("prog_lbl",f"Treniram {symbol} {tf_name}..."))
                try:
                    # Auto-extend lookback za niske TF-ove
                    eff_months = max(months, TF_MIN_MONTHS.get(tf_name, months))
                    min_bars   = TF_MIN_BARS.get(tf_name, 200)

                    # D1/W1/H4: zatraži sve raspoložive barove (broker limit),
                    # ostali TF: cap na 100k
                    large_tf = tf_name in ("D1", "W1", "H4")
                    if large_tf:
                        n_bars = 99999   # uzmi sve što broker ima
                    else:
                        n_bars = min(BARS_PER_MONTH.get(tf_name,2880)*eff_months+300, 100000)
                    self._q.put(("log",f"  {symbol} {tf_name}: učitavam {n_bars:,} barova...","M"))
                    rates=self._mt5.copy_rates_from_pos(symbol,tf_id,0,n_bars)
                    if rates is None or len(rates)<10:
                        self._q.put(("tw",symbol,tf_name,f"Nema podataka ({len(rates) if rates else 0})"))
                        continue
                    df=pd.DataFrame(rates)
                    df["time"]=pd.to_datetime(df["time"],unit="s")
                    df.set_index("time",inplace=True)
                    df.rename(columns={"tick_volume":"volume"},inplace=True)
                    df=df[["open","high","low","close","volume"]]
                    # D1/W1: ne odrežemo historiju — koristimo SVE raspoložive barove
                    if not large_tf:
                        cutoff=pd.Timestamp.now()-pd.DateOffset(months=eff_months)
                        df=df[df.index>=cutoff]
                    if len(df)<min_bars:
                        self._q.put(("tw",symbol,tf_name,
                            f"Premalo bars: {len(df)} (min={min_bars})"))
                        continue
                    warn_few = ""
                    if tf_name == "W1" and len(df) < 500:
                        warn_few = f"  ⚠ W1 ima samo {len(df)} sedmica — model može biti overfit! Provjeri MT5 historiju."
                    elif tf_name == "D1" and len(df) < 1500:
                        warn_few = f"  ⚠ D1 ima samo {len(df)} dana — preporučeno 1500+. Provjeri MT5 historiju."
                    self._q.put(("log",f"  {symbol} {tf_name}: {len(df):,} barova → features...","M"))
                    if warn_few:
                        self._q.put(("log", warn_few, "Y"))
                    feats=compute_features(df, symbol=symbol)
                    labels=generate_labels(df,fwd)
                    data=pd.concat([feats,labels.rename("label")],axis=1).dropna()
                    min_samples=50 if tf_name in ("W1","D1") else 100
                    if len(data)<min_samples:
                        self._q.put(("tw",symbol,tf_name,f"Premalo primeraka: {len(data)}"))
                        continue

                    # Cap na 25.000 uzoraka — D1/W1 koriste sve (inače nemaju dovoljno)
                    MAX_SAMPLES = 25000
                    if not large_tf and len(data) > MAX_SAMPLES:
                        data = data.sample(n=MAX_SAMPLES, random_state=42).sort_index()

                    X=data.drop("label",axis=1).values
                    y=data["label"].values.astype(int)
                    fn=data.drop("label",axis=1).columns.tolist()
                    # Walk-forward: train na 70% (prošlost), val na 15% (bliža prošlost)
                    # Sprečava data leakage — model nikad ne "vidi" budućnost tokom treninga
                    sp_tr=int(len(X)*0.70); sp_vl=int(len(X)*0.85)
                    Xt,Xv=X[:sp_tr],X[sp_tr:sp_vl]; yt,yv=y[:sp_tr],y[sp_tr:sp_vl]
                    if len(np.unique(yt))<2:
                        self._q.put(("tw",symbol,tf_name,"Train set ima samo jednu klasu"))
                        continue

                    # ── Dodaj feedback uzorke (balansirano, 3× težina) ────
                    key=f"{symbol}_{tf_name}"
                    fb_path=MODELS_DIR/f"feedback_{key}.csv"
                    sw_t=np.ones(len(Xt))
                    if fb_path.exists():
                        try:
                            fb=pd.read_csv(fb_path, on_bad_lines='skip')
                            fb_clean=fb.drop(columns=["ts","profit"],errors="ignore")
                            miss_fb=[c for c in fn if c not in fb_clean.columns]
                            if not miss_fb and "label" in fb_clean.columns:
                                # Balansiraj feedback: max 60/40 omjer klasa
                                fb_wins =fb_clean[fb_clean["label"]==1]
                                fb_loss =fb_clean[fb_clean["label"]==0]
                                n_min   =min(len(fb_wins),len(fb_loss))
                                if n_min>0:
                                    max_maj =int(n_min*1.5)  # 60/40
                                    fb_wins =fb_wins.sample(min(len(fb_wins),max_maj),random_state=42)
                                    fb_loss =fb_loss.sample(min(len(fb_loss),max_maj),random_state=42)
                                    fb_bal  =pd.concat([fb_wins,fb_loss]).sample(frac=1,random_state=42)
                                    fb_X=fb_bal[fn].fillna(0).values
                                    fb_y=fb_bal["label"].values.astype(int)
                                    orig_n=len(Xt)
                                    Xt=np.vstack([Xt,fb_X])
                                    yt=np.concatenate([yt,fb_y])
                                    sw_t=np.concatenate([np.ones(orig_n),np.full(len(fb_X),3.0)])
                                    self._q.put(("log",
                                        f"  {symbol} {tf_name}: +{len(fb_X)} feedback "
                                        f"(W:{len(fb_wins)} L:{len(fb_loss)}, 3× težina)","G"))
                        except Exception as fbe:
                            self._q.put(("log",f"  Feedback učitavanje: {fbe}","Y"))

                    small  = len(Xt) < 500
                    medium = len(Xt) < 2000
                    n_est  = 150 if small else (250 if medium else 400)

                    # spw mora biti ovdje — koristi ga Optuna closure ispod
                    n_pos=int((yt==1).sum()); n_neg=int((yt==0).sum())
                    spw=round(n_neg/max(n_pos,1),3)

                    # ── Optuna hyperparameter tuning ──────────────────────────
                    xgb_params = dict(
                        n_estimators=n_est, max_depth=3 if small else (4 if medium else 5),
                        learning_rate=0.05 if small else 0.04,
                        subsample=0.7 if small else 0.8,
                        colsample_bytree=0.7 if small else 0.8,
                        min_child_weight=5 if small else (8 if medium else 10),
                        reg_alpha=0.1, reg_lambda=1.5)
                    if _HAS_OPTUNA and self._optuna_en.get() and not small:
                        self._q.put(("log",f"  {symbol} {tf_name}: Optuna tuning (30 trials)...","M"))
                        def _obj(trial):
                            _p = dict(
                                n_estimators=trial.suggest_int("n_est",100,400),
                                max_depth=trial.suggest_int("depth",3,6),
                                learning_rate=trial.suggest_float("lr",0.01,0.15,log=True),
                                subsample=trial.suggest_float("sub",0.6,1.0),
                                colsample_bytree=trial.suggest_float("col",0.6,1.0),
                                min_child_weight=trial.suggest_int("mcw",3,15),
                                reg_alpha=trial.suggest_float("ra",0.0,1.0),
                                reg_lambda=trial.suggest_float("rl",0.5,3.0),
                                scale_pos_weight=spw, eval_metric="logloss",
                                random_state=42, n_jobs=-1)
                            _m = xgb.XGBClassifier(**_p)
                            _m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False,
                                   sample_weight=sw_t)
                            return accuracy_score(yv,(_m.predict_proba(Xv)[:,1]>0.5).astype(int))
                        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
                        study.optimize(_obj, n_trials=30, show_progress_bar=False)
                        best_p = study.best_params
                        xgb_params.update({
                            "n_estimators": best_p["n_est"],
                            "max_depth": best_p["depth"],
                            "learning_rate": best_p["lr"],
                            "subsample": best_p["sub"],
                            "colsample_bytree": best_p["col"],
                            "min_child_weight": best_p["mcw"],
                            "reg_alpha": best_p["ra"],
                            "reg_lambda": best_p["rl"],
                        })
                        self._q.put(("log",
                            f"  {symbol} {tf_name}: Optuna best acc={study.best_value:.4f} "
                            f"depth={best_p['depth']} lr={best_p['lr']:.4f}","G"))

                    # ── Regime detection: izdvoji trending/ranging podatke ─────
                    Xt_trend=Xt; yt_trend=yt; Xv_trend=Xv; yv_trend=yv
                    Xt_range=Xt; yt_range=yt; Xv_range=Xv; yv_range=yv
                    if self._regime_en.get() and "adx" in fn:
                        adx_idx = fn.index("adx")
                        # Trending: ADX > 0.25 (feature je skaliran /100)
                        t_mask = Xt[:, adx_idx] > 0.25
                        r_mask = Xt[:, adx_idx] <= 0.20
                        tv_mask = Xv[:, adx_idx] > 0.25
                        rv_mask = Xv[:, adx_idx] <= 0.20
                        if t_mask.sum() >= 100:
                            Xt_trend=Xt[t_mask]; yt_trend=yt[t_mask]
                            Xv_trend=Xv[tv_mask] if tv_mask.sum()>10 else Xv
                            yv_trend=yv[tv_mask] if tv_mask.sum()>10 else yv
                        if r_mask.sum() >= 100:
                            Xt_range=Xt[r_mask]; yt_range=yt[r_mask]
                            Xv_range=Xv[rv_mask] if rv_mask.sum()>10 else Xv
                            yv_range=yv[rv_mask] if rv_mask.sum()>10 else yv
                        self._q.put(("log",
                            f"  {symbol} {tf_name}: regime: "
                            f"trend={t_mask.sum()} ranging={r_mask.sum()}","M"))

                    # ── Inkrementalni trening: nastavi od postojećeg modela ─
                    xgb_path=MODELS_DIR/f"xgb_{key}.pkl"
                    rf_path =MODELS_DIR/f"rf_{key}.pkl"
                    existing_xgb=None; existing_rf=None
                    if xgb_path.exists():
                        try:
                            existing_xgb=joblib.load(xgb_path)
                            # Feature mismatch → svježi trening
                            n_feat_xgb = getattr(existing_xgb, 'n_features_in_', len(fn))
                            if n_feat_xgb != len(fn):
                                self._q.put(("log",
                                    f"  {symbol} {tf_name}: XGB feat {n_feat_xgb}→{len(fn)} — svježi trening","Y"))
                                existing_xgb=None
                            else:
                                self._q.put(("log",
                                    f"  {symbol} {tf_name}: nastavljam XGB (dodajem 100 stabala)...","M"))
                        except: existing_xgb=None
                    if rf_path.exists():
                        try:
                            existing_rf=joblib.load(rf_path)
                            # Feature mismatch → svježi trening (sprječava ValueError u predict_proba)
                            n_feat_rf = getattr(existing_rf, 'n_features_in_', len(fn))
                            if n_feat_rf != len(fn):
                                self._q.put(("log",
                                    f"  {symbol} {tf_name}: RF feat {n_feat_rf}→{len(fn)} — svježi trening","Y"))
                                existing_rf=None
                        except: existing_rf=None

                    n_add=100  # stabala koja se dodaju pri inkrementalnom treningu
                    n_fresh=n_est  # stabala pri treningu od nule

                    # ── Balansiranje klasa ──────────────────────────────────
                    # spw je već izračunat iznad (potreban za Optuna closure)
                    cls_ratio=f"{n_pos}W/{n_neg}L" if n_pos+n_neg>0 else "?"
                    self._q.put(("log",
                        f"  {symbol} {tf_name}: klase {cls_ratio}  spw={spw:.2f}","M"))

                    self._q.put(("log",
                        f"  {symbol} {tf_name}: {len(Xt):,} train / {len(Xv):,} val | "
                        f"XGB {'inkrem.' if existing_xgb else n_fresh} stabala...","M"))

                    xm=xgb.XGBClassifier(
                        **{**xgb_params,
                           "n_estimators": n_add if existing_xgb else xgb_params["n_estimators"],
                           "scale_pos_weight": spw,
                           "eval_metric": "logloss",
                           "random_state": 42, "n_jobs": -1})

                    if existing_xgb is not None:
                        try:
                            xm.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False,
                                   xgb_model=existing_xgb.get_booster(),
                                   sample_weight=sw_t)
                        except Exception:
                            xm.n_estimators=n_fresh
                            xm.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False,
                                   sample_weight=sw_t)
                    else:
                        xm.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False,
                               sample_weight=sw_t)

                    if not self._training:
                        self._q.put(("log",f"  {symbol} {tf_name}: trening prekinut.","Y"))
                        break

                    self._q.put(("log",f"  {symbol} {tf_name}: RF {'inkrem.' if existing_rf else n_fresh} stabala...","M"))

                    if existing_rf is not None:
                        try:
                            rm=existing_rf
                            rm.n_estimators+=n_add
                            rm.warm_start=True
                            rm.fit(Xt,yt,sample_weight=sw_t)
                        except Exception:
                            rm=RandomForestClassifier(
                                n_estimators=n_fresh,
                                max_depth=5 if small else (7 if medium else 8),
                                min_samples_split=10 if small else (15 if medium else 20),
                                min_samples_leaf=5 if small else (7 if medium else 10),
                                max_features="sqrt",class_weight="balanced",
                                random_state=42,n_jobs=-1)
                            rm.fit(Xt,yt,sample_weight=sw_t)
                    else:
                        rm=RandomForestClassifier(
                            n_estimators=n_fresh,
                            max_depth=5 if small else (7 if medium else 8),
                            min_samples_split=10 if small else (15 if medium else 20),
                            min_samples_leaf=5 if small else (7 if medium else 10),
                            max_features="sqrt",class_weight="balanced",
                            random_state=42,n_jobs=-1)
                        rm.fit(Xt,yt,sample_weight=sw_t)

                    if not self._training:
                        self._q.put(("log",f"  {symbol} {tf_name}: trening prekinut.","Y"))
                        break

                    # ── LightGBM ──────────────────────────────────────────
                    lm = None; al = 0.0
                    if _HAS_LGB:
                        self._q.put(("log",f"  {symbol} {tf_name}: LightGBM...","M"))
                        lgb_path=MODELS_DIR/f"lgb_{key}.pkl"
                        existing_lgb=None
                        if lgb_path.exists():
                            try:
                                existing_lgb=joblib.load(lgb_path)
                                if getattr(existing_lgb,'n_features_in_',len(fn))!=len(fn):
                                    existing_lgb=None
                            except: existing_lgb=None
                        lm=lgb.LGBMClassifier(
                            n_estimators=n_add if existing_lgb else n_fresh,
                            max_depth=4 if small else (5 if medium else 6),
                            learning_rate=0.05 if small else 0.04,
                            num_leaves=31,subsample=0.8,colsample_bytree=0.8,
                            scale_pos_weight=spw,random_state=42,n_jobs=-1,verbose=-1)
                        if existing_lgb is not None:
                            try:
                                lm.fit(Xt,yt,sample_weight=sw_t,
                                       init_model=existing_lgb.booster_)
                            except Exception:
                                lm.fit(Xt,yt,sample_weight=sw_t)
                        else:
                            lm.fit(Xt,yt,sample_weight=sw_t)
                        al=accuracy_score(yv,(lm.predict_proba(Xv)[:,1]>0.5).astype(int))

                    # ── MLP Neural Network ────────────────────────────────
                    self._q.put(("log",f"  {symbol} {tf_name}: MLP NN...","M"))
                    mlp_path=MODELS_DIR/f"mlp_{key}.pkl"
                    existing_mlp=None
                    if mlp_path.exists():
                        try:
                            existing_mlp=joblib.load(mlp_path)
                            if getattr(existing_mlp,'n_features_in_',len(fn))!=len(fn):
                                existing_mlp=None
                        except: existing_mlp=None
                    if existing_mlp is not None:
                        mm=existing_mlp
                        mm.warm_start=True
                        mm.max_iter=50
                        mm.fit(Xt,yt)
                    else:
                        mm=MLPClassifier(
                            hidden_layer_sizes=(128,64,32),activation="relu",
                            solver="adam",learning_rate_init=0.001,
                            max_iter=200,random_state=42,early_stopping=True,
                            validation_fraction=0.1,n_iter_no_change=15)
                        mm.fit(Xt,yt)
                    am=accuracy_score(yv,(mm.predict_proba(Xv)[:,1]>0.5).astype(int))

                    if not self._training:
                        self._q.put(("log",f"  {symbol} {tf_name}: trening prekinut.","Y"))
                        break

                    # ── Regime modeli (trend/ranging XGB) ────────────────
                    xm_trend = None; xm_range = None
                    if self._regime_en.get() and "adx" in fn:
                        try:
                            if len(Xt_trend) >= 100 and len(np.unique(yt_trend)) == 2:
                                xm_trend = xgb.XGBClassifier(
                                    **{**xgb_params, "scale_pos_weight": spw,
                                       "eval_metric":"logloss","random_state":42,"n_jobs":-1})
                                xm_trend.fit(Xt_trend, yt_trend, verbose=False)
                            if len(Xt_range) >= 100 and len(np.unique(yt_range)) == 2:
                                xm_range = xgb.XGBClassifier(
                                    **{**xgb_params, "scale_pos_weight": spw,
                                       "eval_metric":"logloss","random_state":42,"n_jobs":-1})
                                xm_range.fit(Xt_range, yt_range, verbose=False)
                            if xm_trend or xm_range:
                                joblib.dump(xm_trend, MODELS_DIR/f"xgb_trend_{key}.pkl")
                                joblib.dump(xm_range, MODELS_DIR/f"xgb_range_{key}.pkl")
                                self._q.put(("log",
                                    f"  {symbol} {tf_name}: regime modeli snimljeni","G"))
                        except Exception as re_e:
                            self._q.put(("log",f"  Regime greška: {re_e}","Y"))

                    # ── Ensemble: XGB+RF+LGB+MLP ─────────────────────────
                    px=xm.predict_proba(Xv)[:,1]; pr=rm.predict_proba(Xv)[:,1]
                    all_probs=[px,pr]
                    if lm is not None: all_probs.append(lm.predict_proba(Xv)[:,1])
                    all_probs.append(mm.predict_proba(Xv)[:,1])
                    en=np.mean(all_probs,axis=0)
                    ae=accuracy_score(yv,(en>0.5).astype(int))  # preliminary ae (may be replaced by meta)

                    # ── Meta-learner (stacking): uči težine baza ──────────
                    meta_lr = None
                    if self._meta_en.get() and len(Xt) >= 500:
                        try:
                            # OOF na train setu (5-fold)
                            from sklearn.model_selection import StratifiedKFold
                            kf = StratifiedKFold(n_splits=5, shuffle=False)
                            oof_preds = np.zeros((len(Xt), 4 if lm else 3))
                            for fold_tr, fold_vl in kf.split(Xt, yt):
                                _models = [
                                    xgb.XGBClassifier(**{**xgb_params,"scale_pos_weight":spw,
                                        "eval_metric":"logloss","random_state":42,"n_jobs":-1,"n_estimators":100}),
                                    RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1),
                                ]
                                if lm: _models.append(lgb.LGBMClassifier(n_estimators=100,random_state=42,n_jobs=-1,verbose=-1))
                                _models.append(MLPClassifier(hidden_layer_sizes=(64,32),max_iter=100,random_state=42))
                                for mi, _m in enumerate(_models):
                                    _m.fit(Xt[fold_tr], yt[fold_tr])
                                    oof_preds[fold_vl, mi] = _m.predict_proba(Xt[fold_vl])[:,1]
                            scaler_meta = StandardScaler()
                            oof_scaled  = scaler_meta.fit_transform(oof_preds)
                            meta_lr = LogisticRegression(C=1.0, random_state=42, max_iter=500)
                            meta_lr.fit(oof_scaled, yt)
                            # Evaluiraj na val setu s meta-learnerom
                            val_stack = np.column_stack([px, pr] +
                                ([lm.predict_proba(Xv)[:,1]] if lm else []) +
                                [mm.predict_proba(Xv)[:,1]])
                            val_scaled = scaler_meta.transform(val_stack)
                            en_meta = meta_lr.predict_proba(val_scaled)[:,1]
                            ae_meta = accuracy_score(yv,(en_meta>0.5).astype(int))
                            if ae_meta > ae:
                                en = en_meta   # koristi meta ako je bolji
                                joblib.dump((meta_lr, scaler_meta), MODELS_DIR/f"meta_lr_{key}.pkl")
                                self._q.put(("log",
                                    f"  {symbol} {tf_name}: Meta-learner ens={ae_meta:.4f} "
                                    f"(+{ae_meta-ae:.4f} vs avg) — aktiviran","G"))
                            else:
                                self._q.put(("log",
                                    f"  {symbol} {tf_name}: Meta-learner ens={ae_meta:.4f} "
                                    f"vs avg={ae:.4f} — avg bolji, preskačem","M"))
                        except Exception as ml_e:
                            self._q.put(("log",f"  Meta-learner greška: {ml_e}","Y"))
                    ax=accuracy_score(yv,(px>0.5).astype(int))
                    ar=accuracy_score(yv,(pr>0.5).astype(int))
                    ae=accuracy_score(yv,(en>0.5).astype(int))  # final ae (recomputed after meta may have updated en)
                    hi=en>0.65
                    ph=precision_score(yv[hi],(en[hi]>0.5).astype(int),zero_division=0) if hi.sum()>0 else 0.0

                    imp=pd.Series(rm.feature_importances_,index=fn).sort_values(ascending=False)
                    top5=", ".join(imp.head(5).index.tolist())

                    # ── No-regression guard: ne snimi ako je model lošiji ──
                    meta_path=MODELS_DIR/f"meta_{key}.json"
                    old_ens=0.0
                    if meta_path.exists():
                        try:
                            with open(meta_path) as _f: _om=json.load(_f)
                            old_ens=float(_om.get("acc_ens",0.0))
                        except Exception: old_ens=0.0
                    if old_ens>0 and ae < old_ens-0.005:
                        self._q.put(("log",
                            f"  ⚠ {symbol} {tf_name}: novi ens={ae:.3f} < stari={old_ens:.3f} "
                            f"— zadržavam stari model","Y"))
                        with open(meta_path) as _f: _old_meta=json.load(_f)
                        self._q.put(("tok",symbol,tf_name,_old_meta))
                    else:
                        joblib.dump(xm,MODELS_DIR/f"xgb_{key}.pkl")
                        joblib.dump(rm,MODELS_DIR/f"rf_{key}.pkl")
                        joblib.dump(mm,MODELS_DIR/f"mlp_{key}.pkl")
                        if lm is not None:
                            joblib.dump(lm,MODELS_DIR/f"lgb_{key}.pkl")
                        with open(MODELS_DIR/f"features_{key}.json","w") as fp: json.dump(fn,fp)
                        n_models=2+(1 if lm else 0)+1
                        meta={"symbol":symbol,"tf":tf_name,"bars":len(df),"samples":len(data),
                              "features":len(fn),"acc_xgb":round(ax,4),"acc_rf":round(ar,4),
                              "acc_lgb":round(al,4),"acc_mlp":round(am,4),
                              "acc_ens":round(ae,4),"prec_hi65":round(ph,4),
                              "hi_signals":int(hi.sum()),"n_models":n_models,
                              "top5":top5,"trained_at":datetime.now().strftime("%Y-%m-%d %H:%M")}
                        with open(meta_path,"w") as fp: json.dump(meta,fp)
                        self._q.put(("tok",symbol,tf_name,meta))
                except Exception as e:
                    tb=traceback.format_exc()
                    self._q.put(("tw",symbol,tf_name,str(e)))
                    self._q.put(("log",tb,"R"))
        _model_cache.clear()
        self._q.put(("tdone",))

    def _stop_training(self):
        self._training=False
        self._q.put(("log","Trening prekinut — čeka završetak trenutnog modela.","Y"))

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE
    # ══════════════════════════════════════════════════════════════════════════

    def _start_live(self):
        if self._live_run: return
        if self._mt5 is None:
            messagebox.showwarning("MT5","MT5 nije konektovan.")
            return
        syms=[s for s,v in self._sym_vars.items() if v.get()]
        tfs=[(tf,TF_MAP[tf]) for tf,v in self._live_tf_vars.items() if v.get()]
        trained=[(s,tf,tid) for s in syms for tf,tid in tfs
                 if (MODELS_DIR/f"xgb_{s}_{tf}.pkl").exists()]
        if not trained:
            messagebox.showwarning("Modeli","Nema treniranih modela za odabrane simbole/TF.\n"
                                   "Prvo pokreni trening.")
            return
        self._live_run=True
        self._live_warmup=True
        self._btn_live_start.config(state="disabled")
        self._btn_live_stop.config(state="normal")
        acc=self._mt5.account_info()
        if acc: self._day_start_bal=acc.balance
        self._daily_trades=0
        self._last_day=datetime.now().strftime("%Y-%m-%d")
        self._session_start=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._session_bal_start=acc.balance if acc else 0.0
        self._q.put(("log",f"Live pokrenut — {len(trained)} modela aktivno | "
                     f"Trade: {'DA' if self._trade_en.get() else 'NE'} | "
                     f"Warmup: 1 scan bez trejdova","B"))
        self._live_thread=threading.Thread(
            target=self._live_loop,args=(trained,),daemon=True)
        self._live_thread.start()

    def _stop_live(self):
        self._live_run=False
        self._btn_live_start.config(state="normal")
        self._btn_live_stop.config(state="disabled")
        # Sačekaj da loop thread završi (max 3s) pa tek onda snimi sesiju
        if self._live_thread and self._live_thread.is_alive():
            self._live_thread.join(timeout=3.0)
        self._save_session()
        self._q.put(("log","Live zaustavljen.","Y"))

    def _suggest_best_tf(self):
        """Analiziraj meta podatke svih modela i preporuči TF sa najboljim performansama.

        Formula (po modelu):
          base_score  = acc_ens * 0.35 + prec_hi65 * 0.65
          zero_penalty: ako prec_hi65==0 ili hi_signals==0  → score = acc_ens * 0.35
                        (preciznost nepoznata = bez bonusa, samo accuracy)

        Formula (po TF):
          raw_avg     = prosjek base_score po svim simbolima
          valid_ratio = udio simbola koji imaju hi_signals > 0
          tf_score    = raw_avg * (0.4 + 0.6 * valid_ratio)
          → TF gdje su svi simboli bez high-conf signala dobija 40% od raw_avg
          → TF gdje svi imaju signale dobija pun raw_avg
        """
        tf_raw   = {}   # tf -> [base_score, ...]
        tf_valid = {}   # tf -> [1 ako hi_signals>0 else 0, ...]
        tf_det   = {}   # tf -> [(sym, ae, ph, hi_sig, samples), ...]

        for p in sorted(MODELS_DIR.glob("meta_*.json")):
            try:
                with open(p) as f: meta=json.load(f)
                tf  = meta.get("tf","")
                sym = meta.get("symbol","")
                if not tf: continue
                samples  = meta.get("samples", 0)
                if samples < 50: continue
                ae       = float(meta.get("acc_ens",    0.5))
                ph       = float(meta.get("prec_hi65",  0.0))
                hi_sig   = int(  meta.get("hi_signals", 0))
                # Base score: ako nema high-conf signala, koristi samo accuracy
                if hi_sig == 0 or ph == 0.0:
                    base = ae * 0.35   # bez preciznosti
                else:
                    base = ae * 0.35 + ph * 0.65
                tf_raw.setdefault(tf,[]).append(base)
                tf_valid.setdefault(tf,[]).append(1 if hi_sig > 0 else 0)
                tf_det.setdefault(tf,[]).append((sym, ae, ph, hi_sig, samples))
            except: pass

        if not tf_raw:
            self._lbl_best_tf.config(text="⚠ Nema modela — prvo pokreni trening!", fg=self.YLW)
            self._q.put(("log","📊 Sugestija: nema meta fajlova u models/ folderu.","Y"))
            return

        # TF score sa kaznama za pr=0
        tf_score = {}
        for tf in tf_raw:
            raw_avg     = sum(tf_raw[tf])   / len(tf_raw[tf])
            valid_ratio = sum(tf_valid[tf]) / len(tf_valid[tf])
            tf_score[tf] = raw_avg * (0.4 + 0.6 * valid_ratio)

        ranked = sorted(tf_score.items(), key=lambda x: -x[1])
        best_tf, best_sc = ranked[0]

        # Auto-selektuj samo best TF
        for tf_name, v in self._live_tf_vars.items():
            v.set(tf_name == best_tf)

        self._lbl_best_tf.config(
            text=f"Preporucen: {best_tf}  (score={best_sc:.3f})",
            fg=self.GRN)

        self._q.put(("log","","M"))
        self._q.put(("log","📊 ═══ TF ANALIZA — PREPORUKA (penalizuje pr=0) ═══","M"))
        medals = ["🥇","🥈","🥉"]
        for i,(tf,sc) in enumerate(ranked):
            n      = len(tf_raw[tf])
            vr     = sum(tf_valid[tf]) / n
            med    = medals[i] if i<3 else "  "
            clr    = "G" if i==0 else ("M" if i<3 else "M")
            # Detalji: označi pr=0 sa upozorenjem
            dets_parts = []
            for sym,ae,ph,hi,smp in sorted(tf_det[tf], key=lambda x: -(x[1]*0.35+(x[2]*0.65 if x[3]>0 else 0))):
                flag = "⚠" if hi==0 else ""
                dets_parts.append(f"{sym}(ens={ae:.2f},pr={ph:.2f},hi={hi}){flag}")
            dets = "  ".join(dets_parts)
            penalty_info = f"valid={vr:.0%}" if vr < 1.0 else "ok"
            self._q.put(("log",
                f"  {med} {tf:4s}  score={sc:.3f}  [{penalty_info}]  ({n} mod)  |  {dets}",
                clr))
        self._q.put(("log",
            f"  → Selektovan: {best_tf}  "
            f"(raw×valid_ratio  |  valid={sum(tf_valid[best_tf])/len(tf_valid[best_tf]):.0%})","G"))
        self._q.put(("log","","M"))

    def _save_gemini_key(self):
        """Snimi Gemini API ključ i osvježi sentiment cache."""
        global _GEMINI_KEY
        _GEMINI_KEY = self._gemini_key_v.get().strip()
        if _GEMINI_KEY:
            masked = _GEMINI_KEY[:6] + "..." + _GEMINI_KEY[-4:]
            self._lbl_gemini_status.config(text=f"✅ aktivan ({masked})", fg=self.GRN)
            self._q.put(("log", "🤖 Gemini API key postavljen — analiziram vijesti...", "G"))
            threading.Thread(target=fetch_news_calendar, args=(self._q,), daemon=True).start()
        else:
            self._lbl_gemini_status.config(text="(key nije postavljen)", fg=self.MUT)
            self._q.put(("log", "🤖 Gemini API key obrisan — sentiment filter isključen.", "Y"))

    def _save_telegram(self):
        """Snimi Telegram token/chat i pošalji test poruku."""
        global _TG_TOKEN, _TG_CHAT_ID
        _TG_TOKEN   = self._tg_token_v.get().strip()
        _TG_CHAT_ID = self._tg_chat_v.get().strip()
        if _TG_TOKEN and _TG_CHAT_ID:
            send_telegram(
                "<b>OmniTrader</b> — Telegram notifikacije aktivne!\n"
                "Dobivat ces poruke o trejdovima, dnevni izvjestaj i upozorenja.",
                self._q)
            self._q.put(("log", f"Telegram: token/chat sačuvani — test poruka poslana.", "G"))
        else:
            self._q.put(("log", "Telegram: upiši token I chat ID da aktiviraš notifikacije.", "Y"))

    def _save_session(self):
        """Snimi statistiku sesije u sessions.json."""
        try:
            acc = self._mt5.account_info() if self._mt5 else None
            bal_end = acc.balance if acc else 0.0
            pnl = bal_end - getattr(self, "_session_bal_start", bal_end)
            session = {
                "start":      getattr(self, "_session_start", "?"),
                "end":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trades":     self._daily_trades,
                "bal_start":  round(getattr(self, "_session_bal_start", 0.0), 2),
                "bal_end":    round(bal_end, 2),
                "pnl":        round(pnl, 2),
                "account":    acc.login if acc else 0,
            }
            sp = MODELS_DIR / "sessions.json"
            sessions = []
            if sp.exists():
                try:
                    with open(sp) as f: sessions = json.load(f)
                except: sessions = []
            sessions.append(session)
            sessions = sessions[-200:]  # čuvaj zadnjih 200 sesija
            with open(sp, "w") as f: json.dump(sessions, f, indent=2)

            # Agregatna statistika
            total_trades = sum(s.get("trades", 0) for s in sessions)
            total_pnl    = sum(s.get("pnl", 0.0)  for s in sessions)
            self._q.put(("log",
                f"  💾 Sesija snimljena | Trejdovi: {session['trades']} | "
                f"P&L sesije: {pnl:+.2f} | "
                f"Ukupno sesija: {len(sessions)} | "
                f"Ukupno trejdova: {total_trades} | "
                f"Ukupni P&L: {total_pnl:+.2f}", "B"))
        except Exception as e:
            self._q.put(("log", f"  Session save greška: {e}", "Y"))

    def _live_loop(self,trained):
        import traceback
        while self._live_run:
            try:
                # ── Auto-refresh news kalendara svaka 4 sata ──────────────
                if (not _NEWS_CACHE or
                        (_NEWS_LAST_FETCH and
                         (datetime.utcnow()-_NEWS_LAST_FETCH).total_seconds() > 4*3600)):
                    threading.Thread(target=fetch_news_calendar,
                                     args=(self._q,), daemon=True).start()

                # ── Ažuriraj news indikator u GUI ─────────────────────────
                syms = list({s for s,_,_ in trained})
                news_str = get_next_news(syms)
                self._q.put(("news_bar", news_str))

                today=datetime.now().strftime("%Y-%m-%d")
                if today!=self._last_day:
                    self._last_day=today; self._daily_trades=0
                    acc=self._mt5.account_info()
                    if acc: self._day_start_bal=acc.balance

                acc=self._mt5.account_info()
                if acc and self._day_start_bal>0:
                    dd=(self._day_start_bal-acc.equity)/self._day_start_bal*100
                    if dd>=self._maxdd_v.get():
                        self._q.put(("log",f"⚠ MAX DD dostignut ({dd:.1f}%). Live pauziran.","R"))
                        self._live_run=False; break
                    if self._daily_trades>=self._maxday_v.get():
                        self._q.put(("log",f"Max trejdova/dan dostignut ({self._daily_trades}).","Y"))
                        time.sleep(60); continue

                # ── Detektuj zatvorene pozicije i snimi feedback ───────────
                cur_positions = get_positions(self._mt5)
                cur_tickets   = {p.ticket for p in cur_positions}
                closed = self._prev_tickets - cur_tickets
                for ticket in closed:
                    if ticket not in self._pending_feedback: continue
                    info = self._pending_feedback.pop(ticket)
                    try:
                        from_dt = datetime.now() - timedelta(days=3)
                        deals   = self._mt5.history_deals_get(from_dt, datetime.now())
                        profit  = 0.0
                        if deals:
                            for d in deals:
                                if d.position_id == ticket and d.entry == 1:
                                    profit = d.profit + d.commission + d.swap
                                    break
                        label = 1 if profit > 0 else 0
                        _save_feedback(info["symbol"], info["tf"],
                                       info["feat_names"], info["features"],
                                       label, profit)
                        icon = "WIN" if label == 1 else "LOSS"
                        clr  = "G" if label == 1 else "Y"
                        self._q.put(("log",
                            f"  Naucio: {info['symbol']} {info['tf']} "
                            f"{info['direction']} → {icon} ({profit:+.2f})", clr))
                        # Telegram notifikacija o zatvorenom trejdu
                        send_telegram(
                            f"<b>{'WIN' if label==1 else 'LOSS'} {info['symbol']}</b>\n"
                            f"{info['direction']}  TF: {info['tf']}\n"
                            f"Profit: <b>{profit:+.2f} EUR</b>",
                            self._q)
                    except Exception as fb_e:
                        self._q.put(("log", f"  Feedback greška: {fb_e}", "Y"))
                self._prev_tickets = cur_tickets

                # Warmup — prvi scan samo beleži barove, ne trguje
                if self._live_warmup:
                    for symbol,tf_name,tf_id in trained:
                        bars=self._mt5.copy_rates_from_pos(symbol,tf_id,0,2)
                        if bars is not None and len(bars)>=2:
                            self._last_bars[f"{symbol}_{tf_name}"]=int(bars[-2]["time"])
                    self._live_warmup=False
                    self._q.put(("log","Warmup završen — pratim nove barove, TF filter aktivan.","Y"))
                else:
                    # ── Korak 1: skupi signale svih TF-ova ───────────────
                    round_sigs = {}  # symbol -> [result, ...]
                    min_prec = self._min_prec_v.get()
                    min_adx  = self._min_adx_v.get()
                    # Učitaj prec_hi65 iz meta fajlova (cache za ovaj scan round)
                    meta_prec_cache = {}
                    for p in MODELS_DIR.glob("meta_*.json"):
                        try:
                            with open(p) as _f: _m = json.load(_f)
                            meta_prec_cache[f"{_m['symbol']}_{_m['tf']}"] = float(_m.get("prec_hi65", 0.0))
                        except: pass
                    for symbol,tf_name,tf_id in trained:
                        if not self._live_run: break
                        # Min prec filter — preskoci ako model nema dovoljno preciznosti
                        model_prec = meta_prec_cache.get(f"{symbol}_{tf_name}", 0.0)
                        if model_prec < min_prec:
                            continue
                        r = get_signal(self._mt5, symbol, tf_name, tf_id,
                                       self._conf_v.get(), self._last_bars, self._q)
                        if r:
                            self._q.put(("signal", r))
                            if r["signal"] != "HOLD":
                                # Min ADX filter — preskoci ranging tržište
                                if r["adx"] < min_adx:
                                    self._q.put(("log",
                                        f"  ~ {symbol} {tf_name}: ADX={r['adx']:.1f} < {min_adx:.0f} (ranging) — preskačem","M"))
                                    continue
                                round_sigs.setdefault(symbol, []).append(r)

                    # ── Korak 2: TF filter + otvaranje trejdova ───────────
                    if self._trade_en.get():
                        # Session filter — ne trguj u nedozvoljenoj sesiji
                        if not self._is_session_allowed():
                            cur_sess = get_session_name(datetime.utcnow().hour)
                            self._q.put(("log",
                                f"  {get_session_icon(cur_sess)} Sesija {cur_sess} "
                                f"nije dozvoljena — čekam...", "M"))
                        else:
                          open_pos = get_positions(self._mt5)
                          open_syms = {p.symbol for p in open_pos}
                          n_open = len(open_pos)

                          for symbol, sig_list in round_sigs.items():
                            if not self._live_run: break
                            if n_open >= self._maxopen_v.get(): break
                            if self._daily_trades >= self._maxday_v.get(): break
                            if symbol in open_syms: continue

                            buys  = [r for r in sig_list if r["signal"]=="BUY"]
                            sells = [r for r in sig_list if r["signal"]=="SELL"]

                            # Weighted TF voting — saberi konfidencije po smjeru
                            w_buy  = sum(r["confidence"] for r in buys)
                            w_sell = sum(r["confidence"] for r in sells)

                            if w_buy > 0 and w_sell > 0:
                                ratio = max(w_buy,w_sell)/min(w_buy,w_sell)
                                if ratio < 1.4:  # razlika manja od 40% → konflikt
                                    info=" | ".join(
                                        f"{r['tf']}:{r['signal']}({r['confidence']:.0%})"
                                        for r in sig_list)
                                    self._q.put(("log",
                                        f"  ⚡ {symbol}: TF konflikt [{info}] "
                                        f"buy={w_buy:.2f} sell={w_sell:.2f} — preskačem","Y"))
                                    continue
                                # Jedna strana jasno dominira — trgujemo
                            direction = "BUY" if w_buy >= w_sell else "SELL"
                            active    = buys if direction=="BUY" else sells
                            best      = max(active, key=lambda r: r["confidence"])
                            atr_v     = best["atr"]
                            sl_dist   = atr_v * self._atr_sl_v.get()
                            # Gold needs wider SL to survive volatility spikes;
                            # calc_lot() uses sl_dist so lot is reduced automatically
                            if symbol == "XAUUSD":
                                gold_min_sl = atr_v * 3.0
                                if sl_dist < gold_min_sl:
                                    self._q.put(("log",
                                        f"  XAUUSD: SL rozširen {sl_dist:.2f} → {gold_min_sl:.2f} (3×ATR)", "M"))
                                    sl_dist = gold_min_sl
                            tp_dist   = sl_dist * self._rr_v.get()
                            tfs_ok    = "+".join(r["tf"] for r in active)

                            # ── vconv filter: blokiraj mješovite signale ────
                            # vconv=(bv-sv)/(bv+sv+1): +1=sve BUY, -1=sve SELL, ~0=konfuzija
                            # Zahtijeva jaku konvergenciju (>0.5 za BUY, <-0.5 za SELL)
                            vconv_val = best.get("vconv", 0.0)
                            vconv_ok  = (vconv_val > 0.5  if direction == "BUY"
                                         else vconv_val < -0.5)
                            if not vconv_ok:
                                self._q.put(("log",
                                    f"  ~ {symbol}: vconv={vconv_val:.2f} nedovoljno "
                                    f"({'>' if direction=='BUY' else '<'}"
                                    f"{'0.5' if direction=='BUY' else '-0.5'}) — preskačem", "M"))
                                continue

                            # ── Korelacioni filter ──────────────────────────
                            if self._corr_en.get():
                                ok_corr, corr_reason = correlation_allows(
                                    symbol, direction, open_pos)
                                if not ok_corr:
                                    self._q.put(("log",
                                        f"  ~ {symbol}: {corr_reason} — preskačem","M"))
                                    continue

                            ticket, err = open_trade(
                                self._mt5, symbol, direction, sl_dist, tp_dist,
                                self._risk_v.get(), self._maxlot_v.get(),
                                f"ML:{tfs_ok}:{best['confidence']:.2f}")
                            if ticket:
                                n_open += 1
                                self._q.put(("log",
                                    f"  ✔ OTVOREN {direction} {symbol} | "
                                    f"TF filter: [{tfs_ok}] conf={best['confidence']:.1%} | "
                                    f"sl={sl_dist:.5f} tp={tp_dist:.5f}","G"))
                                self._q.put(("trade_opened",))
                                self._pending_feedback[ticket] = {
                                    "symbol": symbol, "tf": best["tf"],
                                    "direction": direction,
                                    "feat_names": best["feat_names"],
                                    "features": best["features"],
                                }
                                # Telegram notifikacija
                                send_telegram(
                                    f"<b>OTVOREN {direction} {symbol}</b>\n"
                                    f"TF: {tfs_ok}  Conf: {best['confidence']:.1%}\n"
                                    f"SL: {sl_dist:.5f}  TP: {tp_dist:.5f}",
                                    self._q)
                            else:
                                self._q.put(("log",
                                    f"  ✘ GREŠKA otvaranja {symbol}: {err}","R"))

                    # ── Upravljanje SL (break-even + trailing + partial) ─────
                    if self._trade_en.get():
                        _cur_pos = get_positions(self._mt5)
                        if _cur_pos:
                            manage_positions(
                                self._mt5, _cur_pos,
                                self._atr_sl_v.get(), self._q,
                                be_enabled      = self._be_en.get(),
                                trail_enabled   = self._trail_en.get(),
                                partial_enabled = self._partial_en.get())

                self._q.put(("refresh_pos",))
            except Exception as e:
                self._q.put(("log",f"Live loop greška: {e}\n{traceback.format_exc()}","R"))

            for _ in range(self._int_v.get()*10):
                if not self._live_run: break
                time.sleep(0.1)

    # ══════════════════════════════════════════════════════════════════════════
    # POZICIJE
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_positions(self):
        if self._mt5 is None: return
        for item in self._ptree.get_children(): self._ptree.delete(item)
        positions=get_positions(self._mt5)
        total_profit=0.0
        for p in positions:
            cur=self._mt5.symbol_info_tick(p.symbol)
            cur_price=cur.bid if p.type==0 else cur.ask if cur else 0
            profit=p.profit
            total_profit+=profit
            typ="BUY" if p.type==0 else "SELL"
            tag="buy" if p.type==0 else "sell"
            self._ptree.insert("",0,tags=(tag,),values=(
                p.ticket,p.symbol,typ,f"{p.volume:.2f}",
                f"{p.price_open:.5f}",f"{cur_price:.5f}",
                f"{p.sl:.5f}",f"{p.tp:.5f}",
                f"{profit:+.2f}",p.comment))

        acc=self._mt5.account_info()
        dd=0.0
        if acc and self._day_start_bal>0:
            dd=(self._day_start_bal-acc.equity)/self._day_start_bal*100

        self._cards["open"].config(text=str(len(positions)),
            fg=self.ACC if positions else self.MUT)
        self._cards["profit"].config(text=f"{total_profit:+.2f}",
            fg=self.GRN if total_profit>=0 else self.RED)
        self._cards["today"].config(text=str(self._daily_trades),fg=self.TXT)
        self._cards["dd"].config(text=f"{dd:.2f}%",
            fg=self.RED if dd>2 else self.YLW if dd>1 else self.GRN)
        if acc:
            self._lbl_bal.config(
                text=f"B: {acc.balance:.2f}  E: {acc.equity:.2f}  {acc.currency}")

    def _close_selected(self):
        sel=self._ptree.selection()
        if not sel: return
        vals=self._ptree.item(sel[0],"values")
        ticket=int(vals[0])
        if messagebox.askyesno("Zatvori","Zatvori poziciju "+str(ticket)+"?"):
            ok,err=close_position(self._mt5,ticket)
            if ok: self._log(f"Pozicija {ticket} zatvorena.","G")
            else:  self._log(f"Greška zatvaranja {ticket}: {err}","R")
            self._refresh_positions()

    def _close_all(self):
        if not messagebox.askyesno("Zatvori sve","Zatvori SVE ML pozicije?"): return
        positions=get_positions(self._mt5)
        for p in positions:
            ok,err=close_position(self._mt5,p.ticket)
            tag="G" if ok else "R"
            msg=f"{'Zatvorena' if ok else 'Greška'} {p.ticket} {p.symbol}: {err or 'OK'}"
            self._q.put(("log",msg,tag))
        self.root.after(500,self._refresh_positions)

    # ══════════════════════════════════════════════════════════════════════════
    # TABELA REZULTATA
    # ══════════════════════════════════════════════════════════════════════════

    def _insert_row(self,meta,loaded=False):
        for item in self._rtree.get_children():
            v=self._rtree.item(item,"values")
            if v[0]==meta["symbol"] and v[1]==meta["tf"]:
                self._rtree.delete(item); break
        ae=meta["acc_ens"]
        tag="good" if ae>=0.62 else "ok" if ae>=0.57 else "warn" if ae>=0.52 else "bad"
        top1=meta.get("top5","–").split(",")[0].strip()
        self._rtree.insert("","end",tags=(tag,),values=(
            meta["symbol"],meta["tf"],f"{meta['bars']:,}",f"{meta['samples']:,}",
            f"{meta['acc_xgb']:.3f}",f"{meta['acc_rf']:.3f}",f"{meta['acc_ens']:.3f}",
            f"{meta['prec_hi65']:.3f}",meta["hi_signals"],top1,
            meta.get("trained_at","–")))
        self._results[(meta["symbol"],meta["tf"])]=meta

    # ══════════════════════════════════════════════════════════════════════════
    # POLL
    # ══════════════════════════════════════════════════════════════════════════

    def _poll(self):
        try:
            while True:
                msg=self._q.get_nowait(); t=msg[0]
                if t=="log":
                    _,text,clr=msg; self._log(text,clr)
                elif t=="tok":
                    _,sym,tf,meta=msg
                    self._insert_row(meta)
                    ae=meta["acc_ens"]
                    self._log(f"✓ {sym:8s} {tf:4s}  ens={ae:.3f}  rf={meta['acc_rf']:.3f}  "
                              f"xgb={meta['acc_xgb']:.3f}  prec@65%={meta['prec_hi65']:.3f}  "
                              f"n={meta['samples']:,}","G" if ae>=0.57 else "Y")
                    self._done+=1; self._prog.config(value=self._done)
                    self._prog_d.config(text=f"{self._done}/{self._total}",fg=self.TXT)
                elif t=="tw":
                    _,sym,tf,msg2=msg
                    self._log(f"⚠ {sym} {tf}: {msg2}","Y")
                    self._done+=1; self._prog.config(value=self._done)
                elif t=="tdone":
                    self._training=False
                    self._btn_train.config(state="normal")
                    self._btn_tstop.config(state="disabled",bg=self.BDR,fg=self.TXT)
                    self._prog_lbl.config(text="Trening završen!",fg=self.GRN)
                    good=sum(1 for m in self._results.values() if m["acc_ens"]>=0.57)
                    self._log(f"══ TRENING ZAVRŠEN ══  {self._done} modela  |  {good} sa ens≥57%","B")
                    self._status(f"Završeno — {self._done} modela, {good} sa ens≥57%")
                elif t=="prog_lbl":
                    self._prog_lbl.config(text=msg[1],fg=self.ACC)
                    self._status(msg[1])
                elif t=="signal":
                    r=msg[1]
                    sig=r["signal"]
                    tag="buy" if sig=="BUY" else "sell" if sig=="SELL" else "hold"
                    # "SIGNAL" = model detektovao ulaz; "OTVOREN" se upisuje tek kad je trade zaista otvoren
                    akc="SIGNAL" if sig!="HOLD" and self._trade_en.get() else "–"
                    # Drži samo zadnjih 100 redova
                    items=self._stree.get_children()
                    if len(items)>=100: self._stree.delete(items[-1])
                    self._stree.insert("",0,tags=(tag,),values=(
                        r["ts"],r["symbol"],r["tf"],sig,
                        f"{r['confidence']:.1%}",
                        f"{r['p_buy']:.3f}",f"{r['p_sell']:.3f}",
                        r["bv"],r["sv"],r["rsi"],r["adx"],akc))
                elif t=="trade_opened":
                    self._daily_trades+=1
                    self.root.after(0,self._refresh_positions)
                elif t=="refresh_pos":
                    self.root.after(0,self._refresh_positions)
                elif t=="news_bar":
                    txt = msg[1]
                    clr = self.RED if "🔴" in txt else (self.YLW if "🟡" in txt else self.GRN)
                    self._lbl_news.config(text=f"  {txt}  ", fg=clr)
        except queue.Empty: pass
        self.root.after(120,self._poll)

    # ══════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        self._training=False; self._live_run=False
        if self._mt5:
            try: self._mt5.shutdown()
            except: pass
        self.root.after(400,self.root.destroy)

    def run(self): self.root.mainloop()


if __name__=="__main__":
    # Kad radi kao exe — sve greške idu u log fajl pored exe-a
    if getattr(sys, "frozen", False):
        import traceback, logging
        _log_path = Path(sys.executable).parent / "ml_trader_crash.log"
        logging.basicConfig(filename=str(_log_path), level=logging.ERROR,
                            format="%(asctime)s %(levelname)s\n%(message)s\n")
        def _eh(exc_type, exc_val, exc_tb):
            msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            logging.error(msg)
            try:
                import tkinter.messagebox as mb
                mb.showerror("ML Trader — Greška", msg[:800])
            except: pass
        sys.excepthook = _eh

    try:
        App().run()
    except Exception:
        import traceback
        msg = traceback.format_exc()
        err_path = (Path(sys.executable).parent if getattr(sys,"frozen",False)
                    else Path(__file__).parent) / "ml_trader_crash.log"
        with open(err_path, "a") as f:
            f.write(f"\n{'='*60}\n{datetime.now()}\n{msg}\n")
        try:
            import tkinter.messagebox as mb
            mb.showerror("ML Trader — Greška pri pokretanju", msg[:800])
        except: pass
