import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="株式分析ツール", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# --- Localization ---

UI = {
    'ja': {
        'title':      '📈 株式分析ツール',
        'caption':    'テクニカル指標と最新ニュースをまとめて確認',
        'us_label':   '🇺🇸 米国株',
        'jp_label':   '🇯🇵 日本株',
        'placeholder':'ティッカーを入力（例: AAPL, NVDA, 7203.T）',
        'btn':        '分析する',
        'spinner':    'のデータを取得中...',
        'not_found':  'のデータが見つかりません。ティッカーシンボルを確認してください。',
        'price':      '現在値',
        'change':     '前日比',
        'change_pct': '前日比（%）',
        'chart_price':'株価',
        'chart_vol':  '出来高',
        'section':    'テクニカル分析',
        'score':      'スコア',
        'of':         '指標中',
        'buy':        '🟢 買いシグナル',
        'sell':       '🔴 売りシグナル',
        'hold':       '🟡 様子見',
        'gauge_l':    '強い売り',
        'gauge_c':    '中立',
        'gauge_r':    '強い買い',
        'guide':      '📖 各指標の見方',
        'news':       '関連ニュース',
        'footer':     '⚠️ 本ツールは情報提供を目的としており、投資助言ではありません。',
    },
    'en': {
        'title':      '📈 Stock Analysis Tool',
        'caption':    'Technical indicators and latest news — all in one view.',
        'us_label':   '🇺🇸 US Stocks',
        'jp_label':   '🇯🇵 Japan Stocks',
        'placeholder':'Enter ticker (e.g. AAPL, NVDA, 7203.T)',
        'btn':        'Analyze',
        'spinner':    'Fetching data for',
        'not_found':  'No data found for',
        'price':      'Price',
        'change':     'Change',
        'change_pct': 'Change (%)',
        'chart_price':'Price',
        'chart_vol':  'Volume',
        'section':    'Technical Analysis',
        'score':      'Score',
        'of':         'of',
        'buy':        '🟢 Buy Signal',
        'sell':       '🔴 Sell Signal',
        'hold':       '🟡 Hold',
        'gauge_l':    'Strong Sell',
        'gauge_c':    'Neutral',
        'gauge_r':    'Strong Buy',
        'guide':      '📖 Indicator Guide',
        'news':       'Related News',
        'footer':     '⚠️ For informational purposes only. Not financial advice.',
    },
}

NOTE_MAP = {
    'oversold':   {'ja': '売られすぎ圏',     'en': 'Oversold'},
    'overbought': {'ja': '買われすぎ圏',     'en': 'Overbought'},
    'neutral':    {'ja': '中立',            'en': 'Neutral'},
    'macd_pos':   {'ja': 'ヒスト＋（上昇）', 'en': 'Positive (bullish)'},
    'macd_neg':   {'ja': 'ヒスト－（下降）', 'en': 'Negative (bearish)'},
    'above_ma25': {'ja': '株価 > 25日線',   'en': 'Price > MA25'},
    'below_ma25': {'ja': '株価 < 25日線',   'en': 'Price < MA25'},
    'above_ma75': {'ja': '株価 > 75日線',   'en': 'Price > MA75'},
    'below_ma75': {'ja': '株価 < 75日線',   'en': 'Price < MA75'},
    'near_lower': {'ja': '下限バンド付近',   'en': 'Near lower band'},
    'near_upper': {'ja': '上限バンド付近',   'en': 'Near upper band'},
    'mid_band':   {'ja': 'バンド中間',       'en': 'Mid band'},
}

INDICATOR_HELP = {
    'RSI': {
        'ja': ('RSI（相対力指数）',        '過去14日間の値上がり・値下がりの比率。70超で買われすぎ、30未満で売られすぎのサイン。'),
        'en': ('RSI (Relative Strength Index)', 'Ratio of gains to losses over 14 days. Above 70 = overbought, below 30 = oversold.'),
    },
    'MACD': {
        'ja': ('MACD（移動平均収束拡散法）', '短期EMA(12日)と長期EMA(26日)の差。ヒストグラムがプラスなら上昇モメンタム。'),
        'en': ('MACD',                     'Difference of 12-day and 26-day EMA. Positive histogram = upward momentum.'),
    },
    'MA25': {
        'ja': ('25日移動平均線',  '直近25日間の終値の平均。株価がこの線より上なら短期的に強い状態。'),
        'en': ('25-day MA',      '25-day closing price average. Price above = short-term bullish.'),
    },
    'MA75': {
        'ja': ('75日移動平均線',  '直近75日間の終値の平均。株価がこの線より上なら中期的に強い状態。'),
        'en': ('75-day MA',      '75-day closing price average. Price above = mid-term bullish.'),
    },
    'BB': {
        'ja': ('ボリンジャーバンド', '20日移動平均±2σの価格帯。下限付近は反発の可能性、上限付近は反落の可能性。'),
        'en': ('Bollinger Bands',  '20-day MA ±2σ band. Near lower = possible bounce, near upper = possible pullback.'),
    },
}

QUICK_US = [('AAPL','Apple'),('NVDA','NVIDIA'),('TSLA','Tesla'),('MSFT','Microsoft'),('AMZN','Amazon'),('META','Meta')]
QUICK_JP = [('7203.T','トヨタ'),('6758.T','ソニー'),('9984.T','SoftBank'),('7974.T','任天堂'),('6861.T','キーエンス'),('8306.T','三菱UFJ')]

TYPE_ICON = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}


# --- 計算ロジック ---

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return (100 - 100 / (1 + rs)).iloc[-1]


def calc_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd.iloc[-1], signal.iloc[-1], (macd - signal).iloc[-1]


def calc_bollinger(prices, period=20):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return (ma + 2 * std).iloc[-1], ma.iloc[-1], (ma - 2 * std).iloc[-1]


def build_signal(rsi, macd_hist, price, ma25, ma75, bb_upper, bb_lower):
    score, signals = 0, []

    if not np.isnan(rsi):
        if rsi < 30:
            score += 1; signals.append({'label':'RSI','value':f'{rsi:.1f}','type':'bullish','note_key':'oversold'})
        elif rsi > 70:
            score -= 1; signals.append({'label':'RSI','value':f'{rsi:.1f}','type':'bearish','note_key':'overbought'})
        else:
            signals.append({'label':'RSI','value':f'{rsi:.1f}','type':'neutral','note_key':'neutral'})

    if not np.isnan(macd_hist):
        if macd_hist > 0:
            score += 1; signals.append({'label':'MACD','value':f'{macd_hist:.4f}','type':'bullish','note_key':'macd_pos'})
        else:
            score -= 1; signals.append({'label':'MACD','value':f'{macd_hist:.4f}','type':'bearish','note_key':'macd_neg'})

    if not np.isnan(ma25):
        if price > ma25:
            score += 1; signals.append({'label':'MA25','value':f'{ma25:.2f}','type':'bullish','note_key':'above_ma25'})
        else:
            score -= 1; signals.append({'label':'MA25','value':f'{ma25:.2f}','type':'bearish','note_key':'below_ma25'})

    if not np.isnan(ma75):
        if price > ma75:
            score += 1; signals.append({'label':'MA75','value':f'{ma75:.2f}','type':'bullish','note_key':'above_ma75'})
        else:
            score -= 1; signals.append({'label':'MA75','value':f'{ma75:.2f}','type':'bearish','note_key':'below_ma75'})

    if not (np.isnan(bb_upper) or np.isnan(bb_lower)) and bb_upper != bb_lower:
        pos = (price - bb_lower) / (bb_upper - bb_lower)
        if pos < 0.2:
            score += 1; signals.append({'label':'BB','value':f'{pos:.2f}','type':'bullish','note_key':'near_lower'})
        elif pos > 0.8:
            score -= 1; signals.append({'label':'BB','value':f'{pos:.2f}','type':'bearish','note_key':'near_upper'})
        else:
            signals.append({'label':'BB','value':f'{pos:.2f}','type':'neutral','note_key':'mid_band'})

    verdict = 'buy' if score >= 3 else 'sell' if score <= -3 else 'hold'
    return {'score': score, 'max_score': len(signals), 'verdict': verdict, 'signals': signals}


def parse_news(raw_news):
    items = []
    for n in raw_news[:8]:
        try:
            content = n.get('content', {})
            title = content.get('title') or n.get('title', '')
            provider = content.get('provider', {}).get('displayName') or n.get('publisher', '')
            url = (content.get('clickThroughUrl') or {}).get('url') or n.get('link', '')
            pub = content.get('pubDate') or ''
            if title:
                items.append({'title': title, 'publisher': provider, 'url': url, 'date': pub[:10] if pub else ''})
        except Exception:
            pass
    return items


@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='1y')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None, None, None
    try:
        name = t.info.get('longName') or t.info.get('shortName') or ticker
    except Exception:
        name = ticker
    try:
        news = parse_news(t.news or [])
    except Exception:
        news = []
    return df, name, news


def render_signal_gauge(score, max_score, verdict, t):
    pct = max(2, min(98, (score + max_score) / (2 * max_score) * 100))
    if verdict == 'buy':
        color, bg = '#22c55e', 'rgba(34,197,94,0.08)'
        label = t['buy']
    elif verdict == 'sell':
        color, bg = '#ef4444', 'rgba(239,68,68,0.08)'
        label = t['sell']
    else:
        color, bg = '#eab308', 'rgba(234,179,8,0.08)'
        label = t['hold']

    st.markdown(f"""
<div style="background:{bg};border:1px solid {color}44;border-radius:16px;padding:20px 24px;margin:8px 0 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
    <div style="font-size:22px;font-weight:700;color:{color};">{label}</div>
    <div style="font-size:13px;color:#64748b;background:white;border-radius:8px;padding:6px 14px;border:1px solid #e2e8f0;">
      {t['score']}: <b style="color:{color};font-size:15px;">{score:+d}</b> / {max_score} {t['of']}
    </div>
  </div>
  <div style="position:relative;padding:10px 0 6px;">
    <div style="background:linear-gradient(to right,#ef4444,#f97316,#eab308,#84cc16,#22c55e);
                border-radius:99px;height:12px;"></div>
    <div style="position:absolute;left:calc({pct:.1f}% - 13px);top:4px;
                width:28px;height:28px;border-radius:50%;
                background:white;border:3px solid {color};
                box-shadow:0 2px 10px {color}66;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-top:10px;">
    <span>{t['gauge_l']}</span><span>{t['gauge_c']}</span><span>{t['gauge_r']}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# --- Session state init ---

if 'analyze_ticker' not in st.session_state:
    st.session_state.analyze_ticker = None
if 'lang' not in st.session_state:
    st.session_state.lang = 'ja'
if 'sel_ver' not in st.session_state:
    st.session_state.sel_ver = 0


# --- Header ---

header_col, lang_col = st.columns([7, 1])
with header_col:
    t = UI[st.session_state.lang]
    st.title(t['title'])
    st.caption(t['caption'])
with lang_col:
    st.write("")
    st.write("")
    lang_choice = st.radio("", ['🇯🇵 JP', '🇺🇸 EN'], horizontal=True,
                           label_visibility="collapsed",
                           index=0 if st.session_state.lang == 'ja' else 1)
    new_lang = 'ja' if '🇯🇵' in lang_choice else 'en'
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

t = UI[st.session_state.lang]
lang = st.session_state.lang

# クイック選択（Combobox）
ver = st.session_state.sel_ver
qc1, qc2 = st.columns(2)
with qc1:
    us_options = [''] + [f"{label}  ({sym})" for sym, label in QUICK_US]
    us_sel = st.selectbox(t['us_label'], us_options, index=0, key=f'us_{ver}')
    if us_sel:
        st.session_state.analyze_ticker = us_sel.split('(')[-1].rstrip(')').strip()
        st.session_state.sel_ver += 1
        st.rerun()
with qc2:
    jp_options = [''] + [f"{label}  ({sym})" for sym, label in QUICK_JP]
    jp_sel = st.selectbox(t['jp_label'], jp_options, index=0, key=f'jp_{ver}')
    if jp_sel:
        st.session_state.analyze_ticker = jp_sel.split('(')[-1].rstrip(')').strip()
        st.session_state.sel_ver += 1
        st.rerun()

# 入力フォーム
col1, col2 = st.columns([4, 1])
with col1:
    ticker_input = st.text_input("ticker", placeholder=t['placeholder'], label_visibility="collapsed")
with col2:
    if st.button(t['btn'], use_container_width=True, type="primary"):
        if ticker_input:
            st.session_state.analyze_ticker = ticker_input.strip().upper()
            st.rerun()

ticker = st.session_state.analyze_ticker
if not ticker:
    st.stop()

spinner_msg = f"{ticker}{t['spinner']}" if lang == 'ja' else f"{t['spinner']} {ticker}..."
with st.spinner(spinner_msg):
    df, name, news = fetch_stock_data(ticker)

if df is None:
    err = f"「{ticker}」{t['not_found']}" if lang == 'ja' else f"{t['not_found']} {ticker}."
    st.error(err)
    st.stop()

close = df['Close']
price  = float(close.iloc[-1])
prev   = float(close.iloc[-2]) if len(close) >= 2 else price
change = price - prev
change_pct = change / prev * 100 if prev else 0

st.subheader(f"{name}  （{ticker}）")
m1, m2, m3 = st.columns(3)
m1.metric(t['price'],      f"{price:,.2f}")
m2.metric(t['change'],     f"{change:+.2f}")
m3.metric(t['change_pct'], f"{change_pct:+.2f}%")

# チャート
chart_df = df.tail(180).copy()
ma25_s = close.rolling(25).mean().tail(180)
ma75_s = close.rolling(75).mean().tail(180)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.72, 0.28], vertical_spacing=0.04)

fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df['Open'], high=chart_df['High'],
    low=chart_df['Low'],   close=chart_df['Close'],
    name=t['chart_price'],
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=ma25_s, name='MA25',
                         line=dict(color='#ffa726', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=ma75_s, name='MA75',
                         line=dict(color='#ab47bc', width=1.5)), row=1, col=1)

bar_colors = ['#26a69a' if c >= o else '#ef5350'
              for c, o in zip(chart_df['Close'], chart_df['Open'])]
fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'],
                     name=t['chart_vol'], marker_color=bar_colors, opacity=0.6), row=2, col=1)

fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                  plot_bgcolor='#f8fafc', paper_bgcolor='#f8fafc', font_color='#1e293b',
                  legend=dict(orientation='h', yanchor='bottom', y=1.02),
                  margin=dict(l=0, r=0, t=30, b=0))
fig.update_xaxes(gridcolor='#cbd5e1')
fig.update_yaxes(gridcolor='#cbd5e1')
st.plotly_chart(fig, use_container_width=True)

# テクニカル分析
rsi = calc_rsi(close)
_, _, hist_val = calc_macd(close)
bb_upper, _, bb_lower = calc_bollinger(close)
ma25_val = float(close.rolling(25).mean().iloc[-1])
ma75_val = float(close.rolling(75).mean().iloc[-1])
signal = build_signal(rsi, hist_val, price, ma25_val, ma75_val, bb_upper, bb_lower)

st.subheader(t['section'])
render_signal_gauge(signal['score'], signal['max_score'], signal['verdict'], t)

cols = st.columns(len(signal['signals']))
for col, s in zip(cols, signal['signals']):
    with col:
        st.metric(
            label=f"{TYPE_ICON[s['type']]} {s['label']}",
            value=s['value'],
            help=NOTE_MAP[s['note_key']][lang]
        )

with st.expander(t['guide']):
    for key, texts in INDICATOR_HELP.items():
        title, desc = texts[lang]
        st.markdown(f"**{title}**  \n{desc}")

if news:
    st.subheader(t['news'])
    for item in news:
        date_str = f"  {item['date']}" if item['date'] else ''
        pub_str  = f"**{item['publisher']}**{date_str}  " if item['publisher'] else ''
        st.markdown(f"{pub_str}[{item['title']}]({item['url']})")

st.caption(t['footer'])
