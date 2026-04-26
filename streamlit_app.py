import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Analysis Tool", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

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
    hist = macd - signal
    return macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]


def calc_bollinger(prices, period=20):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return (ma + 2 * std).iloc[-1], ma.iloc[-1], (ma - 2 * std).iloc[-1]


def build_signal(rsi, macd_hist, price, ma25, ma75, bb_upper, bb_lower):
    score = 0
    signals = []

    if not np.isnan(rsi):
        if rsi < 30:
            score += 1
            signals.append({'label': 'RSI', 'value': f'{rsi:.1f}', 'type': 'bullish', 'note': '売られすぎ圏'})
        elif rsi > 70:
            score -= 1
            signals.append({'label': 'RSI', 'value': f'{rsi:.1f}', 'type': 'bearish', 'note': '買われすぎ圏'})
        else:
            signals.append({'label': 'RSI', 'value': f'{rsi:.1f}', 'type': 'neutral', 'note': '中立'})

    if not np.isnan(macd_hist):
        if macd_hist > 0:
            score += 1
            signals.append({'label': 'MACD', 'value': f'{macd_hist:.4f}', 'type': 'bullish', 'note': 'ヒスト＋（上昇）'})
        else:
            score -= 1
            signals.append({'label': 'MACD', 'value': f'{macd_hist:.4f}', 'type': 'bearish', 'note': 'ヒスト－（下降）'})

    if not np.isnan(ma25):
        if price > ma25:
            score += 1
            signals.append({'label': 'MA25', 'value': f'{ma25:.2f}', 'type': 'bullish', 'note': '株価 > 25日線'})
        else:
            score -= 1
            signals.append({'label': 'MA25', 'value': f'{ma25:.2f}', 'type': 'bearish', 'note': '株価 < 25日線'})

    if not np.isnan(ma75):
        if price > ma75:
            score += 1
            signals.append({'label': 'MA75', 'value': f'{ma75:.2f}', 'type': 'bullish', 'note': '株価 > 75日線'})
        else:
            score -= 1
            signals.append({'label': 'MA75', 'value': f'{ma75:.2f}', 'type': 'bearish', 'note': '株価 < 75日線'})

    if not (np.isnan(bb_upper) or np.isnan(bb_lower)) and bb_upper != bb_lower:
        pos = (price - bb_lower) / (bb_upper - bb_lower)
        if pos < 0.2:
            score += 1
            signals.append({'label': 'BB', 'value': f'{pos:.2f}', 'type': 'bullish', 'note': '下限バンド付近'})
        elif pos > 0.8:
            score -= 1
            signals.append({'label': 'BB', 'value': f'{pos:.2f}', 'type': 'bearish', 'note': '上限バンド付近'})
        else:
            signals.append({'label': 'BB', 'value': f'{pos:.2f}', 'type': 'neutral', 'note': 'バンド中間'})

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


# 5分キャッシュ
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


# --- UI ---

st.title("📈 Stock Analysis Tool")
st.caption("Technical indicators and latest news — all in one view.")

col1, col2 = st.columns([4, 1])
with col1:
    ticker_input = st.text_input(
        "ticker", placeholder="Enter ticker symbol (e.g. AAPL, NVDA, 7203.T)",
        label_visibility="collapsed"
    )
with col2:
    analyze = st.button("Analyze", use_container_width=True, type="primary")

if analyze and ticker_input:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Fetching data for {ticker}..."):
        df, name, news = fetch_stock_data(ticker)

    if df is None:
        st.error(f"No data found for {ticker}. Please check the ticker symbol.")
        st.stop()

    close = df['Close']
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else price
    change = price - prev
    change_pct = change / prev * 100 if prev else 0

    st.subheader(f"{name}  ({ticker})")
    m1, m2, m3 = st.columns(3)
    m1.metric("Price", f"{price:,.2f}")
    m2.metric("Change", f"{change:+.2f}")
    m3.metric("Change (%)", f"{change_pct:+.2f}%")

    # チャート（ローソク足 + MA + 出来高）
    chart_df = df.tail(180).copy()
    ma25_s = close.rolling(25).mean().tail(180)
    ma75_s = close.rolling(75).mean().tail(180)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'], high=chart_df['High'],
        low=chart_df['Low'], close=chart_df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=ma25_s, name='MA25',
        line=dict(color='#ffa726', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=ma75_s, name='MA75',
        line=dict(color='#ab47bc', width=1.5)
    ), row=1, col=1)

    bar_colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(chart_df['Close'], chart_df['Open'])]
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df['Volume'],
        name='Volume', marker_color=bar_colors, opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#f8fafc',
        font_color='#1e293b',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_xaxes(gridcolor='#cbd5e1')
    fig.update_yaxes(gridcolor='#cbd5e1')

    st.plotly_chart(fig, use_container_width=True)

    # テクニカル指標 & シグナル
    rsi = calc_rsi(close)
    macd_val, sig_val, hist_val = calc_macd(close)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    ma25_val = float(close.rolling(25).mean().iloc[-1])
    ma75_val = float(close.rolling(75).mean().iloc[-1])
    signal = build_signal(rsi, hist_val, price, ma25_val, ma75_val, bb_upper, bb_lower)

    st.subheader("Technical Indicators")

    verdict = signal['verdict']
    note_map = {'bullish': 'Oversold', 'bearish': 'Overbought', 'neutral': 'Neutral',
                'ヒスト＋（上昇）': 'Positive histogram', 'ヒスト－（下降）': 'Negative histogram',
                '株価 > 25日線': 'Price above MA25', '株価 < 25日線': 'Price below MA25',
                '株価 > 75日線': 'Price above MA75', '株価 < 75日線': 'Price below MA75',
                '下限バンド付近': 'Near lower band', '上限バンド付近': 'Near upper band', 'バンド中間': 'Mid band'}
    verdict_label = {'buy': '🟢 Buy Signal', 'sell': '🔴 Sell Signal', 'hold': '🟡 Hold'}[verdict]
    score_text = f"  —  Score: {signal['score']}/{signal['max_score']}"
    if verdict == 'buy':
        st.success(verdict_label + score_text)
    elif verdict == 'sell':
        st.error(verdict_label + score_text)
    else:
        st.warning(verdict_label + score_text)

    type_icon = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}
    cols = st.columns(len(signal['signals']))
    for col, s in zip(cols, signal['signals']):
        with col:
            st.metric(
                label=f"{type_icon[s['type']]} {s['label']}",
                value=s['value'],
                help=note_map.get(s['note'], s['note'])
            )

    if news:
        st.subheader("Related News")
        for item in news:
            date_str = f"  {item['date']}" if item['date'] else ''
            pub_str = f"**{item['publisher']}**{date_str}  " if item['publisher'] else ''
            st.markdown(f"{pub_str}[{item['title']}]({item['url']})")

    st.caption("⚠️ For informational purposes only. Not financial advice.")
