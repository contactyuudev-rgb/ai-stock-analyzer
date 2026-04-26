# 📈 AI Stock Analyzer

A stock analysis tool built with Streamlit that combines technical indicators and the latest news in a single view.

## Features

- **Candlestick chart** with MA25 / MA75 overlay and volume
- **Technical indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Buy / Hold / Sell signal** based on combined indicator scoring
- **Latest news** related to the selected stock
- Supports any ticker available on Yahoo Finance (US, JP, etc.)

## Demo

[▶ Live App](https://appapppy-vqqtnjnz4jvvqmdbscqup5.streamlit.app/)

## Getting Started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech Stack

- [Streamlit](https://streamlit.io)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Plotly](https://plotly.com)
- Python / pandas / numpy

## License

MIT
