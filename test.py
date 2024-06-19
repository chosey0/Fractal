
import finplot as fplt
import yfinance as yf

data = [(instrument, yf.download(instrument, '2020-10-01')) for instrument in ('AAPL','GOOG','TSLA')]
for i,(instrument_a,dfa) in enumerate(data):
    for instrument_b,dfb in data[i+1:]:
        ax = fplt.create_plot(instrument_a+' vs. '+instrument_b+' (green/brown)', maximize=False)
        dfa['Open Close High Low'.split()].plot(kind='candle', ax=ax)
        pb = dfb['Open Close High Low'.split()].plot(kind='candle', ax=ax.overlay(scale=1.0))
        pb.colors['bull_body'] = '#0f0'
        pb.colors['bear_body'] = '#630'
fplt.show()

# del webencodings, peewee, multitasking, urllib3, soupsieve, platformdirs, lxml, idna, html5lib, frozendict, charset-normalizer, certifi, requests, beautifulsoup4, yfinance