import finplot as fplt
import datetime

fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = datetime.timezone.utc

def view_data(name, df, save_img=False, other_data=None):
    def save():
        fplt.screenshot(open(f'img/days/{name}.png', 'wb'))
        
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    if other_data is not None:
        for data, style, color in other_data:
            fplt.plot(data, style=style, color=color, width=2.0)
    
    if save_img:
        fplt.timer_callback(save, .2, single_shot=True)
        fplt.timer_callback(fplt.close, .4, single_shot=True)
    fplt.show()
    # df[["test_open", "test_close", "test_high", "test_low", "test_5", "test_20", "test_120"]] = scaler.fit_transform(df[['Open', 'Close', 'High', 'Low', "5", "20", "120"]])
    
    # fplt.candlestick_ochl(df[["test_open", "test_close", "test_high", "test_low"]], ax=ax1)
    # fplt.plot(df["test_5"], color='#973131', width=2.0, ax=ax1)
    # fplt.plot(df["test_20"], color='#5A639C', width=2.0, ax=ax1)
    # fplt.plot(df["test_120"], color='#A0937D', width=2.0, ax=ax1)
    # fplt.volume_ocv(df[['Open', 'Close', 'Amount']], ax=ax1)