from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import SingleIntervalTicker, ColumnDataSource, HoverTool
from bokeh.models import LinearAxis, Range1d, Legend
from bokeh.layouts import column
import io
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pandastable import Table

plt.style.use('ggplot')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
matplotlib.use('TkAgg')


# 爬取 Yahoo Finance 股價
def YahooData(ticker, start, end):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/\
        9.0.2 Safari/601.3.9'
    }

    url = "https://query1.finance.yahoo.com/v7/finance/download/" + str(ticker)
    x = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    y = int(datetime.strptime(end, '%Y-%m-%d').timestamp())
    url += "?period1=" + str(x) + "&period2=" + str(y) + "&interval=1d&events=history&includeAdjustedClose=true"

    r = requests.get(url, headers=headers)
    pad = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    return pad


# 將不同的技術指標寫成class
class TA:
    def __init__(self, data: pd.DataFrame):
        self.op = data['Open']
        self.hi = data['High']
        self.lo = data['Low']
        self.cl = data['Close']
        self.vo = data['Volume']

    def MA(self, n):
        return self.cl.rolling(window=n).mean()

    def BBands_original(self, n, zu, zd):
        ma = self.cl.rolling(window=n).mean()
        sd = self.cl.rolling(window=n).std()
        up = ma + zu * sd
        lo = ma - zd * sd
        bands = pd.concat([self.cl, ma, up, lo], axis=1)
        bands.columns = ['Close', 'Mid', 'Up', 'Lo']
        return bands

    def BBands_modified(self, n, zu, zd):
        a = 2 / (n + 1)
        Mt = self.cl.ewm(span=n).mean()
        Ut = Mt.ewm(span=n).mean()
        Dt = ((2 - a) * Mt - Ut) / (1 - a)
        mt = abs(self.cl - Dt).ewm(span=n).mean()
        ut = mt.ewm(span=n).mean()
        dt = ((2 - a) * mt - ut) / (1 - a)
        bu = Dt + zu * dt
        bl = Dt - zd * dt
        Dt[0:19] = np.nan
        bu[0:19] = np.nan
        bl[0:19] = np.nan
        bands = pd.concat([self.cl, Dt, bu, bl], axis=1)
        bands.columns = ['Close', 'Mid', 'Up', 'Lo']
        return bands

    def kd(self, n):
        result = (self.cl - self.cl.rolling(window=n).min()) \
                 / (self.cl.rolling(window=n).max() - self.cl.rolling(window=n).min()) * 100
        result[0:8] = 50
        k = result.ewm(span=5).mean()
        d = k.ewm(span=5).mean()
        frame = pd.concat([k, d], axis=1)
        frame.columns = ['K', 'D']
        return frame

    # Relative Strength Index
    def rsi(self, n):
        chg = pd.Series(self.cl - self.cl.shift(1))
        chg_pos = pd.Series(index=chg.index, data=chg[chg > 0])
        chg_pos = chg_pos.fillna(0)
        chg = chg.abs()
        a_chg = chg.rolling(window=n).mean()
        a_chg_pos = chg_pos.rolling(window=n).mean()
        rate = pd.DataFrame(a_chg_pos / a_chg * 100)
        rate.columns = ['value']
        return rate


# 將不同的交易訊號函數寫成class
class Signals:
    def __init__(self, data: pd.DataFrame):
        self.op = data['Open']
        self.hi = data['High']
        self.lo = data['Low']
        self.cl = data['Close']
        self.vo = data['Volume']
        self.ta = TA(data)

    def band_original_method_1(self, zu, zd):
        band = self.ta.BBands_original(20, zu, zd)
        long_signal = pd.Series(np.where(self.cl > band['Up'], 1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(self.cl < band['Lo'], -1, np.nan), index=self.cl.index, name='short')

        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals

    def band_original_method_2(self, zu, zd):
        band = self.ta.BBands_original(20, zu, zd)
        long_signal = pd.Series(np.where(((self.cl > band['Up']) &
                                          (self.cl.shift(1) < band['Up'].shift(1)) &
                                          (self.vo > self.vo.rolling(20).mean())),
                                         1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(((self.cl < band['Lo']) &
                                           (self.cl.shift(1) > band['Lo'].shift(1)) &
                                           (self.vo > self.vo.rolling(20).mean())),
                                          -1, np.nan), index=self.cl.index, name='short')

        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals

    def band_modified_method_1(self, zu, zd):
        band = self.ta.BBands_modified(20, zu, zd)
        long_signal = pd.Series(np.where(self.cl > band['Up'], 1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(self.cl < band['Lo'], -1, np.nan), index=self.cl.index, name='short')

        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals

    def band_modified_method_2(self, zu, zd):
        band = self.ta.BBands_modified(20, zu, zd)
        long_signal = pd.Series(np.where(((self.cl > band['Up']) &
                                          (self.cl.shift(1) < band['Up'].shift(1)) &
                                          (self.vo > self.vo.rolling(20).mean())),
                                         1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(((self.cl < band['Lo']) &
                                           (self.cl.shift(1) > band['Lo'].shift(1)) &
                                           (self.vo > self.vo.rolling(20).mean())),
                                          -1, np.nan), index=self.cl.index, name='short')

        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals

    def MA_breakthrough(self, n, m):
        long = max(n, m)
        short = min(m, n)
        longer_ma = self.ta.MA(long)
        shorter_ma = self.ta.MA(short)
        MA = pd.concat([longer_ma, shorter_ma], axis=1)
        MA.columns = ['longer', 'shorter']
        long_signal = pd.Series(np.where(MA['shorter'] >= MA['longer'], 1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(MA['shorter'] < MA['longer'], -1, np.nan), index=self.cl.index, name='short')

        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals

    def kd_breakthrough(self, n):
        kd = self.ta.kd(n)
        long_signal = pd.Series(np.where(kd['K'] <= kd['D'], 1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(kd['K'] > kd['D'], -1, np.nan), index=self.cl.index, name='short')

        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals

    def RSI(self, lo, up):
        upper = max(lo, up)
        lower = min(lo, up)
        rsi = self.ta.rsi(12)
        long_signal = pd.Series(np.where(rsi['value'] > upper, 1, np.nan), index=self.cl.index, name='long')
        short_signal = pd.Series(np.where(rsi['value'] < lower, -1, np.nan), index=self.cl.index, name='short')
        
        signals = pd.concat([long_signal, short_signal], axis=1)
        signals['all'] = signals['long'].fillna(0) + signals['short'].fillna(0)
        signals['all'] = signals['all'].replace(0, np.nan).ffill().fillna(0)
        signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=signals.index)
        signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=signals.index)
        signals = signals.shift(2).fillna(0)
        return signals


# 報酬率採用每日報酬率累積總和之方式計算
def Returns(data: pd.DataFrame, signal: pd.DataFrame):
    discount = 0.6
    fee = 1.425 / 1000 * discount
    ret_df = pd.concat([data['Close'].pct_change(), signal['all'], signal['long'], signal['short']], axis=1)
    ret_df.columns = ['d_ret', 'position_a', 'position_l', 'position_s']
    ret_df['return_a'] = ret_df['d_ret'] * ret_df['position_a']
    ret_df['return_a'] = np.where((ret_df['position_a'] != ret_df['position_a'].shift(1)), ret_df['return_a'] - 2 * fee,
                                  ret_df['return_a'])
    ret_df['return_l'] = ret_df['d_ret'] * ret_df['position_l']
    ret_df['return_l'] = np.where((ret_df['position_l'] != ret_df['position_l'].shift(1)), ret_df['return_l'] - fee,
                                  ret_df['return_l'])
    ret_df['return_s'] = ret_df['d_ret'] * ret_df['position_s']
    ret_df['return_s'] = np.where((ret_df['position_s'] != ret_df['position_s'].shift(1)), ret_df['return_s'] - fee,
                                  ret_df['return_s'])
    return ret_df


# 年週期分析
class YearlyAnalysis:
    def __init__(self, data):
        data['year'] = data.index.year
        Benchmark = round(data.groupby('year')['d_ret'].sum() * 100, 2)
        Strategy = round(data.groupby('year')['return_a'].sum() * 100, 2)
        Diff = round(Strategy - Benchmark, 2)
        self.YR = pd.concat([Strategy, Benchmark, Diff], axis=1)
        self.YR.columns = ['Strategy', 'Benchmark', 'Diff']

    def graphing(self):
        plt.close()
        plt.bar(self.YR.index, self.YR['Strategy'], alpha=0.5, label='Strategy')
        plt.bar(self.YR.index, self.YR['Benchmark'], alpha=0.5, label='Benchmark')
        plt.legend()
        plt.show()


# 策略損益圖且與 Benchmark 比較
def PnL(data: pd.DataFrame):
    plt.close()
    plt.plot(data['return_a'].cumsum(), label='Strategy')
    plt.plot(data['d_ret'].cumsum(), label='Benchmark')
    plt.fill_between(DrawDown(data['return_a']).index, DrawDown(data['return_a']), 0, label='Strategy drowdown',
                     color='#d62728', alpha=0.15)
    plt.fill_between(DrawDown(data['d_ret']).index, DrawDown(data['d_ret']), 0, label='Benchmark drawdown',
                     color='#1f77b4', alpha=0.15)
    plt.ylabel('Cumulative Return (100%)')
    plt.legend()
    plt.show()


# 與 Benchmark 差異圖
def PnL2(data: pd.DataFrame):
    plt.close()
    plt.plot(data['return_a'].cumsum() - data['d_ret'].cumsum(), label='Difference of Strategy and Benchmark')
    plt.ylabel('Cumulative Return (100%)')
    plt.legend()
    plt.show()


# 多空損益分析圖
def PnL3(data: pd.DataFrame):
    plt.close()
    plt.plot(data['return_a'].cumsum(), label='PnL of Total', color='black')
    plt.plot(data['return_l'].cumsum(), label='PnL of Long', color='#d62728')
    plt.plot(data['return_s'].cumsum(), label='PnL of Short', color='#1f77b4')
    plt.fill_between(DrawDown(data['return_a']).index, DrawDown(data['return_a']), 0, label='Total drawdown',
                     color='black', alpha=0.3, hatch='///')
    plt.fill_between(DrawDown(data['return_l']).index, DrawDown(data['return_l']), 0, label='Long drowdown',
                     color='#d62728', alpha=0.15)
    plt.fill_between(DrawDown(data['return_s']).index, DrawDown(data['return_s']), 0, label='Short drawdown',
                     color='#1f77b4', alpha=0.15)
    newhigh = pd.Series(np.where((data['return_a'].cumsum().cummax() - data['return_a'].cumsum() == 0),
                                 data['return_a'].cumsum(), np.nan), index=data.index)
    newhigh = newhigh.replace(0, np.nan).dropna()
    plt.scatter(newhigh.index, newhigh, color='lime', zorder=6, s=10)
    plt.ylabel('Cumulative Return (100%)')
    plt.legend()
    plt.show()


# 最大風險回撤函數
def DrawDown(data: pd.DataFrame):
    return -(data.dropna().cumsum().cummax() - data.dropna().cumsum())


# 蒙地卡羅隨機模擬
def Monte_Carlo_Simulation(data: pd.DataFrame, n):
    plt.close()
    for i in range(1, n, 1):
        random_position = pd.DataFrame(data['position_a'].sample(frac=1)).set_index(data.index)
        df = pd.concat([data['d_ret'], random_position], axis=1)
        df['new_ret'] = df['d_ret'] * df['position_a']
        plt.plot(df['new_ret'].cumsum())
    plt.plot(data['return_a'].cumsum(), color='black', linewidth=3.0)
    plt.show()


# Bokeh 動態股價圖
def chart(data, id_name, method):
    df = data.copy().reset_index(drop=True).reset_index()
    df["Date"] = [i.date() for i in data.index]
    df["Date"] = df["Date"].astype(str)

    ta = TA(df)
    bands = ta.BBands_original(20, 2, 2)
    KD = ta.kd(9)
    RSI = ta.rsi(12)
    lo = 10
    sh = 5
    if method == 'modified':
        bands = ta.BBands_modified(20, std_up, std_down)
        df["MA" + str(sh)] = df.rolling(5).mean()["Close"].tolist()
        df["MA" + str(lo)] = df.rolling(10).mean()["Close"].tolist()

    elif method == 'original':
        bands = ta.BBands_original(20, std_up, std_down)
        df["MA5"] = df.rolling(5).mean()["Close"].tolist()
        df["MA10"] = df.rolling(10).mean()["Close"].tolist()

    elif method == 'ma':
        lo = max(long_inv, short_inv)
        sh = min(long_inv, short_inv)
        df["MA" + str(sh)] = df.rolling(sh).mean()["Close"].tolist()
        df["MA" + str(lo)] = df.rolling(lo).mean()["Close"].tolist()

    elif method == 'kd':
        KD = ta.kd(day_inv)
        df["MA" + str(sh)] = df.rolling(5).mean()["Close"].tolist()
        df["MA" + str(lo)] = df.rolling(10).mean()["Close"].tolist()

    elif method == 'rsi':
        df["MA" + str(sh)] = df.rolling(5).mean()["Close"].tolist()
        df["MA" + str(lo)] = df.rolling(10).mean()["Close"].tolist()

    df["Mean"] = bands['Mid']
    df["Upper"] = bands['Up']
    df["Lower"] = bands['Lo']

    df["K"] = KD['K']
    df["D"] = KD['D']
    df["RSI"] = RSI['value']

    inc = df.Close > df.Open
    dec = df.Open > df.Close
    inc_data = df[inc]
    dec_data = df[dec]

    df_source = ColumnDataSource(df)
    inc_source = ColumnDataSource(inc_data)
    dec_source = ColumnDataSource(dec_data)

    hover = HoverTool(
        tooltips=[
            ("Date", "@Date"),
            ("Close", "@Close"),
            ("Open", "@Open"),
            ("High", "@High"),
            ("Low", "@Low"),
            ("Volume", "@Volume"),
        ],
        formatters={"@Date": "datetime"}
    )

    hover_DMI = HoverTool(
        tooltips=[
            ("Date", "@Date"),
            ("K", "@K"),
            ("D", "@D"),
            ("RSI", "@RSI")
        ],
        formatters={"@Date": "datetime"}
    )

    title = id_name + " Historical Chart"
    x_end = len(df)
    init = 120
    x_start = x_end - init
    interval_freq = init / 10

    y_start = df["Close"].min() * 0.95
    y_end = df["Close"].max() * 1.05

    y_start_2 = 0
    y_end_2 = max(df["K"].max(), df["D"].max(), df["RSI"].max()) * 1.05

    plot1 = figure(plot_width=1800, title=title, plot_height=650, x_range=(x_start, x_end), y_range=(y_start, y_end),
                   tools=[hover, "pan,zoom_in,zoom_out,crosshair,reset,save"], toolbar_location="above",
                   y_axis_label="price")

    plot2 = figure(plot_width=1800, title="DMI Technical Chart", plot_height=325, x_range=plot1.x_range,
                   y_range=(y_start_2, y_end_2),
                   background_fill_color="#fafafa", tools=[hover_DMI, "pan,zoom_in,zoom_out,crosshair,reset,save"],
                   toolbar_location="above", y_axis_label="value")

    for fig in [plot1, plot2]:
        fig.title.text_font_size = "16pt"
        fig.xaxis.major_label_overrides = {i: date.strftime("%Y/%m/%d") for i, date in
                                           enumerate(pd.to_datetime(df["Date"]))}
        fig.xaxis.ticker = SingleIntervalTicker(interval=interval_freq)

    plot1.segment("index", 'High', "index", "Low", color="black",
                  source=df_source)
    plot1.vbar(x="index", width=0.5, bottom="Open", top="Close", fill_color="#cd5c5c", line_color="black",
               source=inc_source)
    plot1.vbar(x="index", width=0.5, top="Open", bottom="Close", fill_color="#2e8b57", line_color="black",
               source=dec_source)

    ma_legend_items = []
    for ma_name, color in zip(["MA" + str(sh), "MA" + str(lo), "Mean", "Upper", "Lower"],
                              ["deepskyblue", "navajowhite", "slateblue", "pink", "lightgreen"]):
        ma_df = df[["index", ma_name]]
        source = ColumnDataSource(ma_df)
        ma_line = plot1.line(x="index", y=ma_name, line_width=2, color=color, alpha=0.8,
                             muted_color=color, muted_alpha=0.2, source=source)
        ma_legend_items.append((ma_name, [ma_line]))

    legend = Legend(items=ma_legend_items, location=(0, 250))
    plot1.add_layout(legend, "left")

    y2_start = df["Volume"].min() * 0.95
    y2_end = df["Volume"].max() * 1.05
    plot1.extra_y_ranges = {"vol": Range1d(y2_start, y2_end)}
    plot1.vbar(x="index", width=0.5, top="Volume", bottom=0, y_range_name="vol", color="blue", alpha=0.2,
               source=df_source)
    plot1.add_layout(LinearAxis(y_range_name="vol", axis_label="vol"), "right")

    DMI_legend_items = []
    for index_name, color in zip(["K", "D", "RSI"], ["indianred", "yellowgreen", "mediumpurple"]):
        index_line = plot2.line("index", index_name, line_width=3, color=color, alpha=0.8, muted_color=color,
                                muted_alpha=0.2, source=df_source)
        DMI_legend_items.append((index_name, [index_line]))

    legend = Legend(items=DMI_legend_items, location=(0, 50))
    plot2.add_layout(legend, "left")

    for fig in [plot1, plot2]:
        fig.legend.label_text_font_size = "8pt"
        fig.legend.click_policy = "mute"

    show(column(plot1, plot2))


# 各種比率計算
class Ratio:
    def __init__(self, series: pd.DataFrame):
        self.data = series

    def Cumulative_Return(self):
        return self.data.dropna().cumsum()[-1]

    def Annual_Return(self):
        return ((1 + self.data.mean()) ** 252) - 1

    def Annual_Volatility(self):
        return self.data.std() * (252 ** 0.5)

    def MDD(self):
        return max(-DrawDown(self.data))

    def MDD_Return(self):
        MS = -DrawDown(self.data)
        MDD = max(MS)
        return self.data.dropna().cumsum()[-1] / MDD

    def Sharpe(self):
        rf = (0.002358 + 1) ** (1 / 252) - 1
        return ((self.data.mean() - rf) / self.data.std()) * (252 ** 0.5)

    def Sortino(self):
        rf = (0.002358 + 1) ** (1 / 252) - 1
        return ((self.data.mean() - rf) / self.data[self.data < 0].std()) * (252 ** 0.5)


# 敏感性分析，依據不同策略的敏感性分析為不同的class，每個策略敏感性分析下都有不同評斷標準的函數
class BBandOriginMethod1SA:
    def __init__(self, stock):
        self.stock = stock
        self.upwide = []
        self.dnwide = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.total_return.append(temp_ratio.Cumulative_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.total_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_return.append(temp_ratio.Annual_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_volatility.append(temp_ratio.Annual_Volatility())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_volatility, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.MDD.append(temp_ratio.MDD())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.MDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.RonMDD.append(temp_ratio.MDD_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.RonMDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('RonMDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sharpe.append(temp_ratio.Sharpe())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sharpe, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sortino.append(temp_ratio.Sortino())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sortino, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sortino')
        plt.show()


class BBandOriginMethod2SA:
    def __init__(self, stock):
        self.stock = stock
        self.upwide = []
        self.dnwide = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.total_return.append(temp_ratio.Cumulative_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.total_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_return.append(temp_ratio.Annual_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_volatility.append(temp_ratio.Annual_Volatility())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_volatility, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.MDD.append(temp_ratio.MDD())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.MDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.RonMDD.append(temp_ratio.MDD_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.RonMDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('RonMDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sharpe.append(temp_ratio.Sharpe())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sharpe, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_original_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sortino.append(temp_ratio.Sortino())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sortino, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sortino')
        plt.show()


class BBandModifiedMethod1SA:
    def __init__(self, stock):
        self.stock = stock
        self.upwide = []
        self.dnwide = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.total_return.append(temp_ratio.Cumulative_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.total_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_return.append(temp_ratio.Annual_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_volatility.append(temp_ratio.Annual_Volatility())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_volatility, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.MDD.append(temp_ratio.MDD())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.MDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.RonMDD.append(temp_ratio.MDD_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.RonMDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('RonMDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sharpe.append(temp_ratio.Sharpe())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sharpe, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_1(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sortino.append(temp_ratio.Sortino())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sortino, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sortino')
        plt.show()


class BBandModifiedMethod2SA:
    def __init__(self, stock):
        self.stock = stock
        self.ta = TA(stock)
        self.upwide = []
        self.dnwide = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.total_return.append(temp_ratio.Cumulative_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.total_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_return.append(temp_ratio.Annual_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_volatility.append(temp_ratio.Annual_Volatility())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_volatility, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.MDD.append(temp_ratio.MDD())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.MDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.RonMDD.append(temp_ratio.MDD_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.RonMDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('RonMDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sharpe.append(temp_ratio.Sharpe())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sharpe, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).band_modified_method_2(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sortino.append(temp_ratio.Sortino())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sortino, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sortino')
        plt.show()


class MASA:
    def __init__(self, stock):
        self.stock = stock
        self.ta = TA(stock)
        self.longday = []
        self.shortday = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.total_return.append(temp_ratio.Cumulative_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.total_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short day')
        ax.set_zlabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.annual_return.append(temp_ratio.Annual_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.annual_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short Day')
        ax.set_zlabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.annual_volatility.append(temp_ratio.Annual_Volatility())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.annual_volatility, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short Day')
        ax.set_zlabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.MDD.append(temp_ratio.MDD())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.MDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short Day')
        ax.set_zlabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.RonMDD.append(temp_ratio.MDD_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.RonMDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short Day')
        ax.set_zlabel('RonMDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.sharpe.append(temp_ratio.Sharpe())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.sharpe, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short Day')
        ax.set_zlabel('sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).MA_breakthrough(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.longday.append(i)
                self.shortday.append(j)
                self.sortino.append(temp_ratio.Sortino())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.longday, self.shortday, self.sortino, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('Long Day')
        ax.set_ylabel('Short Day')
        ax.set_zlabel('sortino')
        plt.show()


class KDSA:
    def __init__(self, stock):
        self.stock = stock
        self.ta = TA(stock)
        self.day = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.total_return.append(temp_ratio.Cumulative_Return())

        plt.plot(self.day, self.total_return)
        plt.xlabel('Days')
        plt.ylabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.annual_return.append(temp_ratio.Annual_Return())

        plt.plot(self.day, self.annual_return)
        plt.xlabel('Days')
        plt.ylabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.annual_volatility.append(temp_ratio.Annual_Volatility())

        plt.plot(self.day, self.annual_volatility)
        plt.xlabel('Days')
        plt.ylabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.MDD.append(temp_ratio.MDD())

        plt.plot(self.day, self.MDD)
        plt.xlabel('Days')
        plt.ylabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.RonMDD.append(temp_ratio.MDD_Return())

        plt.plot(self.day, self.RonMDD)
        plt.xlabel('Days')
        plt.ylabel('Return on MDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.sharpe.append(temp_ratio.Sharpe())

        plt.plot(self.day, self.sharpe)
        plt.xlabel('Days')
        plt.ylabel('Sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            temp_signal = Signals(self.stock).kd_breakthrough(i)
            temp_return = Returns(self.stock, temp_signal)['return_a']
            temp_ratio = Ratio(temp_return)
            self.day.append(i)
            self.sortino.append(temp_ratio.Sortino())

        plt.plot(self.day, self.sortino)
        plt.xlabel('Days')
        plt.ylabel('Sortino')
        plt.show()


class RSISA:
    def __init__(self, stock):
        self.stock = stock
        self.upwide = []
        self.dnwide = []
        self.total_return = []
        self.annual_return = []
        self.annual_volatility = []
        self.MDD = []
        self.RonMDD = []
        self.sharpe = []
        self.sortino = []

    def total_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.total_return.append(temp_ratio.Cumulative_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.total_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Total Return')
        plt.show()

    def annual_return_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_return.append(temp_ratio.Annual_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_return, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Return')
        plt.show()

    def annual_volatility_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.annual_volatility.append(temp_ratio.Annual_Volatility())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.annual_volatility, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Annual Volatility')
        plt.show()

    def mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.MDD.append(temp_ratio.MDD())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.MDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('MDD')
        plt.show()

    def return_on_mdd_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.RonMDD.append(temp_ratio.MDD_Return())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.RonMDD, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('RonMDD')
        plt.show()

    def sharpe_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sharpe.append(temp_ratio.Sharpe())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sharpe, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sharpe')
        plt.show()

    def sortino_graphing(self, xfrom, xto, xby, yfrom, yto, yby):
        plt.close()

        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_signal = Signals(self.stock).RSI(i, j)
                temp_return = Returns(self.stock, temp_signal)['return_a']
                temp_ratio = Ratio(temp_return)
                self.upwide.append(i)
                self.dnwide.append(j)
                self.sortino.append(temp_ratio.Sortino())

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.sortino, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('sortino')
        plt.show()


# 績效表格
class Performance:
    def __init__(self, pnl):
        self.all = pnl['return_a']
        self.long = pnl['return_l']
        self.short = pnl['return_s']
        self.benchmark = pnl['d_ret']

    @staticmethod
    def display(series: pd.DataFrame):
        temp_dict = {
            'Total Return': str(round(Ratio(series).Cumulative_Return() * 100, 2)) + ' %',
            'Yearly Return': str(round(Ratio(series).Annual_Return() * 100, 2)) + ' %',
            'Yearly Volatility': str(round(Ratio(series).Annual_Volatility() * 100, 2)) + ' %',
            'MDD': str(round(max(-DrawDown(series)) * 100, 2)) + ' %',
            'Return on MDD': str(round(Ratio(series).MDD_Return(), 2)),
            'Sharpe Ratio': str(round(Ratio(series).Sharpe(), 2)),
            'Sortino Ratio': str(round(Ratio(series).Sortino(), 2))
        }
        return pd.DataFrame(list(temp_dict.items()), columns=['indicator', 'value'])

    def all_return(self):
        return self.display(self.all)

    def long_return(self):
        return self.display(self.long)

    def short_return(self):
        return self.display(self.short)

    def benchmark_return(self):
        return self.display(self.benchmark)

    def p_table(self):
        t = pd.concat([self.all_return(), self.long_return()['value'], self.short_return()['value'],
                       self.benchmark_return()['value']], axis=1)
        t.set_index('indicator', inplace=True)
        t.columns = ['all', 'long', 'short', 'benchmark']
        print('=' * 60)
        return t


Region_Dict = {
    '香港': '.HK',
    '紐約': '',
    '台灣上市': '.TW',
    '台灣上櫃': '.TWO',
    '上海': '.SH',
    '深圳': '.SZ',
    '東京': '.T',
    '韓國': '.KS',
    '法蘭克福': '.DE',
    '倫敦': '.L',
    '巴黎': '.PA',
    '期貨': '=F'}

Region_List = ['香港',
               '紐約',
               '台灣上市',
               '台灣上櫃',
               '上海',
               '深圳',
               '東京',
               '韓國',
               '法蘭克福',
               '倫敦',
               '巴黎',
               '期貨']

Strategy_List = [
    '基本布林通道策略一',
    '基本布林通道策略二',
    '調整布林通道策略一',
    '調整布林通道策略二',
    '長短均線突破策略',
    'KD指標突破策略',
    'RSI指標策略']


def main():
    # 敏感性分析自訂參數
    # 布林通道相關敏感性分析自訂參數
    def entry_parameters_bl():
        plt.close()

        def save_parameters():
            global std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv
            std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv = float(std_1.get()), \
            float(std_2.get()), float(std_3.get()), float(std_4.get()), float(std_5.get()), float(std_6.get())
            window.destroy()

        window = tk.Tk()
        window.title('請輸入參數')
        window_s = ttk.Style(window)
        window_s.theme_use('default')

        tk.Label(window, text="上標準差起始", bg="white").grid(padx=5, pady=5)
        std_1 = ttk.Entry(window)
        std_1.grid(padx=5, pady=5)
        tk.Label(window, text="上標準差結束", bg="white").grid(padx=5, pady=5)
        std_2 = ttk.Entry(window)
        std_2.grid(padx=5, pady=5)
        tk.Label(window, text="上標準差間隔", bg="white").grid(padx=5, pady=5)
        std_3 = ttk.Entry(window)
        std_3.grid(padx=5, pady=5)
        tk.Label(window, text="下標準差起始", bg="white").grid(padx=5, pady=5)
        std_4 = ttk.Entry(window)
        std_4.grid(padx=5, pady=5)
        tk.Label(window, text="下標準差結束", bg="white").grid(padx=5, pady=5)
        std_5 = ttk.Entry(window)
        std_5.grid(padx=5, pady=5)
        tk.Label(window, text="下標準差間隔", bg="white").grid(padx=5, pady=5)
        std_6 = ttk.Entry(window)
        std_6.grid(padx=5, pady=5)
        ttk.Button(window, text='確定', command=save_parameters).grid(padx=5, pady=5)

    # 長短均線敏感性分析自訂參數
    def entry_parameters_ma():
        plt.close()

        def save_parameters():
            global long_day_bgn, long_day_end, long_day_inv, short_day_bgn, short_day_end, short_day_inv
            long_day_bgn, long_day_end, long_day_inv, short_day_bgn, short_day_end, short_day_inv = int(long_1.get()),\
            int(long_2.get()), int(long_3.get()), int(short_1.get()), int(short_2.get()), int(short_3.get())
            window.destroy()

        window = tk.Tk()
        window.title('請輸入參數')
        window_s = ttk.Style(window)
        window_s.theme_use('default')

        tk.Label(window, text="長天起始", bg="white").grid(padx=5, pady=5)
        long_1 = ttk.Entry(window)
        long_1.grid(padx=5, pady=5)
        tk.Label(window, text="長天結束", bg="white").grid(padx=5, pady=5)
        long_2 = ttk.Entry(window)
        long_2.grid(padx=5, pady=5)
        tk.Label(window, text="長天間隔", bg="white").grid(padx=5, pady=5)
        long_3 = ttk.Entry(window)
        long_3.grid(padx=5, pady=5)
        tk.Label(window, text="短天起始", bg="white").grid(padx=5, pady=5)
        short_1 = ttk.Entry(window)
        short_1.grid(padx=5, pady=5)
        tk.Label(window, text="短天結束", bg="white").grid(padx=5, pady=5)
        short_2 = ttk.Entry(window)
        short_2.grid(padx=5, pady=5)
        tk.Label(window, text="短天間隔", bg="white").grid(padx=5, pady=5)
        short_3 = ttk.Entry(window)
        short_3.grid(padx=5, pady=5)
        ttk.Button(window, text='確定', command=save_parameters).grid(padx=5, pady=5)

    # KD敏感性分析自訂參數
    def entry_parameters_kd():
        plt.close()

        def save_parameters():
            global kd_day_bgn, kd_day_end, kd_day_inv
            kd_day_bgn, kd_day_end, kd_day_inv = int(kd_bgn.get()), int(kd_end.get()), int(kd_inv.get())
            window.destroy()

        window = tk.Tk()
        window.title('請輸入參數')
        window_s = ttk.Style(window)
        window_s.theme_use('default')

        tk.Label(window, text="天數起始", bg="white").grid(padx=5, pady=5)
        kd_bgn = ttk.Entry(window)
        kd_bgn.grid(padx=5, pady=5)
        tk.Label(window, text="天數結束", bg="white").grid(padx=5, pady=5)
        kd_end = ttk.Entry(window)
        kd_end.grid(padx=5, pady=5)
        tk.Label(window, text="天數間隔", bg="white").grid(padx=5, pady=5)
        kd_inv = ttk.Entry(window)
        kd_inv.grid(padx=5, pady=5)
        ttk.Button(window, text='確定', command=save_parameters).grid(padx=5, pady=5)

    # RSI敏感性分析自訂參數
    def entry_parameters_rsi():
        plt.close()

        def save_parameters():
            global rsi_top_bgn, rsi_top_end, rsi_top_inv, rsi_dn_bgn, rsi_dn_end, rsi_dn_inv
            rsi_top_bgn, rsi_top_end, rsi_top_inv, rsi_dn_bgn, rsi_dn_end, rsi_dn_inv = float(rsi_bgn_1.get()), \
            float(rsi_end_1.get()), float(rsi_inv_1.get()), float(rsi_bgn_2.get()), float(rsi_end_2.get()), float(rsi_inv_2.get())
            window.destroy()

        window = tk.Tk()
        window.title('請輸入參數')
        window_s = ttk.Style(window)
        window_s.theme_use('default')

        tk.Label(window, text="上標準差起始", bg="white").grid(padx=5, pady=5)
        rsi_bgn_1 = ttk.Entry(window)
        rsi_bgn_1.grid(padx=5, pady=5)
        tk.Label(window, text="上標準差結束", bg="white").grid(padx=5, pady=5)
        rsi_end_1 = ttk.Entry(window)
        rsi_end_1.grid(padx=5, pady=5)
        tk.Label(window, text="上標準差間隔", bg="white").grid(padx=5, pady=5)
        rsi_inv_1 = ttk.Entry(window)
        rsi_inv_1.grid(padx=5, pady=5)
        tk.Label(window, text="下標準差起始", bg="white").grid(padx=5, pady=5)
        rsi_bgn_2 = ttk.Entry(window)
        rsi_bgn_2.grid(padx=5, pady=5)
        tk.Label(window, text="下標準差結束", bg="white").grid(padx=5, pady=5)
        rsi_end_2 = ttk.Entry(window)
        rsi_end_2.grid(padx=5, pady=5)
        tk.Label(window, text="下標準差間隔", bg="white").grid(padx=5, pady=5)
        rsi_inv_2 = ttk.Entry(window)
        rsi_inv_2.grid(padx=5, pady=5)
        ttk.Button(window, text='確定', command=save_parameters).grid(padx=5, pady=5)

    # 績效展示GUI
    def picture(ret_df):
        def draw_chart():
            if Which_strategy == '調整布林通道策略一' or Which_strategy == '調整布林通道策略二':
                chart(stock, stock_id, 'modified')
            elif Which_strategy == '基本布林通道策略一' or Which_strategy == '基本布林通道策略二':
                chart(stock, stock_id, 'original')
            elif Which_strategy == '長短均線突破策略':
                chart(stock, stock_id, 'ma')
            elif Which_strategy == 'KD指標突破策略':
                chart(stock, stock_id, 'kd')
            elif Which_strategy == 'RSI指標策略':
                chart(stock, stock_id, 'rsi')
            canvs.draw()

        def draw_Yearly_return():
            YearlyAnalysis(ret_df).graphing()
            canvs.draw()

        def show_Yearly_return():
            table1 = YearlyAnalysis(ret_df).YR
            table_show = tk.Tk()
            table_show.title('年週期分析表')
            table_show.geometry('300x230')
            frame = tk.Frame(table_show)
            frame.grid()
            pt = Table(frame, dataframe=table1)
            pt.show()
            pt.showIndex()

        def show_performance():
            table2 = Performance(ret_df).p_table()
            table_show = tk.Tk()
            table_show.title('總績效表')
            table_show.geometry('530x190')
            frame = tk.Frame(table_show)
            frame.grid()
            pt = Table(frame, dataframe=table2)
            pt.show()
            pt.showIndex()

        def draw_PnL():
            PnL(ret_df)
            canvs.draw()

        def draw_PnL2():
            PnL2(ret_df)
            canvs.draw()

        def draw_PnL3():
            PnL3(ret_df)
            canvs.draw()

        def draw_Monte_Carlo():
            Monte_Carlo_Simulation(ret_df, 50)
            canvs.draw()

        def draw_CumRet():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).total_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).total_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).total_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).total_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).total_return_graphing(long_day_bgn, long_day_end, long_day_inv,
                                                  short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).total_return_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).total_return_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv,
                                                   rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        def draw_AnuRet():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).annual_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).annual_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).annual_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).annual_return_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).annual_return_graphing(long_day_bgn, long_day_end, long_day_inv,
                                                   short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).annual_return_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).annual_return_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv,
                                                    rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        def draw_AnuVol():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).annual_volatility_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).annual_volatility_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).annual_volatility_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).annual_volatility_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).annual_volatility_graphing(long_day_bgn, long_day_end, long_day_inv,
                                                       short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).annual_volatility_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).annual_volatility_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv,
                                                        rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        def draw_MDD():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).mdd_graphing(long_day_bgn, long_day_end, long_day_inv,
                                         short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).mdd_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).mdd_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv, rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        def draw_RonMDD():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).return_on_mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).return_on_mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).return_on_mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).return_on_mdd_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).return_on_mdd_graphing(long_day_bgn, long_day_end, long_day_inv,
                                                   short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).return_on_mdd_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).return_on_mdd_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv,
                                                    rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        def draw_Sharpe_Ratio():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).sharpe_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).sharpe_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).sharpe_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).sharpe_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).sharpe_graphing(long_day_bgn, long_day_end, long_day_inv,
                                            short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).sharpe_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).sharpe_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv,
                                             rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        def draw_Sortino_Ratio():
            if Which_strategy == '基本布林通道策略一':
                BBandOriginMethod1SA(stock).sortino_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '基本布林通道策略二':
                BBandOriginMethod2SA(stock).sortino_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略一':
                BBandModifiedMethod1SA(stock).sortino_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '調整布林通道策略二':
                BBandModifiedMethod2SA(stock).sortino_graphing(std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv)
            elif Which_strategy == '長短均線突破策略':
                MASA(stock).sortino_graphing(long_day_bgn, long_day_end, long_day_inv,
                                             short_day_bgn, short_day_end, short_day_inv)
            elif Which_strategy == 'KD指標突破策略':
                KDSA(stock).sortino_graphing(kd_day_bgn, kd_day_end, kd_day_inv)
            elif Which_strategy == 'RSI指標策略':
                RSISA(stock).sortino_graphing(rsi_top_bgn, rsi_top_end, rsi_top_inv,
                                              rsi_dn_bgn, rsi_dn_end, rsi_dn_inv)
            canvs.draw()

        pic = tk.Tk()
        pic.title('回測結果')
        pic_s = ttk.Style(pic)
        pic_s.theme_use('default')
        f = Figure(figsize=(0, 0))
        canvs = FigureCanvasTkAgg(f, master=pic)
        canvs.get_tk_widget().grid()
        ttk.Button(pic, text='動態股價圖', command=draw_chart).grid(padx=10, pady=5, ipadx=31)
        tk.Label(pic, text='========績效展示========', bg='white').grid(padx=10, pady=5)
        ttk.Button(pic, text='總績效表', command=show_performance).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='年週期分析圖', command=draw_Yearly_return).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='年週期分析表', command=show_Yearly_return).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='策略與B&H績效分析', command=draw_PnL).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='策略與B&H差異', command=draw_PnL2).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='多單空單績效分析', command=draw_PnL3).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='蒙地卡羅隨機模擬', command=draw_Monte_Carlo).grid(padx=10, pady=5)
        tk.Label(pic, text='========敏感性分析========', bg='white').grid(padx=10, pady=5)
        ttk.Button(pic, text='累積總報酬', command=draw_CumRet).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='年化報酬率', command=draw_AnuRet).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='年化波動率', command=draw_AnuVol).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='最大風險回撤', command=draw_MDD).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='風險報酬比', command=draw_RonMDD).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='夏普比率', command=draw_Sharpe_Ratio).grid(padx=10, pady=5, ipadx=31)
        ttk.Button(pic, text='索丁諾比率', command=draw_Sortino_Ratio).grid(padx=10, pady=5, ipadx=31)
        if com2.get() == '基本布林通道策略一' or com2.get() == '基本布林通道策略二' or com2.get() == '調整布林通道策略一' or \
                com2.get() == '調整布林通道策略二':
            ttk.Button(pic, text='自訂參數', command=entry_parameters_bl).grid(padx=10, pady=20)
        elif com2.get() == '長短均線突破策略':
            ttk.Button(pic, text='自訂參數', command=entry_parameters_ma).grid(padx=10, pady=20)
        elif com2.get() == 'KD指標突破策略':
            ttk.Button(pic, text='自訂參數', command=entry_parameters_kd).grid(padx=10, pady=20)
        elif com2.get() == 'RSI指標策略':
            ttk.Button(pic, text='自訂參數', command=entry_parameters_rsi).grid(padx=10, pady=20)

    # 主程式警告訊息
    def warning():
        def close_war():
            war.destroy()

        war = tk.Toplevel(root)
        tk.Message(war, text="該個股代號不存在，請重新輸入！", width=1000).grid()
        ttk.Button(war, text="返回", command=close_war).grid(padx=10, pady=10)

    def warning2():
        def close_war():
            war.destroy()

        war = tk.Toplevel(root)
        tk.Message(war, text="尚未選擇日期！", width=1000).grid()
        ttk.Button(war, text="返回", command=close_war).grid(padx=10, pady=10)

    def warning3():
        def close_war():
            war.destroy()

        war = tk.Toplevel(root)
        tk.Message(war, text=f"該個股首日為{first_day}", width=1000).grid()
        ttk.Button(war, text="返回", command=close_war).grid(padx=10, pady=10)

    # 主程式
    Which_region = com1.get()
    if Start_date == "" or End_date == "":
        warning2()
        return
    else:
        stock_id = s_id.get()
        stock_id += Region_Dict.get(Which_region)
        stock = YahooData(stock_id, Start_date, End_date)
        if len(stock) == 0:
            print('該個股代號不存在，請重新輸入')
            warning()
            return

        first_day = ""
        for i in range(10):
            first_day += str(stock.index[0])[i]
        if first_day != Start_date:
            warning3()

        Which_strategy = com2.get()
        sig = Signals(stock)
        if Which_strategy == '基本布林通道策略一':
            sig = sig.band_original_method_1(std_up, std_down)
            return_data = Returns(stock, sig)
            picture(return_data)

        elif Which_strategy == '基本布林通道策略二':
            sig = sig.band_original_method_2(std_up, std_down)
            return_data = Returns(stock, sig)
            picture(return_data)

        elif Which_strategy == '調整布林通道策略一':
            sig = sig.band_modified_method_1(std_up, std_down)
            return_data = Returns(stock, sig)
            picture(return_data)

        elif Which_strategy == '調整布林通道策略二':
            sig = sig.band_modified_method_2(std_up, std_down)
            return_data = Returns(stock, sig)
            picture(return_data)

        elif Which_strategy == '長短均線突破策略':
            sig = sig.MA_breakthrough(long_inv, short_inv)
            return_data = Returns(stock, sig)
            picture(return_data)

        elif Which_strategy == 'KD指標突破策略':
            sig = sig.kd_breakthrough(day_inv)
            return_data = Returns(stock, sig)
            picture(return_data)

        elif Which_strategy == 'RSI指標策略':
            sig = sig.RSI(std_up, std_down)
            return_data = Returns(stock, sig)
            picture(return_data)


# 第一層GUI
def calendar_view_begin():
    def return_begin():
        begin_date.configure(text=str(cal.selection_get()))
        global Start_date
        Start_date = str(cal.selection_get())
        top.destroy()

    top = tk.Toplevel(root)
    cal = Calendar(top, font="14", selectmode='day', year=2010, month=12, day=1, showweeknumbers=False,
                   showothermonthdays=False, weekendbackground="white")
    cal.grid()
    ttk.Button(top, text="確認", command=return_begin).grid()


def calendar_view_end():
    def return_end():
        end_date.configure(text=str(cal.selection_get()))
        global End_date
        End_date = str(cal.selection_get())
        top.destroy()

    top = tk.Toplevel(root)
    cal = Calendar(top, font="14", selectmode='day', year=2020, month=12, day=31, showweeknumbers=False,
                   showothermonthdays=False, weekendbackground="white")
    cal.grid()
    ttk.Button(top, text="確認", command=return_end).grid()


# 輸入想使用策略的參數
def entry_strategy_para():
    def entry_strategy_para1():
        def save_strategy_para():
            global std_up, std_down
            std_up = float(para1.get())
            std_down = float(para2.get())
            entry_strategy.destroy()
            main()

        entry_strategy = tk.Tk()
        entry_strategy.title('請輸入上下標準差')
        window_s = ttk.Style(entry_strategy)
        window_s.theme_use('default')

        tk.Label(entry_strategy, text="上標準差", bg="white").grid(column=0, row=0)
        para1 = ttk.Entry(entry_strategy)
        para1.grid(padx=5, pady=5, column=1, row=0)
        tk.Label(entry_strategy, text="下標準差", bg="white").grid(column=0, row=1)
        para2 = ttk.Entry(entry_strategy)
        para2.grid(padx=5, pady=5, column=1, row=1)
        ttk.Button(entry_strategy, text='確定', command=save_strategy_para).grid(padx=5, pady=5, column=1)

    def entry_strategy_para2():
        def save_strategy_para():
            global long_inv, short_inv
            long_inv = int(para1.get())
            short_inv = int(para2.get())
            entry_strategy.destroy()
            main()

        entry_strategy = tk.Tk()
        entry_strategy.title('請輸入長短天數')
        window_s = ttk.Style(entry_strategy)
        window_s.theme_use('default')

        tk.Label(entry_strategy, text="長天數", bg="white").grid(column=0, row=0)
        para1 = ttk.Entry(entry_strategy)
        para1.grid(padx=5, pady=5, column=1, row=0)
        tk.Label(entry_strategy, text="短天數", bg="white").grid(column=0, row=1)
        para2 = ttk.Entry(entry_strategy)
        para2.grid(padx=5, pady=5, column=1, row=1)
        ttk.Button(entry_strategy, text='確定', command=save_strategy_para).grid(padx=5, pady=5, column=1)

    def entry_strategy_para3():
        def save_strategy_para():
            global day_inv
            day_inv = int(para1.get())
            entry_strategy.destroy()
            main()

        entry_strategy = tk.Tk()
        entry_strategy.title('請輸入天數')
        window_s = ttk.Style(entry_strategy)
        window_s.theme_use('default')

        tk.Label(entry_strategy, text="天數", bg="white").grid(column=0, row=0)
        para1 = ttk.Entry(entry_strategy)
        para1.grid(padx=5, pady=5, column=1, row=0)
        ttk.Button(entry_strategy, text='確定', command=save_strategy_para).grid(padx=5, pady=5, column=1)

    def entry_strategy_para4():
        def save_strategy_para():
            global std_up, std_down
            std_up = int(para1.get())
            std_down = int(para2.get())
            entry_strategy.destroy()
            main()

        entry_strategy = tk.Tk()
        entry_strategy.title('請輸入上下閾值（0-100）')
        window_s = ttk.Style(entry_strategy)
        window_s.theme_use('default')

        tk.Label(entry_strategy, text="上閾值", bg="white").grid(column=0, row=0)
        para1 = ttk.Entry(entry_strategy)
        para1.grid(padx=5, pady=5, column=1, row=0)
        tk.Label(entry_strategy, text="下閾值", bg="white").grid(column=0, row=1)
        para2 = ttk.Entry(entry_strategy)
        para2.grid(padx=5, pady=5, column=1, row=1)
        ttk.Button(entry_strategy, text='確定', command=save_strategy_para).grid(padx=5, pady=5, column=1)

    if com2.get() == '基本布林通道策略一' or com2.get() == '基本布林通道策略二' or com2.get() == '調整布林通道策略一' or \
            com2.get() == '調整布林通道策略二':
        entry_strategy_para1()
    elif com2.get() == '長短均線突破策略':
        entry_strategy_para2()
    elif com2.get() == 'KD指標突破策略':
        entry_strategy_para3()
    elif com2.get() == 'RSI指標策略':
        entry_strategy_para4()


# 說明
def support():
    def close_sup():
        sup.destroy()

    sup = tk.Tk()
    sup.title('說明')
    sup_s = ttk.Style(root)
    sup_s.theme_use('default')
    sup.geometry("710x500")
    tk.Label(sup, text='='*38 + '策略說明' + '='*38 + '\n'
             + '基本布林通道策略一: 碰到上緣時做空，碰到下緣時做多' + '\n'
             + '基本布林通道策略一: 碰到上緣時做空，碰到下緣時做多，並輔以成交量為濾網' + '\n'
             + '調整布林通道策略一: 經由數學公式推導，減少通道滯後，碰到上緣時做空，碰到下緣時做多' +'\n'
             + '調整布林通道策略二: 經由數學公式推導，減少通道滯後，碰到上緣時做空，碰到下緣時做多，並輔以成交量為濾網' + '\n'
             + '長短均線突破策略: 短天期均線向上穿越長天期均線時做多，長天期均線向上穿越短天期均線時做空' + '\n'
             + 'KD指標突破策略: K線向下穿越D線時做多，K線向上穿越D線時做空' + '\n'
             + 'RSI指標策略: RSI向上穿越設定上閾值做多，RSI向下穿越設定下閾值做空' + '\n'*20
             + '敬請期待未來本團隊擴充策略包！若有好的策略或指標也歡迎提供!' + '\n'
             + '投資一定有風險，本系統只顯示過去歷史回測結果，並不對使用者損益負責',
             wraplength=700, justify="left").grid()
    tk.Button(sup, text="返回", command=close_sup, bg="white").grid(padx=10, pady=10)


Start_date = ""
End_date = ""
# 主程式選擇策略的參數
std_up, std_down, long_inv, short_inv, day_inv = 0, 0, 0, 0, 0

# 敏感性分析的參數-布林用 default
std_top_bgn, std_top_end, std_top_inv, std_dn_bgn, std_dn_end, std_dn_inv = 0.5, 2.5, 0.25, 0.5, 2.5, 0.25

# 敏感性分析的參數-長短均線用 default
long_day_bgn, long_day_end, long_day_inv, short_day_bgn, short_day_end, short_day_inv = 30, 90, 4, 5, 20, 1

# 敏感性分析的參數-KD用 default
kd_day_bgn, kd_day_end, kd_day_inv = 5, 20, 1

# 敏感性分析的參數-RSI用 default
rsi_top_bgn, rsi_top_end, rsi_top_inv, rsi_dn_bgn, rsi_dn_end, rsi_dn_inv = 70, 80, 1, 20, 30, 1

root = tk.Tk()
root.geometry("700x600")
root.title('個股回測系統')
s = ttk.Style(root)
s.theme_use('default')

photo = tk.PhotoImage(file='NOXH.png')
theLabel = tk.Label(root, image=photo)
theLabel.place(x=0, y=0, relwidth=1, relheight=1)

tk.Label(root, text="請選擇交易所", bg="#0f386e", fg='white').grid(padx=300, pady=10)
com1 = ttk.Combobox(root, value=Region_List)
com1.grid(padx=10, pady=5)
tk.Label(root, text="回測起始日", bg="#0f386e", fg='white').grid(padx=10, pady=10)
begin_date = ttk.Button(root, text='請點擊', command=calendar_view_begin)
begin_date.grid(padx=10, pady=5)
tk.Label(root, text="回測結束日", bg="#0f386e", fg='white').grid(padx=10, pady=5)
end_date = ttk.Button(root, text='請點擊', command=calendar_view_end)
end_date.grid(padx=10, pady=10)
tk.Label(root, text="請輸入該交易所股票代碼", bg="#0f386e", fg='white').grid(padx=10, pady=10)
s_id = ttk.Entry(root)
s_id.grid(padx=10, pady=5)
tk.Label(root, text="請選擇交易策略", bg="#0f386e", fg='white').grid(padx=10, pady=10)
com2 = ttk.Combobox(root, value=Strategy_List)
com2.grid(padx=10, pady=5)
ttk.Button(root, text='確認', command=entry_strategy_para).grid(padx=10, pady=10)
sup_bt = tk.Button(root, text='說明', command=support).grid(pady=135, sticky='e')
root.mainloop()