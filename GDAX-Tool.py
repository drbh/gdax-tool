# from gevent import monkey
# monkey.patch_all(socket=False,thread=False)

import GDAX

import pandas as pd


import time
from threading import Thread
from flask import Flask, render_template, session, request
from flask.ext.socketio import SocketIO, emit, join_room, disconnect



import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from cStringIO import StringIO
import base64
from scipy import sparse, stats

import datetime

app = Flask(__name__)
# app.debug = True
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
thread = None


class myWebsocketClient(GDAX.WebsocketClient):
    def onOpen(self):
        self.url = "wss://ws-feed.gdax.com/"
        self.products = ["ETH-USD"]
        self.MessageCount = 0
        self.message = "starter"
        print("Lets count the messages!")
    def onMessage(self, msg):
        self.MessageCount += 1
        if 'price' in msg and 'type' in msg:
            self.message = msg
    def onClose(self):
        print("-- Goodbye! --")

def getShortHistorical():
    publicClient = GDAX.PublicClient()
    time_now = datetime.datetime.now() + datetime.timedelta(hours=4) # ACCOUNT FOR NY OFF BY 4 hours
    rates = publicClient.getProductHistoricRates(product="LTC-USD",granularity='30',start=time_now - datetime.timedelta(hours=2) ,end=time_now )
    df = pd.DataFrame(data=rates,columns=["time", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df.time,unit='s') - datetime.timedelta(hours=4)
    del df['time']
    return df


def getLongHistorical():
    publicClient = GDAX.PublicClient()
    time_now = datetime.datetime.now() + datetime.timedelta(hours=4) # ACCOUNT FOR NY OFF BY 4 hours
    long_rates = publicClient.getProductHistoricRates(product="LTC-USD", granularity='3600', start=time_now - datetime.timedelta(days=5), end=time_now)
    long_df = pd.DataFrame(data=long_rates, columns=["time", "low", "high", "open", "close", "volume"])
    long_df.index = pd.to_datetime(long_df.time, unit='s') - datetime.timedelta(hours=4)
    del long_df['time']
    return long_df

# wsClient = myWebsocketClient()
# wsClient.start()

def background_stuff():
    """ Let's do it a bit cleaner """
    print "BACKGROUND"



    # # global socketio

    # # print(wsClient.url, wsClient.products)
    # while (wsClient.MessageCount < 30):
    #     print("\nMessageCount =", "%i \n" % wsClient.MessageCount)
    #     # time.sleep(1)
    #     # socketio.emit('my response', {'data': ["TEST"]}, namespace="/test", broadcast=True)
    # #
    # wsClient.close()
    #
    # while True:
    #     time.sleep(1)
    #     t = str(time.clock())
    #     print t

    def minute_passed(oldepoch):
        return time.time() - oldepoch >= .1

    global wsClient

    # t = time.time()
    # for i in range(3000):
    # # while True:
    #     # print time.clock(), t
    #     # if time.clock() > ( t + .1 ):
    #     # show = True #minute_passed(t)
    #     # if show:
    #     # print show, time.time(), t, time.time() - t
    #     t = time.time()
    #     cur_time = str(t)
    #     socketio.emit('message', {'data': cur_time, "msg": wsClient.message['price'] }, namespace="/test", broadcast=True)

    # global thread
    # thread = None


html = '''
<html>
    <body>
        <img src="data:image/png;base64,{}" />
    </body>
</html>
'''

@app.route('/')
def hello_world():
    df = getShortHistorical()
    long = getLongHistorical()

    df = df.iloc[::-1]

    df['change'] = df.close.diff(1)

    df['gain'] = df['change'][df['change'] > 0 ]
    df['loss'] = df['change'][df['change'] < 0].abs()
    timeseries = df.close.tail(60)

    fig, axes = plt.subplots(nrows=6, ncols=1)

    timeseries.plot(ax=axes[0], color='blue', label='Original')

    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()

    grs = df.gain.rolling(window=12,min_periods=1).mean()
    lrs = df.loss.rolling(window=12,min_periods=1).mean()

    smallest = 99999
    biggest = 0
    was_min = []
    was_max = []
    for x in df.close:
        if x < smallest:
            smallest = x
            was_min.append(True)
            was_max.append(False)
        elif x > biggest:
            biggest = x
            was_min.append(False)
            was_max.append(True)
        else:
            was_min.append(False)
            was_max.append(False)

    df['was_min'] = was_min * df.close
    df['was_max'] = was_max * df.close

    rs = grs / lrs
    rsi =  100 - ( 100 /(1 + ( rs ) ) )

    df['rsi'] = rsi
    df['rs'] = rs

    # grs.plot(color='green', label='Rolling Mean')
    # lrs.plot(color='yellow', label='Rolling Mean')
    df.was_min.plot(color="red", label="min", ax=axes[2])
    df.was_max.plot(color="green", label="min", ax=axes[3])

    df.change.rolling(window=6,min_periods=1).sum().plot(color="red", label="min", ax=axes[1])

    df.rsi.plot( color="orange", label="rsi", ax=axes[4] )

    # Hodrick Prescott filter
    def hp_filter(x, lamb=5000):
        w = len(x)
        b = [[1] * w, [-2] * w, [1] * w]
        D = sparse.spdiags(b, [0, 1, 2], w - 2, w)
        I = sparse.eye(w)
        B = (I + lamb * (D.transpose() * D))
        return sparse.linalg.dsolve.spsolve(B, x)

    def mad(data, axis=None):
        return np.mean(np.abs(data - np.mean(data, axis)), axis)

    def AnomalyDetection(x, alpha=0.2, lamb=5000):
        """
        x         : pd.Series
        alpha     : The level of statistical significance with which to
                    accept or reject anomalies. (expon distribution)
        lamb      : penalize parameter for hp filter
        return r  : Data frame containing the index of anomaly
        """
        # calculate residual
        xhat = hp_filter(x, lamb=lamb)
        resid = x - xhat

        # drop NA values
        ds = pd.Series(resid)
        ds = ds.dropna()

        # Remove the seasonal and trend component,
        # and the median of the data to create the univariate remainder
        md = np.median(x)
        data = ds - md

        # process data, using median filter
        ares = (data - data.median()).abs()
        data_sigma = data.mad() + 1e-12
        ares = ares / data_sigma

        # compute significance
        p = 1. - alpha
        R = stats.expon.interval(p, loc=ares.mean(), scale=ares.std())
        threshold = R[1]

        # extract index, np.argwhere(ares > md).ravel()
        r_id = ares.index[ares > threshold]

        return r_id

    # detect anomaly
    r_idx = AnomalyDetection(df.close)

    df[['close']].loc[r_idx].plot(kind="line", ax=axes[5])
    #
    # # plot the result
    # plt.plot(df.close, 'b-')
    # plt.plot(r_idx, df.close[r_idx], 'ro')
    #
    # rolmean.plot(ax=axes[0], color='red', label='Rolling Mean')
    # # Plot rolling statistics:
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)


    io = StringIO()
    fig.savefig(io, format='png')
    data = base64.encodestring(io.getvalue())

    # return html.format(data)
    last_three = long[long.index > long.index[0] - datetime.timedelta(hours=3)].close.mean()
    last_six = long[long.index > long.index[0] - datetime.timedelta(hours=6)].close.mean()
    last_twelve = long[long.index > long.index[0] - datetime.timedelta(hours=12)].close.mean()
    last_day = long[long.index > long.index[0] - datetime.timedelta(days=1)].close.mean()
    last_week = long[long.index > long.index[0] - datetime.timedelta(days=7)].close.mean()


    avgs = dict({
        "3hr": last_three,
        "6hr": last_six,
        "12hr": last_twelve,
        "1dy": last_day,
        "1wk": last_week
    })

    tbl = pd.DataFrame.from_dict(avgs.items())

    tbl[2] = df.close[0]
    tbl[3] = df.close[0] - tbl[1]
    print "test"
    return render_template('index.html',  big_table = df.to_html(), graph = html.format(data), small_table = tbl.to_html() )


@socketio.on('my event', namespace='/test')
def my_event(msg):
    print msg['data']

@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    if thread is None:
        thread = Thread(target=background_stuff)
        thread.start()
    emit('my response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)