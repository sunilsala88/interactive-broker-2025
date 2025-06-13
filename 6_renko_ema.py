# Strategy Overview
# Renko Box Size:
# Calculated using the ATR (14) of the previous day’s 5-minute candles.
# Rounded to the nearest 10 paisa for each stock.
# Indicators Used:
# Renko Charts for price filtering.
# EMA (Exponential Moving Average) applied to Renko closing prices instead of candlestick data.
# Trading Hours:
# The algorithm starts scanning for trades at 9:20 AM to allow for initial market stabilization.
# Uses the first 5-minute candle (9:15 - 9:20 AM) for calculations.
# Profit/Loss Tracking:
# A separate P&L tracker is maintained for each stock to monitor individual performance.
# Trading for a specific stock stops once the profit or loss reaches $10 for the day.
# 2. Entry Rules
# Buy: When the Renko closing price is greater than the EMA.
# Sell: When the Renko closing price is less than the EMA.
# Note: The EMA calculation must be based on Renko box closing prices, not candlestick data.
# 3. Exit Rules
# Stop Loss (SL):
# 2 Renko bricks (e.g., if the box size is $1.5, SL = $3).
# Take Profit (TP):
# Square off 50% of the position at 2 Renko bricks profit.
# Trail the remaining position until a Renko box reversal (opposite-direction brick forms).
# Trade Limits:
# Maximum 2 trades per stock per day.
# Stop trading a stock once its profit or loss reaches ₹1,000 for the day (tracked only after closing the position).


import pandas as pd
import numpy as np
import datetime
import pendulum as dt
from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook
import time
import logging

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf


strategy_name='renko_ema'

logging.basicConfig(level=logging.INFO, filename=f"{strategy_name}.log",filemode='a',format="%(asctime)s - %(message)s")
logging.getLogger('ib_async').setLevel(logging.CRITICAL)

def calculate_ema(prices, period):
    """
    Calculate the Exponential Moving Average (EMA) of a list of prices.
    
    :param prices: List of floats or ints (e.g., daily closing prices)
    :param period: Integer - the period for the EMA (e.g., 10, 20, 50)
    :return: List of EMA values (same length as prices, with None for first period-1 values)
    """
    if len(prices) < period:
        return [None] * len(prices)

    ema_values = [None] * (period - 1)  # EMA not defined for the first 'period - 1' values
    sma = sum(prices[:period]) / period
    ema_values.append(sma)

    multiplier = 2 / (period + 1)
    
    for price in prices[period:]:
        ema_prev = ema_values[-1]
        ema_current = (price - ema_prev) * multiplier + ema_prev
        ema_values.append(ema_current)

    return ema_values

print('strategy started')
logging.info('strategy started')


ib = IB()
ib.connect('127.0.0.1', 7497, clientId=22)


time_zone='America/New_York'
tickers = ['TSLA','NVDA','GOOG','AMZN','AAPL','META','NFLX','MSFT','LNTH','AMD']
exchange='SMART'
currency='USD'
account_no='DUH316001'
ord_validity='GTC'
quantity_=1
#start time
start_hour,start_min=9,30
#end time
end_hour,end_min=15,30

contract_objects={}
for ticker in tickers:
    c=ib.qualifyContracts(Stock(ticker,exchange, currency))[0]
    # print(c)
    contract_objects.update({ticker:c})
print(contract_objects)

def get_historical_data(contract,duration='3 D',candle='5 mins'):

    bars = ib.reqHistoricalData(
        contract, endDateTime=dt.now(time_zone), durationStr=duration,
        barSizeSetting=candle, whatToShow='TRADES', useRTH=True)
    # convert to pandas dataframe (pandas needs to be installed):
    df = util.df(bars)

    df.set_index('date', inplace=True)
    return df


def candle_renko_refresh(ticker,brick_size,data):
    print('candle_renko_refresh')
    calculated_values = {}
    matplotlib.use('Agg')  # Use a non-interactive backend
    mpf.plot(data, type='renko', renko_params=dict(brick_size=brick_size),return_calculated_values=calculated_values,returnfig=True,style='yahoo')
    # plt.show()
    plt.close()
    renko_df = pd.DataFrame(calculated_values)
    print(renko_df)

    def count_bricks(sign_list):
        list1=[]
        pos_count=0
        neg_count=0
        for k in range(len(sign_list)):
            i=sign_list[k]
            if i>0:
                if sign_list[k-1]<0:
                    pos_count=1
                    list1.append(pos_count)
                else:
                    pos_count+=1
                    list1.append(pos_count)

            elif i<0:
                if sign_list[k-1]>0:
                    neg_count=-1
                    list1.append(neg_count)
                else:
                    neg_count-=1
                    list1.append(neg_count)
            else:
                list1.append(0)
        return list1


    renko_df.drop(columns=['renko_volumes','minx','maxx','miny','maxy'],inplace=True,axis=1)

    renko_df['pos_count']=count_bricks(renko_df['renko_bricks'].diff().tolist())



    high=[0]
    low=[0]
    open=[0]
    close=[0]
    #go through renko_bricks and pos_count columns
    for renko,pos in zip(renko_df['renko_bricks'],renko_df['pos_count']):


        if pos<0:
            open.append(renko)
            high.append(renko)
            close.append(renko-brick_size)
            low.append(renko-brick_size)

        elif pos>0:
            close.append(renko)
            high.append(renko)
            open.append(renko-brick_size)
            low.append(renko-brick_size)

    renko_df['OPEN'] = open
    renko_df['HIGH'] = high
    renko_df['LOW'] = low
    renko_df['CLOSE'] = close

    # calculate donchain channel
    renko_df.set_index('renko_dates',inplace=True)
    
   
    renko_df['ema'] =  calculate_ema(renko_df['CLOSE'], 9)
    print(renko_df.tail(30))
    # #filter data for current date
    # renko_df=renko_df[renko_df.index.date==dt.now(time_zone).date()]
    return renko_df


contract=contract_objects['NVDA']
# data=get_historical_data(contract,duration='1 M',candle='1 min')
data=get_historical_data(contract,duration='1 Y',candle='1 day')
print(data)

# data1=data.reset_index()
# import pandas as pd
# data1['date']=pd.to_datetime(data1['date'])
# data1['date']=data1['date'] + pd.to_timedelta('15:30:00')
# data1=data1.set_index('date')
# data1

# candle_renko_refresh('TSLA',2,data1)


def calculate_brick_size(hist_data):
    brick_size=1

    return brick_size

def main_strategy_code():

    print("inside main strategy")
    pos=ib.positions(account=account_no)
    print(pos)
    if len(pos)==0:
        pos_df=pd.DataFrame([])
    else:
        pos_df=util.df(pos)
        pos_df['name']=[cont.symbol for cont in pos_df['contract']]
        pos_df=pos_df[pos_df['position']!=0]
    print(pos_df)
    ord=ib.reqAllOpenOrders()
    if len(ord)==0:
        ord_df=pd.DataFrame([])
    else:
        ord_df=util.df(ord)
        ord_df['name']=[cont.symbol for cont in ord_df['contract']]
    print(ord_df)
    logging.info('Fetched order_df and position_df')

    for ticker in tickers:
        logging.info(ticker)
        print('ticker name is',ticker,'################')
        ticker_contract=contract_objects.get(ticker)
     


        hist_df=get_historical_data(ticker_contract,duration='1 M',candle='5 min')

        print(hist_df)

        renko_df=candle_renko_refresh(ticker,2,hist_df)
        print(renko_df)

        closing_price=hist_df['close'].iloc[-1]
        ema=renko_df['ema'].iloc[-1]
        print('closing price',closing_price,'current_ema',ema)

        print(hist_df_hourly.close.iloc[-1])
      
        capital=int(float([v for v in ib.accountValues(account=account_no) if v.tag == 'AvailableFunds' ][0].value))
        print(capital)
        quantity=int((capital)//hist_df_hourly.close.iloc[-1])  
        print(quantity)
        logging.info('Checking condition')
current_time = dt.now(time_zone)
print(current_time)

start_time=dt.datetime(current_time.year,current_time.month,current_time.day,start_hour,start_min,tz=time_zone)
end_time=dt.datetime(current_time.year,current_time.month,current_time.day,end_hour,end_min,tz=time_zone)
print(start_time)
print(end_time)

logging.info('Checking if start time has been reached')
while start_time>dt.now(time_zone):
    print(dt.now(time_zone))
    time.sleep(1)

logging.info('Starting the main code')
print('startig strategy')

candle_size=60

# ib.newOrderEvent += order_open_handler
# ib.orderStatusEvent += order_open_handler
# ib.cancelOrderEvent += order_open_handler

logging.info('Starting the main code with candle size'+str(candle_size))
main_strategy_code()
while dt.now(time_zone)<end_time:
    
    now = dt.now(time_zone)
    print(now)



    #running strategy every 5 min
    if now.second==1 and now.minute in range(0,60,5):
        main_strategy_code()



    time.sleep(1)



