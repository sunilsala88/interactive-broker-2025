#super trend and ema strategy
#closing greater than daily ema long
#super positive in hourly

import pandas as pd
import numpy as np
import datetime
import pendulum as dt
from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook
import time
import logging

strategy_name='super_ema'

logging.basicConfig(level=logging.INFO, filename=f"{strategy_name}.log",filemode='a',format="%(asctime)s - %(message)s")
logging.getLogger('ib_async').setLevel(logging.CRITICAL)

def supertrend(high, low, close, length=10, multiplier=3):
    """
    Supertrend function that matches pandas_ta.supertrend output.
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of close prices
        length (int): The ATR period. Default: 7
        multiplier (float): The ATR multiplier. Default: 3.0
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            SUPERT - The trend value
            SUPERTd - The direction (1 for long, -1 for short)
            SUPERTl - The long values
            SUPERTs - The short values
    """
    # Calculate ATR using the pandas_ta method (RMA - Rolling Moving Average)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/length, adjust=False).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Initialize direction and trend
    direction = [1]  # Start with long
    trend = [lowerband.iloc[0]]  # Start with lowerband
    long = [lowerband.iloc[0]]
    short = [np.nan]

    # Iterate through the data to calculate the Supertrend
    for i in range(1, len(close)):
        if close.iloc[i] > upperband.iloc[i - 1]:
            direction.append(1)
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            direction.append(-1)
        else:
            direction.append(direction[i - 1])
            if direction[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if direction[i] == -1 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        if direction[i] == 1:
            trend.append(lowerband.iloc[i])
            long.append(lowerband.iloc[i])
            short.append(np.nan)
        else:
            trend.append(upperband.iloc[i])
            long.append(np.nan)
            short.append(upperband.iloc[i])

    # Create DataFrame to return
    df = pd.DataFrame({
        "SUPERT": trend,
        "SUPERTd": direction,
        "SUPERTl": long,
        "SUPERTs": short,
    }, index=close.index)

    return df

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
ib.connect('127.0.0.1', 7497, clientId=2)


time_zone='America/New_York'
tickers = ['TSLA','NVDA','GOOG','AMZN','AAPL','META','NFLX','MSFT','LNTH']
exchange='SMART'
currency='USD'
account_no='DUH316001'
ord_validity='GTC'
quantity_=1
#start time
start_hour,start_min=8,52
#end time
end_hour,end_min=15,0

contract_objects={}
for ticker in tickers:
    c=ib.qualifyContracts(Stock(ticker,exchange, currency))[0]
    # print(c)
    contract_objects.update({ticker:c})
print(contract_objects)

def get_historical_data(contract,duration='10 D',candle='1 hour'):

    bars = ib.reqHistoricalData(
        contract, endDateTime=dt.now(time_zone), durationStr=duration,
        barSizeSetting=candle, whatToShow='TRADES', useRTH=True)
    # convert to pandas dataframe (pandas needs to be installed):
    df = util.df(bars)

    df.set_index('date', inplace=True)
    s=supertrend(df['high'], df['low'], df['close'])
    df['super']=s['SUPERT']
    df['superd']=s['SUPERTd']
    df['ema']=calculate_ema(df['close'], 10)
    return df


try:
    order_filled_dataframe=pd.read_csv(f'{strategy_name}-orders.csv')
    order_filled_dataframe.set_index('time',inplace=True)

except:
    column_names = ['time','ticker','price','action']
    order_filled_dataframe = pd.DataFrame(columns=column_names)
    order_filled_dataframe.set_index('time',inplace=True)


def order_open_handler(order):
    global order_filled_dataframe
    if order.orderStatus.status=='Filled':
        print('order filled')
        logging.info('order filled')
        name=order.contract.localSymbol
        a=[name,order.orderStatus.avgFillPrice,order.order.action]
        # if name not in order_filled_dataframe.ticker.to_list():
        order_filled_dataframe.loc[order.fills[0].execution.time] = a
        order_filled_dataframe.to_csv(f'{strategy_name}-orders.csv')
        message=order.contract.localSymbol+" "+order.order.action+"  "+str(order.orderStatus.avgFillPrice)
        logging.info(message)

def check_market_order_placed(name):
    ord=ib.reqAllOpenOrders()

    if ord:
        ord_df=pd.DataFrame(ord)
        print(ord_df)
        print(type(ord_df))
        # ord_df.to_csv('order_list.csv')
        ord_df['name']=[c['localSymbol'] for c in list(ord_df['contract'])]
        ord_df['ord_type']=[c['orderType']for c in list(ord_df['order'])]
        a=ord_df[(ord_df['name']==name) & (ord_df['ord_type']=='MKT') ]
        print(a)
        if a.empty:
            return True

        else:
            return False
    else:
        return True


def trade_buy_stocks(stock_name):


    #market order
    contract = contract_objects[stock_name]
    # ord=MarketOrder(action='BUY',totalQuantity=1)
    if check_market_order_placed(stock_name):
        ord=Order(orderId=ib.client.getReqId(),orderType='MKT',totalQuantity=1,action='BUY',account=account_no,tif=ord_validity)
        trade=ib.placeOrder(contract,ord)
        ib.sleep(1)
        logging.info(trade)
        logging.info('Placed market buy order')
  
    else:
        logging.info('market order already placed')
        print('market order already placed')
        return 0        


def trade_sell_stocks(stock_name): #closing_price, quantitys=1  ????


    #market order
    global current_balance
    #market order
    contract = contract_objects[stock_name]
    # ord=MarketOrder(action='SELL',totalQuantity=1,AccountValue=account_no)
    if check_market_order_placed(stock_name):
       
        ord=Order(orderId=ib.client.getReqId(),orderType='MKT',totalQuantity=1,action='SELL',account=account_no,tif=ord_validity)
        trade=ib.placeOrder(contract,ord)
        ib.sleep(1)
        logging.info(trade)
        logging.info('Placed market sell order')


    else:
        logging.info('market order already placed')
        print('market order already placed')
        return 0





def strategy(hist_df_hourly,hist_df_daily,ticker):
    logging.info('inside strategy')
    print('inside strategy')

    hourly_closing_price=hist_df_hourly.close.iloc[-1]
    buy_condition=hist_df_hourly['super'].iloc[-1]>0 and hist_df_daily['ema'].iloc[-1]<hist_df_hourly['close'].iloc[-1]
    # buy_condition=False
    sell_condition=hist_df_hourly['super'].iloc[-1]<0 and hist_df_daily['ema'].iloc[-1]>hist_df_hourly['close'].iloc[-1]
    # sell_condition=False
    current_balance=int(float([v for v in ib.accountValues(account=account_no) if v.tag == 'AvailableFunds' ][0].value))
    # atr_value=hist_df_daily['atr'].iloc[-1]
    # print(atr_value)
    # logging.info(atr_value)
    if current_balance>hist_df_hourly.close.iloc[-1]:
        if buy_condition:
            logging.info('buy condiiton satisfied')
            trade_buy_stocks(ticker)
        elif sell_condition:
            logging.info('sell condition satisfied')
            trade_sell_stocks(ticker)
        else :
            logging.info('no condition satisfied')
    else:
        logging.info('we dont have enough money')
        logging.info('current balance is',current_balance,'stock price is ',hist_df_hourly['close'].iloc[-1])


def close_ticker_postion(name):
    pos=ib.positions(account=account_no)
    if pos:
        df2=util.df(pos)
        df2['ticker_name']=[cont.symbol for cont in df2['contract']]
        cont=contract_objects[name]
        quant=df2[df2['ticker_name']==name].position.iloc[0]
        print(cont)
        print(quant)
        if quant>0:
            #sell
            # ord=MarketOrder(action='SELL',totalQuantity=quant)
            ord=Order(orderId=ib.client.getReqId(),orderType='MKT',totalQuantity=quantity_,action='SELL',account=account_no,tif=ord_validity)
            ib.placeOrder(cont,ord)
            logging.info('Closing position'+'SELL'+name)
          
        elif quant<0:
            #buy
            # ord=MarketOrder(action='BUY',totalQuantity=quant)
            ord=Order(orderId=ib.client.getReqId(),orderType='MKT',totalQuantity=quantity_,action='BUY',account=account_no,tif=ord_validity)
            ib.placeOrder(cont,ord)
            logging.info('Closing position'+'BUY'+name)



def close_ticker_open_orders(ticker):
    ord=ib.openTrades()
    
   
    if ord:
        df1=util.df(ord)
        print(df1.to_csv('new3.csv'))
        print(df1.columns)
        df1['ticker_name']=[cont.symbol for cont in df1['contract']]
        order_object=df1[df1['ticker_name']==ticker].order.iloc[0]
        print(order_object)
        ib.cancelOrder(order_object)
        logging.info('Canceled current order')




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
     

        hist_df_hourly=get_historical_data(ticker_contract,duration='10 D', candle='1 hour')
        hist_df_daily=get_historical_data(ticker_contract,duration='50 D', candle='1 day')
        print(hist_df_hourly)
        print(hist_df_daily)


        print(hist_df_hourly.close.iloc[-1])
      
        capital=int(float([v for v in ib.accountValues(account=account_no) if v.tag == 'AvailableFunds' ][0].value))
        print(capital)
        quantity=int((capital)//hist_df_hourly.close.iloc[-1])  
        print(quantity)
        logging.info('Checking condition')

        if quantity==0:
            logging.info('we dont have enough money so we cannot trade')
            continue

        if pos_df.empty:
            print('we dont have any position')
            logging.info('we dont have any position')
      
            strategy(hist_df_hourly,hist_df_daily,ticker)


        elif len(pos_df)!=0 and ticker not in pos_df['name'].tolist():
            logging.info('we have some position but current ticker is not in position')
            print('we have some position but current ticker is not in position')
            strategy(hist_df_hourly,hist_df_daily,ticker)
        


        elif len(pos_df)!=0 and ticker in pos_df["name"].tolist():
            logging.info('we have some position and current ticker is in position')
            print('we have some position and current ticker is in position')
            
            if pos_df[pos_df["name"]==ticker]["position"].values[0] == 0:
                logging.info('we have current ticker in position but quantity is 0')
                print('we have current ticker in position but quantity is 0')
                strategy(hist_df_hourly,hist_df_daily,ticker)

            elif pos_df[pos_df["name"]==ticker]["position"].values[0] > 0  :
                logging.info('we have current ticker in position and is long')
                print('we have current ticker in position and is long')
                sell_condition=hist_df_hourly['super'].iloc[-1]<0 and hist_df_daily['ema'].iloc[-1]>hist_df_hourly['close'].iloc[-1]
                # sell_condition=True
                # current_balance=int(float([v for v in ib.accountValues(account=account_no) if v.tag == 'AvailableFunds' ][0].value))
                # if current_balance>hist_df.close.iloc[-1]:
                if sell_condition:
                            hourly_closing_price=hist_df_hourly['close'].iloc[0]
                            # atr_value=hist_df_daily['atr'].iloc[-1]
                            print('sell condition satisfied')
                            logging.info('sell condition satisfied')
                            # close_ticker_open_orders(ticker)
                            close_ticker_postion(ticker)
                            strategy(hist_df_hourly,hist_df_daily,ticker)
                            # trade_sell_stocks(ticker,hourly_closing_price+atr_value)
                        

            elif pos_df[pos_df["name"]==ticker]["position"].values[0] < 0 :
                print('we have current ticker in position and is short')
                logging.info('we have current ticker in position and is short')
                hourly_closing_price=hist_df_hourly['close'].iloc[0]
                atr_value=hist_df_daily['atr'].iloc[-1]
                buy_condition=hist_df_hourly['super'].iloc[-1]>0 and hist_df_daily['ema'].iloc[-1]<hist_df_hourly['close'].iloc[-1]
                # buy_condition=True
                # current_balance=int(float([v for v in ib.accountValues(account=account_no) if v.tag == 'AvailableFunds' ][0].value))
                # if current_balance>hist_df.close.iloc[-1]:
         
                if buy_condition:
                            print('buy condiiton satisfied')
                            logging.info('buy condiiton satisfied')
                            # close_ticker_open_orders(ticker)
                            close_ticker_postion(ticker)
                            strategy(hist_df_hourly,hist_df_daily,ticker)
                        
                            
       





# hourly_data = get_historical_data(contract_objects[tickers[0]], duration='30 D', candle='1 hour')  
# print(hourly_data) 
# daily_data = get_historical_data(contract_objects[tickers[0]], duration='200 D', candle='1 day')
# print(daily_data)



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

ib.newOrderEvent += order_open_handler
ib.orderStatusEvent += order_open_handler
ib.cancelOrderEvent += order_open_handler

logging.info('Starting the main code with candle size'+str(candle_size))
main_strategy_code()
while dt.now(time_zone)<end_time:
    
    now = dt.now(time_zone)
    print(now)

    #running strategy every 1 min
    if now.second==1:
        main_strategy_code()

    # #running strategy every 5 min
    # if now.second==1 and now.minute in range(0,60,5):
    #     main_strategy_code()

    #  #running strategy every 15 min
    # if now.second==1 and now.minute in range(0,60,15):#[0,15,30,45]
    #     main_strategy_code()


    #  #running strategy every hour
    # if now.second==1 and now.minute ==1:
    #     main_strategy_code()

    time.sleep(1)



