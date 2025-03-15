


from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook
import time
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=15)

ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract=Stock('TSLA','SMART','USD')
# contract = ib.qualifyContracts(Future(symbol='BANKNIFTY',currency='INR',exchange='NSE',lastTradeDateOrContractMonth='202501'))[0]
# contract = ib.qualifyContracts(Future(symbol='BANKNIFTY',currency='INR',exchange='NSE',lastTradeDateOrContractMonth='20250227'))[0]
# contract=Contract(symbol='BANKNIFTY',currency='INR',exchange='NSE',secType='IND')
print(contract)


ct=time.time()

import datetime as dt
bars = ib.reqHistoricalTicks(
    contract, startDateTime=dt.datetime.now()-dt.timedelta(days=1),endDateTime=dt.datetime.now(),
    whatToShow='TRADES',numberOfTicks=1000,useRth=True)

# convert to pandas dataframe (pandas needs to be installed):
df = util.df(bars)
print(df.to_csv('data_tick1.csv'))

time_taken=time.time()-ct
print(time_taken)


