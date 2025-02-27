


from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook
import time
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract = ib.qualifyContracts(Option(symbol='BANKNIFTY',currency='INR',exchange='NSE',lastTradeDateOrContractMonth='20250227',right='C',strike=49200))[0]
print(contract)

ct=time.time()
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='10 D',
    barSizeSetting='1 hour', whatToShow='TRADES', useRTH=True)

# convert to pandas dataframe (pandas needs to be installed):
df = util.df(bars)
print(df)

time_taken=time.time()-ct
print(time_taken)