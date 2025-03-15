
import datetime as dt
from ib_async import *

# Connect to Interactive Brokers
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the contract (Example: AAPL stock)




def fetch_historical_data(contract, start_date, end_date, months_per_batch=1):
    """
    Fetch historical data for a given contract between start_date and end_date in 2-month intervals.
    """

    
    historical_data = []
    current_end_date = end_date
    
    while start_date < current_end_date:
        end_date_str = current_end_date.strftime("%Y%m%d %H:%M:%S")
        ib.reqhist
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_date_str,
            durationStr=f'{months_per_batch} M',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True
        )
        print(util.df(bars))
        if bars:
            historical_data.extend(b for b in bars if b not in historical_data)
        
        current_end_date -= dt.timedelta(days=months_per_batch * 30)
    
    
    return historical_data


# contract = Index('NIFTY50','NSE','INR')
contract=Stock('TSLA','SMART','USD')
start_date = dt.datetime.now() - dt.timedelta(days=100)
end_date = dt.datetime.now()

data = fetch_historical_data(contract, start_date, end_date)
df=util.df(data)
print(df)
df.to_csv('data.csv')

