
from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=93)


# def onBarUpdate(bars, hasNewBar):
#     print(util.df(bars))


# contract2=Crypto('ETH','PAXOS','USD')
# bars = ib.reqRealTimeBars(contract2, 5, 'TRADES', False)
# bars.updateEvent += onBarUpdate

# ib.sleep(30)
# bars.updateEvent -= onBarUpdate
# ib.cancelRealTimeBars(bars)
# ib.run()




# contract1=Crypto('ETH','PAXOS','USD')
# contract2=Forex('GBPUSD')



# def abc(t):
#     t=list(t)[0]
#     print(t.contract.symbol,t.time,t.bid,t.ask)

# market_data=ib.reqMktData(contract1, "", False, False)
# market_data=ib.reqMktData(contract2, "", False, False)

# ib.pendingTickersEvent += abc
# # ib.sleep(20)
# # ib.pendingTickersEvent -= pending_tick
# ib.run()




import datetime as dt

ib.placeOrder(Stock('AMZN','SMART','USD'),LimitOrder('BUY',1,185.40))
print('placed order')
def order_handler(o):
    print(o)
    print(dt.datetime.now())



ib.newOrderEvent += order_handler
ib.orderStatusEvent += order_handler
ib.cancelOrderEvent += order_handler

ib.run()