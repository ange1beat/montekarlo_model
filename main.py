import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocksFile = pd.read_csv('./all_stocks.csv')
stocks = pd.DataFrame(data = stocksFile)

tickers = ['AAL', 'AAPL', 'MSFT', 'AMZN']

stocks_closed_AAL = stocks[(stocks.Name == 'AAL')]
stocks_closed_AAPL = stocks[(stocks.Name == 'AAPL')]
stocks_closed_MSFT = stocks[(stocks.Name == 'MSFT')]
stocks_closed_AMZN = stocks[(stocks.Name == 'AMZN')]

n1 = np.array(stocks_closed_AAL['close'])
n2 = np.array(stocks_closed_AAPL['close'])
n3 = np.array(stocks_closed_MSFT['close'])
n4 = np.array(stocks_closed_AMZN['close'])

all_stocks = pd.DataFrame({
     'AAL': n1, 
     'AAPL': n2, 
     'MSFT' : n3, 
     'AMZN' : n4
    })


pct_stocks = all_stocks.pct_change()
mean_stocks = pct_stocks.mean()
cov_stocks = pct_stocks.cov()

print(all_stocks.head())

robot = np.array([0.1, 0.3, 0.1, 0.5])

revenue_stock = np.sum(mean_stocks * robot)
standart_deviation = np.sqrt(np.dot(robot.T, np.dot(cov_stocks, robot)))
sharpo_robot = revenue_stock/standart_deviation

robot_result = np.array([revenue_stock, standart_deviation, sharpo_robot])
robot_result = np.array([revenue_stock, standart_deviation, sharpo_robot])
robot_result = np.concatenate((robot_result, robot), axis = 0)
robot_sim_result = pd.DataFrame(robot_result, columns = ['Robot'], index=['ret', 'stdev','sharpe', tickers[0], tickers[1], tickers[2], tickers[3]])

print(robot_sim_result)

num_iterations = 10000
simulation_res = np.zeros((4+len(tickers)-1,num_iterations))

for i in range(num_iterations):
    weights = np.array(np.random.random(4))
    weights /= np.sum(weights)

    revenue_stock1 = np.sum(mean_stocks * weights)
    standart_deviation1 = np.sqrt(np.dot(weights.T, np.dot(cov_stocks, weights)))

    simulation_res[0, i] = revenue_stock1
    simulation_res[1, i] = standart_deviation1

    simulation_res[2, i] = simulation_res[0, i] / simulation_res[1, i]

    for j in range(len(weights)):
        simulation_res[j+3, i] = weights[j]

sim_frame = pd.DataFrame(simulation_res.T, columns=['ret','stdev', 'sharpe', tickers[0], tickers[1], tickers[2], tickers[3]])

max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()]
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]

print ("The portfolio for max Sharpe Ratio:\n", max_sharpe)
print ("The portfolio for min risk:\n", min_std)

fig, ax = plt.subplots(figsize=(10, 10))

plt.scatter(sim_frame.stdev,sim_frame.ret,c=sim_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')


plt.scatter(max_sharpe[1],max_sharpe[0],marker=(5,1,0),color='r',s=600)

plt.scatter(min_std[1],min_std[0],marker=(5,1,0),color='b',s=600)

plt.scatter(standart_deviation, revenue_stock,marker=(5,1,0),color='g',s=600)

plt.show()


