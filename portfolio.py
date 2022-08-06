import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np

##################################################

st.write("""
# Customize your portfolio
""")
'Please input the stock you might consider in portfolio.'
sd = st.date_input("Start date", datetime.date(2020,5,31))
ed = st.date_input("End date")
stockGroup = st.selectbox('Input the stock you want to select: ',
    ('- please select your ans -','HSI','all stock in HSI'))

HSI = {'0001.HK','0002.HK','0003.HK','0005.HK','0006.HK',
        '0011.HK','0012.HK','0016.HK','0017.HK','0027.HK',
        '0066.HK','0101.HK','0175.HK','0241.HK','0267.HK',
        '0288.HK','0291.HK','0316.HK','0386.HK','0388.HK',
        '0669.HK','0688.HK','0700.HK','0762.HK','0823.HK',
        '0857.HK','0868.HK','0881.HK','0883.HK','0939.HK',
        '0941.HK','0960.HK','0968.HK','0981.HK','0992.HK',
        '1038.HK','1044.HK','1093.HK','1109.HK','1113.HK',
        '1177.HK','1211.HK','1299.HK','1378.HK','1398.HK',
        '1810.HK','1876.HK','1928.HK','1997.HK','2007.HK',
        '2020.HK','2269.HK','2313.HK','2318.HK','2319.HK',
        '2331.HK','2382.HK','2388.HK','2628.HK','2688.HK',
        '3690.HK','3968.HK','3988.HK','6098.HK','6862.HK',
        '9618.HK','9633.HK','9988.HK','9999.HK'}
USS = {'AAPL','TSLA','MSFT','AAL'}
#HSI = {'0005.HK','0006.HK','0016.HK','0316.HK','1211.HK','2388.HK','9999.HK'}

if stockGroup == "- please select your ans -" :
    st.stop()

if stockGroup == 'HSI':
    selectedStock = st.multiselect('Please select at least 2 stocks: ',
        HSI)
if stockGroup == 'all stock in HSI':
    selectedStock = HSI

if len(selectedStock) == 0:
    st.stop()

colT1, colT2, colT3 = st.columns(3)
with colT3:
    if st.checkbox('Compare the selected stocks') == False:
        st.stop()
    
##################################################

# show selected stock method 1 ################
#'The stock you selected are: ', selectedStock

# show selected stock method 2 ################
#i = str(selectedStock)
#st.write('The stock you selected are: ', i)

# show selected stock method 3 ################
st.write("""
##### The stocks you have selected are:
""")
for i in range(len(selectedStock)):
    st.write(i+1 , ' : ', selectedStock[i])

##################################################

#get adj. closing price in once
dailyPrice = yf.download(selectedStock,sd)['Adj Close']
title = "Portfolio Adj close Price History"

#st.write(dailyPrice)
if '0016.HK' in dailyPrice.columns:
#if dailyPrice.columns.str.contains('0016.HK'):
    dailyPrice.at['2022-06-14T00:00:00','0016.HK']=93.30
#st.write(dailyPrice)

##################################################

st.write("""
### Daily Price of Stocks
""")

#plt.figure(figsize=(20, 8))
#for c in df.columns:
#  plt.plot(df[c], label = c)
  
#plt.figure(figsize=(20, 8))
#for c in df.columns:
#  st.line_chart(df[c], label = c)
st.line_chart(dailyPrice)

#plt.title(title)
#plt.xlabel('Date', fontsize = 18)
#plt.ylabel('Adj close Price USD($)', fontsize = 18)
#plt.legend(df.columns.values, loc = 'upper left')
#plt.show()

##################################################

st.write("""
## Daily Return of Stocks
""")
dailyReturn = dailyPrice.pct_change()
st.line_chart(dailyReturn)

##################################################
##################################################
##################################################
##################################################

#Annualized Covariance Matrix
covMatrixAnnual = dailyReturn.cov()*252
#st.write(covMatrixAnnual)

##################################################

st.write("""
## Correlation between Stocks **need to discuss use method 1 or method 2
""")
#corr info: https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
#color scale: https://plotly.com/python/colorscales/, 'temps' 

# Method 1 #################################
st.write("""
##### Method 1 (copy by notes)
""")
corrMatrix = covMatrixAnnual.corr().round(2) # round to 2 decimal places
corrFig = px.imshow(corrMatrix, zmin = -1, zmax = 1, color_continuous_scale='rdylgn', text_auto=True)
st.write(corrFig)
st.write(corrMatrix)

# Method 2 #################################
st.write("""
##### Method 2 (by using daily return)
""")
corrMatrix = dailyReturn.corr().round(2) # round to 2 decimal places
corrFig = px.imshow(corrMatrix, zmin = -1, zmax = 1, color_continuous_scale='rdylgn', text_auto=True)
st.write(corrFig)
st.write(corrMatrix)

##################################################
colT1, colT2, colT3 = st.columns(3)
with colT3:
    if st.checkbox('Show optimize portfolio weighting') == False:
        st.stop()

st.write("""
## Portfolio Efficient Frontier
""")

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
# need pip install pyportfolioopt


mu = expected_returns.mean_historical_return(dailyPrice)
S = risk_models.sample_cov(dailyPrice)
#st.write(mu)
#st.write(S)

def plot_efficient_frontier_and_max_sharpe(mu, S):  
    # Optimize portfolio for maximal Sharpe ratio 
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(8,6))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ =   ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r",     label="Max Sharpe")
# Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
# Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend()
    plt.grid()
    #plt.tight_layout()
    #plt.show()
    return fig

st.write(plot_efficient_frontier_and_max_sharpe(mu, S))

##################################################

st.write("""
## Stock weighting in Portoflio
""")

ef = EfficientFrontier(mu, S)
ef.max_sharpe(risk_free_rate=0.02)
weights = ef.clean_weights()
st.write(weights)

weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['weights']
st.write(weights_df)

weights = np.array(weights_df['weights'])
st.write(weights)

##################################################

colT1, colT2 = st.columns(2)
with colT1:

    st.write("""
    ## Portoflio Performance by package"pypfopt"
    """)
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    st.write('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
    st.write('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
    st.write('Sharpe ratio: {}'.format(sharpe_ratio.round(2)))
    st.write(ef.portfolio_performance())

##################################################
with colT2:
    st.write("""
    ## Portoflio Performance by using same weighting
    """)

    #'Portfolio Variance:'
    port_variance = np.dot(weights.T,np.dot(covMatrixAnnual, weights))
    #st.write(port_variance)

    #'Portfolio Volatility:'
    port_volatility = np.sqrt(port_variance)
    #st.write(port_volatility)

    #'Portfolio simple annual return'
    portfoilSimpAnnualRe = np.sum(dailyReturn.mean() * weights) * 252
    #st.write(portfoilSimpAnnualRe)

    percent_var = str(round(port_variance, 2)* 100) + '%'
    percent_vols = str(round(port_volatility, 2)* 100) + '%'
    percent_ret = str(round(portfoilSimpAnnualRe, 2)* 100) + '%'

    st.write('Expected annual return: ' + percent_ret)
    st.write('Annual volatility / risk: ' + percent_vols)
    st.write('Annual variance: ' + percent_var)


##################################################
##################################################
colT1, colT2, colT3 = st.columns(3)
with colT3:
    box = st.checkbox('run portfolio by enter weighting')

if box == True:

    st.write("""
    ## Portoflio Performance by enter weighting
    """)

    weights = np.array([])

    for tickerSymbol in selectedStock:
        t = st.number_input('Stock '+ tickerSymbol + ' weighting is')
        weights = np.append(weights,t)    

    if st.checkbox('show performance') == True:

        st.write(weights)

        'Portfolio Variance:'
        port_variance = np.dot(weights.T,np.dot(covMatrixAnnual, weights))
        st.write(port_variance)

        'Portfolio Volatility:'
        port_volatility = np.sqrt(port_variance)
        st.write(port_volatility)

        'Portfolio simple annual return'
        portfoilSimpAnnualRe = np.sum(dailyReturn.mean() * weights) * 252
        st.write(portfoilSimpAnnualRe)

        percent_var = str(round(port_variance, 2)* 100) + '%'
        percent_vols = str(round(port_volatility, 2)* 100) + '%'
        percent_ret = str(round(portfoilSimpAnnualRe, 2)* 100) + '%'

        st.write('Expected annual return: ' + percent_ret)
        st.write('Annual volatility / risk: ' + percent_vols)
        st.write('Annual variance: ' + percent_var)


##################################################
##################################################





##################################################

st.write("""
# Details of each Stocks (in optimize porfotlio)
""")

weights_df = weights_df[weights_df.weights>0]
selectedStock = list(weights_df.index)
#st.write(selectedStock)
#st.write(weights_df)

for tickerSymbol in selectedStock:
    #tickerSymbol = i
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1d', start=sd, end=ed)
    
    st.header("""Stock : """ + str(tickerSymbol) + "")
    # text size: https://docs.streamlit.io/library/api-reference/text

    st.write("""
    ##### Stock close Price
    """)
    st.line_chart(tickerDf.Close)
    #st.write(tickerDf)
    
    st.write("""
    ##### Stock Daily Volumn
    """)
    st.line_chart(tickerDf.Volume)
    
    st.write("""
    ##### Stock Simple Return
    """)
    tickerDf['simple_rtn'] = tickerDf.Close.pct_change()
    st.line_chart(tickerDf.simple_rtn)


##################################################







