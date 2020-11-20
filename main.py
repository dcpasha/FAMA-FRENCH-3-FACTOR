import pandas_datareader.data as web  # To import financial data directly into python
import pandas as pd

# Let's calculate Fama-French 3 Factor Model for a stock
# Let's do it for Microsoft Corp ($MSFT)

start = '2019-09-30'
end = '2020-09-30'

# Import data for a security from yahoo finance
stock = web.DataReader('MSFT', 'yahoo', start, end)

# Import data for three Fama-French factors
ff_factors = pd.read_csv(
    '/Users/pavelpotapov/PycharmProjects/FamaFrench3factorModel/F-F_Research_Data_Factors_daily.CSV', skiprows=3)

# TODO: Both stock and ff_factors are dataframes. We need to reformat ff_factors, so the date looks just like in stock
# TODO: Merge them together and perform the analysis

# We are only interested in the "Adj Close" price of the stock. Drop the rest of the columns.
stock = stock['Adj Close']
stock = stock.pct_change()[1:]

