import pandas_datareader.data as web  # To import financial data directly into python
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Let's calculate Fama-French 3 Factor Model for a stock
# Let's do it for Microsoft Corp ($MSFT)

start = '2019-09-30'
end = '2020-09-30'

# Import data for a security from yahoo finance
stock = web.DataReader('MSFT', 'yahoo', start, end)

# Import data for three Fama-French factors
ff_factors = pd.read_csv(
    '/Users/pavelpotapov/PycharmProjects/FamaFrench3factorModel/F-F_Research_Data_Factors_daily.CSV', skiprows=3)

# 9:23 am
# TODO (DONE): Both stock and ff_factors are dataframes. We need to reformat ff_factors, so the date looks just like in stock.
# TODO (DONE): Merge them together
# TODO (DONE): Perform the analysis
# TODO: Write my own function for OLS

# We are only interested in the 'Adj Close' price of the stock, but we will also keep the 'Volume'
# 'Adj Close' is the closing price after adjustments for all applicable splits and dividend distributions.
# 'Volume' is how many shars are traded each day.
stock = stock.loc[:, ['Volume', 'Adj Close']]
stock['Return Pct'] = stock['Adj Close'].pct_change()[1:]
stock.dropna(subset=['Return Pct'], inplace=True)

# Converting an object column to date format and set it as index of the dataframe.
ff_factors['Date'] = pd.to_datetime(ff_factors['Unnamed: 0'], format='%Y%m%d', errors='coerce')
ff_factors.set_index('Date', inplace=True)

# Drop the useless column and the last row because it contains information about the data source.
ff_factors.drop(columns='Unnamed: 0', inplace=True)
ff_factors.drop(ff_factors.tail(1).index, inplace=True)

# Merging two dataframes using their indexes. It is equivalent to SQL inner join statement.
df = pd.merge(stock, ff_factors, left_index=True, right_index=True)
df.drop(columns=['Volume', 'Adj Close'], inplace=True)
df['Excess returns'] = df['Return Pct'] - df['RF']

# Modeling.
# Do we need to do this? Divide them by 100 because our factors are whole numbers and not percentages.
# ff_factors/100
df1 = ff_factors.copy()
df1 = df1.apply(lambda x: x/100)

df1.rename(columns={'Mkt-RF':'mrk_rf'}, inplace=True)

df1 = pd.merge(stock, df1, left_index=True, right_index=True)
df1.drop(columns=['Volume', 'Adj Close'], inplace=True)
df1['Excess_returns'] = df1['Return Pct'] - df1['RF']

model = sm.formula.ols(formula="Excess_returns ~ mrk_rf + SMB + HML", data=df1)
results = model.fit()
print(results.summary())

# Now you know how to calculate the alpha and beta of any portfolio returns against the Fama & French’s 3 factors model.
