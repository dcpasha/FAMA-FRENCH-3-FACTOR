import pandas_datareader.data as web  # To import financial data directly into python
import pandas as pd
import statsmodels.api as sm


# Let's calculate Fama-French 3 Factor Model for a stock
def getStockPrice(stock_ticker, start_date, end_date):
    # Get Price data for any stock for any specified duration.
    stock = web.DataReader(stock_ticker, 'yahoo', start_date, end_date)

    # Getting the daily return of the stock.
    # We use the 'Adj Close' price of any stock.
    # It is the closing price after adjustments for all applicable splits and dividend distributions.
    stock = stock['Adj Close'].pct_change()[1:]

    # Converting the Series to Dataframe to do all of operations.
    stock = stock.to_frame()

    # Renaming the column.
    stock.rename(columns={'Adj Close': 'Return Pct'}, inplace=True)

    return stock


def getFamaFrench3Factor():
    ff_factors = pd.read_csv(
        '/Users/pavelpotapov/PycharmProjects/FamaFrench3factorModel/F-F_Research_Data_Factors_daily.CSV', skiprows=3)

    # Converting an object column to date format and set it as index of the dataframe.
    ff_factors['Date'] = pd.to_datetime(ff_factors['Unnamed: 0'], format='%Y%m%d', errors='coerce')
    ff_factors.set_index('Date', inplace=True)

    # Drop the unnecessary column and rename some columns.
    ff_factors.drop(columns='Unnamed: 0', inplace=True)
    ff_factors.rename(columns={'Mkt-RF': 'mrk_rf'}, inplace=True)
    # Drop the last row because it contains information about the data source.
    ff_factors.drop(ff_factors.tail(1).index, inplace=True)

    # Converting our factor data to percentages.
    ff_factors = ff_factors.apply(lambda x: x / 100)

    return ff_factors


def modelingFamaFrench(stock_ticker, start_date, end_date):
    stock = getStockPrice(stock_ticker,start_date,end_date)
    factors = getFamaFrench3Factor()

    # Merging two dataframes using their indexes. It is equivalent to SQL inner join statement.
    df = pd.merge(stock, factors, left_index=True, right_index=True)

    # Calculating 'Excessive Returns'
    df['Excess_returns'] = df['Return Pct'] - df['RF']

    # Modeling
    model = sm.formula.ols(formula="Excess_returns ~ mrk_rf + SMB + HML", data=df)
    result = model.fit()

    return result

# Now we can calculate the alpha and beta of any stock returns against the Fama & French’s 3 factors model.
results = modelingFamaFrench('MSFT', '2019-09-30', '2020-09-30')
print(results.summary())
