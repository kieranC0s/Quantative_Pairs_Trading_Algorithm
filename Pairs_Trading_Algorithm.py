from AlgorithmImports import *
import numpy as np
import pandas as pd
from scipy import stats
from math import floor
from datetime import timedelta
from collections import deque
import itertools as it
from decimal import Decimal

class PairsTradingAlgorithm(QCAlgorithm):

    tickerSymbols = ["ADBE","AAP","AMD","AES","AFL","A","APD","AKAM","ALB","ALK","ARE","ALGN","ALLE","LNT","ALL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","ABC","AME","AMGN","APH","ADI","ANSS","ANTM","AON","APA","AAPL","AMAT","APTV","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","BKR","BLL","BAC","BBWI","BAX","BDX","BRK.B","BBY","BIO","TECH","BIIB","BLK","BK","BA","BKNG","BWA","BXP","BSX","BMY","AVGO","BR","BRO","BF.B","CHRW","CDNS","CZR","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","CNC","CNP","CDAY","CERN","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CTXS","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CPRT","GLW","CTVA","COST","CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DISCA","DISH","DG","DLTR","D","DPZ","DOV","DOW","DTE","DUK","DRE","DD","DXC","EMN","ETN","EBAY","ECL","EIX","EW","EA","LLY","EMR","ENPH","ETR","EOG","EFX","EQIX","EQR","ESS","EL","ETSY","RE","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FB","FAST","FRT","FDX","FIS","FITB","FRC","FE","FISV","FLT","FMC","F","FTNT","FTV","FBHS","FOX","BEN","FCX","GPS","GRMN","IT","GNRC","GD","GE","GIS","GM","GPC","GILD","GPN","GL","GS","HAL","HBI","HAS","HCA","PEAK","HSIC","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUM","HBAN","HII","IBM","IEX","IDXX","INFO","ITW","ILMN","INCY","IR","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","IPGP","IQV","IRM","JBHT","JKHY","J","SJM","JNJ","JCI","JPM","JNPR","KSU","K","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LEG","LDOS","LEN","LNC","LIN","LYV","LKQ","LMT","L","LOW","LUMN","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","TAP","MDLZ","MPWR","MNST","MCO","MS","MSI","MSCI","NDAQ","NTAP","NFLX","NWL","NEM","NWS","NEE","NLSN","NKE","NI","NSC","NTRS","NOC","NLOK","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","OKE","ORCL","OGN","OTIS","PCAR","PKG","PH","PAYX","PAYC","PYPL","PENN","PNR","PBCT","PEP","PKI","PFE","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PTC","PEG","PSA","PHM","PVH","QRVO","QCOM","PWR","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RHI","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS","SNA","SO","LUV","SWK","SBUX","STT","STE","SYK","SIVB","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","COO","HIG","HSY","MOS","TRV","DIS","TMO","TJX","TSCO","TT","TDG","TRMB","TFC","TWTR","TYL","TSN","USB","UDR","ULTA","UA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX","VFC","VIAC","VTRS","V","VNO","VMC","WRB","GWW","WAB","WBA","WMT","WM","WAT","WEC","WFC","WELL","WST","WDC","WU","WRK","WY","WHR","WMB","WLTW","WYNN","XEL","XLNX","XYL","YUM","ZBRA","ZBH","ZION","ZTS"]
    #Initialize method for the PairsTradingAlgorithm class.
    def Initialize(self):
        #Initializes start and end dates, sets cash, gets symbols and historical data, schedules "Rebalance" method to run on the first day of every month.
        self.SetStartDate(2020,1,1)
        self.SetEndDate(2021,1,1)
        self.SetCash(100000)
        self.threshold = 2.75
        self.symbols = []

        for i in PairsTradingAlgorithm.tickerSymbols:
            self.symbols.append(self.AddEquity(i, Resolution.Hour).Symbol)
        
        self.pairs = {}
        hoursPerDay = 8
        businessDaysPerWeek = 5
        weeksPerMonth = 4
        numberOfMonths = 2
        numberOfHours = numberOfMonths * weeksPerMonth * businessDaysPerWeek * hoursPerDay
        self.formation_period = numberOfHours
        self.history_price = {}

        for symbol in self.symbols:
            hist = self.History([symbol], self.formation_period+1, Resolution.Hour)
            if hist.empty: 
                PairsTradingAlgorithm.tickerSymbols.remove(symbol)
                self.symbols.remove(symbol)
            else:
                self.history_price[str(symbol)] = deque(maxlen=self.formation_period)
                for tuple in hist.loc[str(symbol)].itertuples():
                    self.history_price[str(symbol)].append(float(tuple.close))
                if len(self.history_price[str(symbol)]) < self.formation_period:
                    self.symbols.remove(symbol)
                    PairsTradingAlgorithm.tickerSymbols.remove(symbol)
                    self.history_price.pop(str(symbol))

        self.AddEquity("SPY", Resolution.Hour)
        self.Schedule.On(self.DateRules.MonthStart("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.Rebalance)
        self.count = 0
        self.symbol_pairs = None
        self.sorted_pairs = None
        
    
    # OnData method for the PairsTradingAlgorithm class.
    def OnData(self, data):
        # Updates the history_price dictionary with the current close prices for each symbol. If sorted_pairs is None, the method ends. Otherwise, it calculates position sizes, spreads, and means and standard deviations for each symbol pair in sorted_pairs, and opens or closes positions as necessary based on the spread and the threshold value.
        for symbol in self.symbols:
            #ether the data object contains a key with the current symbol and whether the string version of symbol is in the history_price dictionary.
            if data.Bars.ContainsKey(symbol) and str(symbol) in self.history_price:
                self.history_price[str(symbol)].append(float(data[symbol].Close))
        if self.sorted_pairs is None: return

        # calculate positon sizes
        positionSizes = []
        n = len(self.sorted_pairs)
        for i in range(n):
            x = float(i + 1) / float(n)
            x = 1 - x
            x *= 8
            value = 1 / (1 + math.exp(x - 4))
            positionSizes.append(value)

        positionsSum = sum(positionSizes)
        normalizationFactor = 1 / positionsSum
        for i, p, in enumerate(positionSizes):
            positionSizes[i] = p * normalizationFactor
        positionSizes = list(reversed(positionSizes))

        for index, pair in enumerate(self.sorted_pairs): 
            stock1 = pair[0]
            stock2 = pair[1]

            # calculate the spread of two price series
            if stock1 not in self.history_price or stock2 not in self.history_price:
                continue
            self.Debug("Pair: " + str(stock1) + " " + str(stock2))
            #calculates the difference between the two price series for the current pair of symbols and stores it in a variable called spread.
            spread = np.array(self.history_price[str(stock1)]) - np.array(self.history_price[str(stock2)])
            mean = np.mean(spread)
            std = np.std(spread)
            #calculates the ratio of the price of the first symbol to the price of the second symbol in the current pair and stores it in a variable called ratio.
            ratio = self.Portfolio[stock1].Price / self.Portfolio[stock2].Price
            if spread[-1] > mean + self.threshold * std:
                if not self.Portfolio[stock1].Invested and not self.Portfolio[stock2].Invested:
                    self.SetHoldings(stock1, -positionSizes[index])
                    self.SetHoldings(stock2, positionSizes[index])            
            
            elif spread[-1] < mean - self.threshold * std: 
                if not self.Portfolio[stock1].Invested and not self.Portfolio[stock2].Invested:
                    self.SetHoldings(stock1, positionSizes[index])
                    self.SetHoldings(stock2, -positionSizes[index]) 
                    
            # the position is closed when prices revert back
            elif self.Portfolio[stock1].Invested and self.Portfolio[stock2].Invested:
                    self.Liquidate(stock1) 
                    self.Liquidate(stock2)                

    #Rebalance method for the PairsTradingAlgorithm class.
    def Rebalance(self):
        #Assigns the list of correlated stock pairs returned by the getCorrelatedStocks method to the symbol_pairs and sorted_pairs attributes. If prevPairs is not None, it liquidates any pairs in prevPairs that are not in sorted_pairs.
        prevPairs = self.sorted_pairs
        self.symbol_pairs = list(map(lambda s: [s.ticker1, s.ticker2], self.getCorrelatedStocks()))
        self.sorted_pairs = self.symbol_pairs

        if prevPairs is not None:
            for pair in prevPairs:
                if pair not in self.sorted_pairs:
                    self.Liquidate(pair[0])
                    self.Liquidate(pair[1])

    #getCorrelatedStocks method for the PairsTradingAlgorithm class.
    def getCorrelatedStocks(self):
        #Gets the historical data for the initialTicker symbol, converts it to a pivot table, and gets the correlations between the initialTicker and the other ticker symbols in tickerSymbols. It then creates a list of Correlation objects, filters it to remove correlated pairs with a score of 1 (perfect correlation) and duplicate tickers, and returns the resulting list.
        initialTicker = "AAPL"
        self.Securities[initialTicker].SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        hoursPerDay = 8
        businessDaysPerWeek = 5
        weeksPerMonth = 4
        numberOfMonths = 2
        numberOfHours = numberOfMonths * weeksPerMonth * businessDaysPerWeek * hoursPerDay
        history = self.History(self.Symbol(initialTicker), numberOfHours, Resolution.Hour)
        dates = list(map(lambda t: t[1], history.index.tolist()))
        symbols = [initialTicker for i in range(len(dates))]
        data = {
            'Date': dates,
            'Close': list(history['close']),
            'Symbol': symbols,
        }
        df = pd.DataFrame(data)
        df_pivot = df.pivot('Date','Symbol','Close').reset_index()
        for ticker in PairsTradingAlgorithm.tickerSymbols:
            if ticker == "AAPL":
                continue
            self.Securities[ticker].SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            hoursPerDay = 8
            businessDaysPerWeek = 5
            weeksPerMonth = 4
            numberOfMonths = 2
            numberOfHours = numberOfMonths * weeksPerMonth * businessDaysPerWeek * hoursPerDay
            h = self.History(self.Symbol(ticker), numberOfHours, Resolution.Hour) 
            if 'close' not in h.columns:
                continue
            values = h['close'].tolist()
            try:
                df_pivot[ticker] = values
            except:
                pass
        corr_df = df_pivot.corr(method='pearson')
        # Convert matrix to list of correlations
        correlations = []
        i = 0
        for row in corr_df.columns:
            for col in corr_df.columns[i:]:
                val = corr_df.loc[row, col]
                if val == 1:
                    continue
                corr = Correlation(row, col, val)
                correlations.append(corr)
            i += 1
        numberOfPairs = 30
        # Sort correlated stocks
        correlations = sorted(correlations, key=lambda c: c.score, reverse=True)
        uniquedCorrelations = []
        tickerSet = set()
        pairCount = 0
        for correlation in correlations:
            if pairCount >= numberOfPairs:
                break
            if correlation.ticker1 in tickerSet or correlation.ticker2 in tickerSet:
                continue
            tickerSet.add(correlation.ticker1)
            tickerSet.add(correlation.ticker2)
            uniquedCorrelations.append(correlation)
            pairCount += 1
        return uniquedCorrelations



#Pair class for representing a pair of correlated stock symbols.
class Correlation:
    #Contains attributes for the symbols and their respective prices, as well as a distance method for calculating the sum of squared deviations between the normalized prices of the two symbols.
    def __init__(self, ticker1, ticker2, score):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.score = score

    def __eq__(self, other):
        return self.ticker1 == other.ticker1 and self.ticker2 == other.ticker2
        
    def __le__(self, other):
        return self.score < other.score
    
    def __str__(self):
        return self.ticker1 + ", " + self.ticker2 + " = " + str(self.score)