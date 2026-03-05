'''
Stock Trading Intelligent Agent.

Holds several strategies - known best practices / techniques, heuristics, classical ML predictors.
Can execute multiple types of trade.

Can be initialised to several different markets (although once initialised, it is set to operate in that market only.)

Based on past performance of certain techniques use a Top-P / Top-K Sampling approach to decide which approach to use at the next opportunity.
Will place and manage trades autonomously without user intervention.

Can automatically generate graphs / diagrams to show performance of the portfolio.
'''

class Agent:
    '''
    The intelligent stock trading agent.
    '''

    def __init__(self, agentId, mic='XLON', preferredStrategy=None, bannedStrategies=[]):
        '''
        'mic': 
        - String representing the Market Identifier Code - defines what exchange this agent will be operating within.
        - NASDAQ = 'XNAS', London Stock Exchange = 'XLON', Hong Kong Exchanges and Clearing = 'XHKG', Japan Exchange Group = 'XPJX'.

        'preferredStrategy': String representing a preferred strategy, useful for experimentation with specific approaches.

        'bannedStrategies': List of banned strategies, useful for experimentation.
        '''
        self.mic = mic
        self.preferredStrategy = preferredStrategy
        self.bannedStrategies = bannedStrategies
        self.agentId = agentId
    
    def runIteration():
        '''
        Called iteratively by a background process.
        1. Fetches news and current prices for specific MIC.
        2. Decide approach based on past performance / sampling.
        3. Decide trade (and trade type) using approach chosen.
        4. Execute trade via stock exchange. 
        '''


