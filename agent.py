'''
Stock Trading Intelligent Agent.
'''

class agent:
    '''
    The intelligent stock trading agent.
    '''

    def __init__(self, mic='XLON', preferredStrategy=None, bannedStrategies=[]):
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

