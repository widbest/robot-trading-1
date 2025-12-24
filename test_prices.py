from data_provider import DataProvider

provider = DataProvider()
print('Fetching current market prices...')

prices = provider._fetch_current_market_prices()
print('Current market prices:')
for instrument, price in prices.items():
    print(f'  {instrument}: ${price:.2f}')

for instrument in ['XAU/USD', 'NDX100', 'GER40']:
    print(f'\nTesting {instrument}:')
    data = provider.get_price_data(instrument, '1H', 50)
    if data is not None:
        print(f'  Success: {len(data)} periods')
        print(f'  Current: ${data["close"].iloc[-1]:.2f}')
    else:
        print('  No data available')