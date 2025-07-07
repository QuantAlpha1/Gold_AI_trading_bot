import MetaTrader5 as mt5

mt5.initialize()
print("All MT5 attributes:", dir(mt5))  # Should show many methods
print("Example: EURUSD bid:", mt5.symbol_info("EURUSD").bid)
mt5.shutdown()