import MetaTrader5 as mt5

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 failed to initialize. Error:", mt5.last_error())
    quit()

print("MT5 initialized successfully!")
print("MT5 Version:", mt5.version())

# --- Get Account Balance & Equity ---
account_info = mt5.account_info()
if account_info is None:
    print("Failed to get account info. Error:", mt5.last_error())
else:
    print("\n--- Account Information ---")
    print("Login:", account_info.login)
    print("Balance:", account_info.balance)
    print("Equity:", account_info.equity)
    print("Profit:", account_info.profit)
    print("Currency:", account_info.currency)
    print("Leverage:", account_info.leverage)

# --- Get Open Positions ---
positions = mt5.positions_get()
if positions is None:
    print("\nNo open positions or error:", mt5.last_error())
elif len(positions) > 0:
    print("\n--- Open Positions ---")
    for pos in positions:
        print(
            f"Position ID: {pos.ticket}",
            f"Symbol: {pos.symbol}",
            f"Type: {'BUY' if pos.type == mt5.ORDER_BUY else 'SELL'}",
            f"Volume: {pos.volume}",
            f"Open Price: {pos.price_open}",
            f"Current Price: {pos.price_current}",
            f"Profit: {pos.profit}",
            sep=" | "
        )
else:
    print("\nNo open positions found.")

# Shutdown MT5 connection
mt5.shutdown()