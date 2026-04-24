from backtesting.simulator import run_backtest

result = run_backtest("RELIANCE.NS")

# 🔥 Print last 10 entries
for r in result[-10:]:
    print(r)

# 🔥 Print final portfolio value
if result:
    print("\nFinal Portfolio Value:", result[-1]["portfolio"])