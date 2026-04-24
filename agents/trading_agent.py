class TradingAgent:

    def __init__(self, initial_balance=100000):
        self.balance = initial_balance
        self.position = 0  # number of shares
        self.entry_price = 0

    def act(self, decision, price):

        action = decision["action"]

        # 🔥 STOP LOSS (5%)
        if self.position > 0 and price < self.entry_price * 0.95:
            print(f"🛑 STOP LOSS triggered at {price}")
            self.balance += self.position * price
            self.position = 0
            self.entry_price = 0
            return

        # 🎯 TAKE PROFIT (5%)
        if self.position > 0 and price > self.entry_price * 1.05:
            print(f"💰 TAKE PROFIT at {price}")
            self.balance += self.position * price
            self.position = 0
            self.entry_price = 0
            return

        # 🔥 TEMP: force entry for simulation
        if self.position == 0 and action == "SELL":
            action = "BUY"

        # 🟢 BUY
        if action == "BUY" and self.position == 0:
            qty = int(self.balance // price)
            if qty > 0:
                self.position = qty
                self.entry_price = price
                self.balance -= qty * price

                print(f"🟢 BUY {qty} shares at {price}")

        # 🔴 SELL
        elif action == "SELL" and self.position > 0:
            self.balance += self.position * price
            print(f"🔴 SELL {self.position} shares at {price}")

            self.position = 0
            self.entry_price = 0

        else:
            print("⏸️ HOLD")

    def status(self, current_price):

        portfolio_value = self.balance + (self.position * current_price)

        return {
            "balance": round(self.balance, 2),
            "position": self.position,
            "portfolio_value": round(portfolio_value, 2)
        }