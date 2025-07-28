import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import requests
import threading
import time
from datetime import datetime

MOCK_PRICES = {
    'BTC': 65000,
    'ETH': 3000,
    'SOL': 150,
    'XRP': 0.5,
    'ADA': 0.4,
    'BNB': 500
}

class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(model, history):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    data = np.array(history).reshape(-1, 1).astype(np.float32)
    X = torch.tensor(data[:-1])
    y = torch.tensor(data[1:])
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

def fetch_price(coin):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin.lower()}&vs_currencies=usd"
        response = requests.get(url)
        return response.json()[coin.lower()]['usd']
    except:
        return MOCK_PRICES.get(coin, 0)

def fetch_historical(coin, days=1):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin.lower()}/market_chart?vs_currency=usd&days={days}"
        response = requests.get(url)
        data = response.json()['prices']
        df = pd.DataFrame(data, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame({'timestamp': [datetime.now()], 'price': [fetch_price(coin)]})

class CyberdudebivashCryptoPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyberdudebivash Crypto Platform")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')

        self.wallet = {'USD': 10000, 'BTC': 0.1, 'ETH': 1, 'SOL': 10}
        self.prices = {}
        self.history = {'BTC': [fetch_price('BTC')] * 20}
        self.pnl_history = []
        self.futures_positions = {}
        self.current_pair = 'BTC'
        self.historical_data = fetch_historical('BTC')

        self.ai_model = SimplePredictor()
        train_model(self.ai_model, self.history['BTC'])
        self.ai_running = False
        self.ai_thread = threading.Thread(target=self.ai_auto_trade, daemon=True)

        self.create_gui()
        self.update_prices()

    def create_gui(self):
        # Top bar
        top_frame = tk.Frame(self.root, bg='#000', height=50)
        top_frame.pack(fill='x')
        tk.Label(top_frame, text="Cyberdudebivash", bg='#000', fg='#fff', font=('Arial', 16)).pack(side='left', padx=10)
        for text in ['Trade', 'Earn', 'QuickBuy', 'Learn']:
            tk.Label(top_frame, text=text, bg='#000', fg='#fff', font=('Arial', 12)).pack(side='left', padx=20)
        tk.Button(top_frame, text="Login", bg='#ff6600', fg='#fff').pack(side='right', padx=10)
        tk.Button(top_frame, text="Register", bg='#ff6600', fg='#fff').pack(side='right', padx=10)

        # Pair info
        pair_frame = tk.Frame(self.root, bg='#1a1a1a', height=50)
        pair_frame.pack(fill='x')
        self.pair_label = tk.Label(pair_frame, text="BTC/USD", bg='#1a1a1a', fg='#fff', font=('Arial', 14))
        self.pair_label.pack(side='left', padx=10)
        self.price_info_label = tk.Label(pair_frame, text="", bg='#1a1a1a', fg='#fff')
        self.price_info_label.pack(side='left', padx=20)
        tk.Button(pair_frame, text="Update", command=self.update_prices, bg='#333', fg='#fff').pack(side='left')

        # Main content grid
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_columnconfigure(2, weight=1)

        # Left: Coin list
        left_frame = tk.Frame(main_frame, bg='#1a1a1a')
        left_frame.grid(row=0, column=0, sticky='nsew')
        self.coin_tree = ttk.Treeview(left_frame, columns=('Coin', 'Price', 'Change'), show='headings')
        self.coin_tree.heading('Coin', text='Coin')
        self.coin_tree.heading('Price', text='Price')
        self.coin_tree.heading('Change', text='24h %')
        self.coin_tree.pack(fill='both', expand=True)
        self.coin_tree.bind('<<TreeviewSelect>>', self.select_pair)

        # Center: Chart
        center_frame = tk.Frame(main_frame, bg='#1a1a1a')
        center_frame.grid(row=0, column=1, sticky='nsew')
        self.fig, self.ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=center_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        timeframe_frame = tk.Frame(center_frame, bg='#1a1a1a')
        timeframe_frame.pack()
        for tf in ['1m', '5m', '15m', '1h', '4h', '1d', '1w']:
            tk.Button(timeframe_frame, text=tf, bg='#333', fg='#fff', command=lambda t=tf: self.change_timeframe(t)).pack(side='left')

        # Right: Order book and trade
        right_frame = tk.Frame(main_frame, bg='#1a1a1a')
        right_frame.grid(row=0, column=2, sticky='nsew')
        tk.Label(right_frame, text="Order Book", bg='#1a1a1a', fg='#fff').pack()
        self.order_tree = ttk.Treeview(right_frame, columns=('Price', 'Qty', 'Total'), show='headings')
        self.order_tree.heading('Price', text='Price')
        self.order_tree.heading('Qty', text='Qty')
        self.order_tree.heading('Total', text='Total')
        self.order_tree.pack(fill='y', expand=True)
        trade_panel = tk.Frame(right_frame, bg='#1a1a1a')
        trade_panel.pack()
        tk.Label(trade_panel, text="Amount:", bg='#1a1a1a', fg='#fff').pack()
        self.amount_entry = tk.Entry(trade_panel)
        self.amount_entry.pack()
        tk.Button(trade_panel, text="Buy BTC", bg='#00ff00', command=self.buy_coin).pack(side='left')
        tk.Button(trade_panel, text="Sell BTC", bg='#ff0000', command=self.sell_coin).pack(side='left')

        # AI Toggle
        ai_button = tk.Button(self.root, text="Toggle AI Auto-Trade", command=self.toggle_ai, bg='#ff6600', fg='#fff')
        ai_button.pack(side='bottom')

    def select_pair(self, event):
        selected = self.coin_tree.selection()
        if selected:
            self.current_pair = self.coin_tree.item(selected[0])['values'][0]
            self.historical_data = fetch_historical(self.current_pair)
            self.update_chart()
            self.update_pair_info()
            self.update_order_book()

    def change_timeframe(self, tf):
        days_map = {'1m': 0.001, '5m': 0.003, '15m': 0.01, '1h': 0.04, '4h': 0.16, '1d': 1, '1w': 7}
        days = days_map.get(tf, 1)
        self.historical_data = fetch_historical(self.current_pair, days=days)
        self.update_chart()

    def update_prices(self):
        for coin in MOCK_PRICES:
            self.prices[coin] = fetch_price(coin)
        self.coin_tree.delete(*self.coin_tree.get_children())
        for coin in self.prices:
            change = ((self.prices[coin] - MOCK_PRICES[coin]) / MOCK_PRICES[coin]) * 100 if MOCK_PRICES[coin] else 0
            tag = 'green' if change > 0 else 'red'
            self.coin_tree.insert('', 'end', values=(coin, f"${self.prices[coin]:.2f}", f"{change:.2f}%"), tags=(tag,))
        self.coin_tree.tag_configure('green', foreground='green')
        self.coin_tree.tag_configure('red', foreground='red')
        self.update_pair_info()
        self.update_chart()
        self.update_order_book()

    def update_pair_info(self):
        price = self.prices.get(self.current_pair, 0)
        change = ((price - MOCK_PRICES.get(self.current_pair, 0)) / MOCK_PRICES.get(self.current_pair, 0)) * 100 if MOCK_PRICES.get(self.current_pair, 0) else 0
        high = self.historical_data['price'].max()
        low = self.historical_data['price'].min()
        volume = np.random.randint(1000000, 10000000)  # Mock
        text = f"Price: ${price:.2f} | Change: {change:.2f}% | High: ${high:.2f} | Low: ${low:.2f} | Vol: {volume}"
        self.price_info_label.config(text=text)

    def update_chart(self):
        self.ax.clear()
        df = self.historical_data
        self.ax.plot(df['timestamp'], df['price'], color='white')
        self.ax.set_title(f"{self.current_pair} Chart", color='white')
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylabel('Price (USD)', color='white')
        self.ax.grid(color='gray')
        self.ax.tick_params(colors='white')
        self.canvas.draw()

    def update_order_book(self):
        self.order_tree.delete(*self.order_tree.get_children())
        price = self.prices.get(self.current_pair, 0)
        for i in range(5):
            bid_price = price - i * 0.01 * price
            ask_price = price + i * 0.01 * price
            qty = np.random.uniform(0.1, 10)
            self.order_tree.insert('', 'end', values=(f"{bid_price:.2f}", f"{qty:.4f}", f"{bid_price * qty:.2f}"), tags=('green',))
            self.order_tree.insert('', 'end', values=(f"{ask_price:.2f}", f"{qty:.4f}", f"{ask_price * qty:.2f}"), tags=('red',))
        self.order_tree.tag_configure('green', foreground='green')
        self.order_tree.tag_configure('red', foreground='red')

    def buy_coin(self):
        amount = float(self.amount_entry.get())
        price = self.prices.get(self.current_pair, 0)
        cost = amount * price
        if self.wallet['USD'] >= cost:
            self.wallet['USD'] -= cost
            self.wallet[self.current_pair] = self.wallet.get(self.current_pair, 0) + amount
            print(f"Bought {amount} {self.current_pair} (Spot)")

    def sell_coin(self):
        amount = float(self.amount_entry.get())
        if self.wallet.get(self.current_pair, 0) >= amount:
            price = self.prices.get(self.current_pair, 0)
            revenue = amount * price
            self.wallet[self.current_pair] -= amount
            self.wallet['USD'] += revenue
            print(f"Sold {amount} {self.current_pair} (Spot)")

    def toggle_ai(self):
        if self.ai_running:
            self.ai_running = False
        else:
            self.ai_running = True
            self.ai_thread = threading.Thread(target=self.ai_auto_trade, daemon=True)
            self.ai_thread.start()

    def ai_auto_trade(self):
        while self.ai_running:
            train_model(self.ai_model, self.history['BTC'])
            last_price = torch.tensor([[self.history['BTC'][-1]]])
            pred = self.ai_model(last_price).item()
            current = self.prices['BTC']
            if pred > current * 1.01:
                self.buy_coin_ai('BTC', 0.001)
            elif pred < current * 0.99:
                self.sell_coin_ai('BTC', 0.001)
            time.sleep(60)

    def buy_coin_ai(self, coin, amount):
        price = self.prices.get(coin, 0)
        cost = amount * price
        if self.wallet['USD'] >= cost:
            self.wallet['USD'] -= cost
            self.wallet[coin] = self.wallet.get(coin, 0) + amount
            print(f"AI Bought {amount} {coin}")

    def sell_coin_ai(self, coin, amount):
        if self.wallet.get(coin, 0) >= amount:
            price = self.prices.get(coin, 0)
            revenue = amount * price
            self.wallet[coin] -= amount
            self.wallet['USD'] += revenue
            print(f"AI Sold {amount} {coin}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CyberdudebivashCryptoPlatform(root)
    root.mainloop()