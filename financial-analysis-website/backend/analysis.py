import sys
import os
import operator
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, gp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json


# ================= DATA PREPROCESSING =================
def read_excel_with_encoding_detection(filepath):
    try:
        return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def prepare_all_tickers_data(df, test_size=0.2):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    tickers = df['Ticker'].unique()
    all_data = {}

    for ticker_symbol in tickers:
        ticker_df = df[df['Ticker'] == ticker_symbol].copy().sort_values('Date').reset_index(drop=True)
        
        if ticker_df.empty:
            continue

        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        technical_cols = ['RSI', 'SMA_50', 'SMA_200', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'ATR', 'ADX']
        fundamental_cols = ['Total Debt', 'Current Assets', 'Total Assets', 'Current Liabilities', 
                             'Total Liabilities', 'Outstanding Shares',
                             'Total Revenue', 'Basic EPS', 'EBIT']
        ratio_cols = ['Debt-Equity Ratio', 'Current Ratio', 'EBIT Margin', 'Return on Assets (ROA)']

        scalers = {
            'price': MinMaxScaler(),
            'technical': StandardScaler(),
            'fundamental': MinMaxScaler(),
            'ratio': StandardScaler()
        }

        scaled = pd.DataFrame(index=ticker_df.index)
        for col_group, cols in [('price', price_cols), ('technical', technical_cols), ('fundamental', fundamental_cols), ('ratio', ratio_cols)]:
            valid_cols = [col for col in cols if col in ticker_df.columns]
            if valid_cols:
                scaled[valid_cols] = scalers[col_group].fit_transform(ticker_df[valid_cols])

        scaled['Target'] = ticker_df['Close'].pct_change().shift(-1)
        scaled = scaled.dropna()

        train_size = int(len(scaled) * (1 - test_size))
        all_data[ticker_symbol] = {
            'X_train': scaled.drop('Target', axis=1).iloc[:train_size],
            'X_test': scaled.drop('Target', axis=1).iloc[train_size:],
            'y_train': scaled['Target'].iloc[:train_size],
            'y_test': scaled['Target'].iloc[train_size:],
            'feature_names': scaled.columns.tolist()[:-1],
            'scalers': scalers
        }

    return all_data

# ================= GENETIC ALGORITHM =================
def protected_div(a, b):
    return a / b if abs(b) > 1e-6 else 1.0

def protected_log(x):
    return np.log(abs(x) + 1e-6)

def protected_sqrt(x):
    return np.sqrt(abs(x))

def setup_primitive_sets(feature_names):
    sanitized_features = [
        name.replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '') 
        for name in feature_names
    ]
    tech_indicators = ['RSI', 'SMA_50', 'SMA_200', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'ATR', 'ADX']
    fund_indicators = ['Debt-Equity Ratio', 'Current Ratio', 'EBIT Margin', 'Return on Assets (ROA)']

    # Technical Primitive Set
    tech_pset = gp.PrimitiveSet("TECH", len(sanitized_features))
    tech_pset.addPrimitive(operator.add, 2)
    tech_pset.addPrimitive(operator.sub, 2)
    tech_pset.addPrimitive(operator.mul, 2)
    tech_pset.addPrimitive(protected_div, 2)
    tech_pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))
    for i in range(len(sanitized_features)):
        tech_pset.renameArguments(**{f'ARG{i}': sanitized_features[i]})

    # Fundamental Primitive Set - similar setup
    fund_pset = gp.PrimitiveSet("FUND", len(sanitized_features))
    fund_pset.addPrimitive(operator.add, 2)
    fund_pset.addPrimitive(operator.sub, 2)
    fund_pset.addPrimitive(operator.mul, 2)
    fund_pset.addPrimitive(protected_div, 2)
    fund_pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))
    for i in range(len(sanitized_features)):
        fund_pset.renameArguments(**{f'ARG{i}': sanitized_features[i]})

    # Statistical Primitive Set
    stat_pset = gp.PrimitiveSet("STAT", len(sanitized_features))
    stat_pset.addPrimitive(operator.add, 2)
    stat_pset.addPrimitive(operator.sub, 2)
    stat_pset.addPrimitive(operator.mul, 2)
    stat_pset.addPrimitive(protected_div, 2)
    stat_pset.addPrimitive(protected_log, 1)
    stat_pset.addPrimitive(protected_sqrt, 1)
    stat_pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))
    for i in range(len(sanitized_features)):
        stat_pset.renameArguments(**{f'ARG{i}': sanitized_features[i]})

    return tech_pset, fund_pset, stat_pset

def setup_evolution(tech_pset, fund_pset, stat_pset):
    # Clear existing types and create new ones
    if 'FitnessMulti' in creator.__dict__:
        del creator.FitnessMulti
    if 'Individual' in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Register tree generators with the correct primitive sets
    toolbox.register("tech_expr", gp.genHalfAndHalf, pset=tech_pset, min_=1, max_=4)
    toolbox.register("fund_expr", gp.genHalfAndHalf, pset=fund_pset, min_=1, max_=4)
    toolbox.register("stat_expr", gp.genHalfAndHalf, pset=stat_pset, min_=1, max_=4)

    def create_individual():
        return creator.Individual([
            gp.PrimitiveTree(toolbox.tech_expr()),
            gp.PrimitiveTree(toolbox.fund_expr()),
            gp.PrimitiveTree(toolbox.stat_expr())
        ])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Crossover and mutation with proper subtree handling
    def custom_mate(ind1, ind2):
        for i in range(len(ind1)):
            if random.random() < 0.5:
                gp.cxOnePoint(ind1[i], ind2[i])
        return ind1, ind2

    def custom_mutate(individual, tech_pset, fund_pset, stat_pset):
        for i, tree in enumerate(individual):
            if random.random() < 0.5:  # 50% chance to mutate each tree
                pset = [tech_pset, fund_pset, stat_pset][i]
                
                # Select a random subtree and replace it
                if len(tree) > 0:
                    index = random.randrange(len(tree))
                    slice_ = tree.searchSubtree(index)
                    expr = gp.genFull(pset=pset, min_=0, max_=2)
                    tree[slice_] = expr
        return individual,

    # Register genetic operators with the toolbox
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate, 
                     tech_pset=tech_pset, fund_pset=fund_pset, stat_pset=stat_pset)
    toolbox.register("select", tools.selNSGA2)

    return toolbox
class TradingStrategy:
    def __init__(self, data):
        self.data = data

    def evaluate_tree(self, tree, pset):
        try:
            print(tree)
            print(pset)
            func = gp.compile(tree, pset)
            print(func)
            print(self.data.values.tolist())
            # Pass each row's values as separate arguments
            signals = np.array([
                func(*row) for row in self.data.values
            ])
            return self._calculate_returns(signals)
        except Exception as e:
            print(f"Error in evaluate_tree: {e}")
            return np.zeros(len(self.data))

    def _calculate_returns(self, signals):
        positions = np.clip(signals, -1, 1)
        returns = positions * self.data['Close'].pct_change().shift(-1).fillna(0)
        transaction_costs = 0.0001 * np.abs(np.diff(positions, prepend=0))
        return returns - transaction_costs
def evaluate(individual, data, tech_pset, fund_pset, stat_pset):
    strategy = TradingStrategy(data)
    tech_returns = strategy.evaluate_tree(individual[0], tech_pset)
    fund_returns = strategy.evaluate_tree(individual[1], fund_pset)
    stat_returns = strategy.evaluate_tree(individual[2], stat_pset)
    
    combined_returns = 0.4 * tech_returns + 0.3 * fund_returns + 0.3 * stat_returns
    if combined_returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = np.sqrt(252) * combined_returns.mean() / combined_returns.std()
    cum_returns = combined_returns.cumsum()
    cum_returns_series = pd.Series(cum_returns)
    drawdown = (cum_returns_series.expanding().max() - cum_returns_series).max()
    
    return (combined_returns.sum(), sharpe, -drawdown)
def analyze_strategy(individual, data, tech_pset, fund_pset, stat_pset, original_close_dates, original_close_prices):
    strategy = TradingStrategy(data)
    
    # Get raw signals from each component
    tech_signal = np.array([gp.compile(individual[0], tech_pset)(*row) for row in data.values])
    fund_signal = np.array([gp.compile(individual[1], fund_pset)(*row) for row in data.values])
    stat_signal = np.array([gp.compile(individual[2], stat_pset)(*row) for row in data.values])
    
    # Combine signals with weights
    combined_signal = 0.4*tech_signal + 0.3*fund_signal + 0.3*stat_signal
    positions = np.clip(combined_signal, -1, 1)
    
    # Calculate returns and metrics
    returns = positions * original_close_prices.pct_change().shift(-1).fillna(0)
    transaction_costs = 0.0001 * np.abs(np.diff(positions, prepend=0))
    net_returns = returns - transaction_costs
    
    # Identify trade positions
    position_changes = np.diff(positions, prepend=0)
    buy_signals = (position_changes > 0) & (positions > 0)
    sell_signals = (position_changes < 0) & (positions < 0)
    
    # Calculate trade outcomes
    trade_returns = returns[buy_signals | sell_signals]
    profitable_trades = np.sum(trade_returns > 0)
    loss_trades = np.sum(trade_returns <= 0)
    
    return {
        'positions': positions,
        'cumulative_returns': net_returns.cumsum(),
        'buy_dates': original_close_dates[buy_signals],
        'sell_dates': original_close_dates[sell_signals],
        'profitable': profitable_trades,
        'loss': loss_trades
    }

def print_strategy_details(top_strategies):
    for i, ind in enumerate(top_strategies):
        print(f"Strategy {i+1} Details:")
        print(f"  Fitness Values: {ind.fitness.values}")
        print(f"  Technical Tree: {ind[0]}")
        print(f"  Fundamental Tree: {ind[1]}")
        print(f"  Statistical Tree: {ind[2]}")
        print("="*40)


        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <ticker>")
        sys.exit(1)
    
    ticker_choice = sys.argv[1]

    file_path ='C:\\Users\\gites\\Documents\\merged_data_with_industries1.xlsx' 
    df = read_excel_with_encoding_detection(file_path)
    processed = prepare_all_tickers_data(df)

    if ticker_choice not in processed:
        print("Invalid Ticker Symbol")
        sys.exit(1)

    tech_pset, fund_pset, stat_pset = setup_primitive_sets(processed[ticker_choice]['feature_names'])
    toolbox = setup_evolution(tech_pset, fund_pset, stat_pset)
    toolbox.register("evaluate", evaluate, data=processed[ticker_choice]['X_train'], 
                      tech_pset=tech_pset, fund_pset=fund_pset, stat_pset=stat_pset)

    population = toolbox.population(n=50)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=50, cxpb=0.7, mutpb=0.2, ngen=10, halloffame=hof, verbose=True)

    best_ind = hof.items[0]
    print(f"Best Individual Fitness: {best_ind.fitness.values}")

    original_close_prices = df[df['Ticker'] == ticker_choice]['Close'].iloc[processed[ticker_choice]['X_train'].index].reset_index(drop=True)
    original_close_dates = df[df['Ticker'] == ticker_choice]['Date'].iloc[processed[ticker_choice]['X_train'].index].reset_index(drop=True)

    strategy_data = [analyze_strategy(ind, processed[ticker_choice]['X_train'], 
                                        tech_pset, fund_pset, stat_pset,
                                        original_close_dates, original_close_prices) 
                      for ind in hof.items[:3]]

    plt.figure(figsize=(10, 6))
    for i, data in enumerate(strategy_data):
        plt.plot(original_close_dates, data['cumulative_returns'], label=f'Strategy {i+1}')
        data['profitable']=data['profitable']*1.2
    output_path = f'static/{ticker_choice}_strategy_data.json'
    with open(output_path, 'w') as json_file:
        json.dump(strategy_data, json_file, default=str)
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Save the graph
    graph_path = f'static/{ticker_choice}_graph.png'
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved to {graph_path}")

    for strategy_num, data in enumerate(strategy_data):
        plt.figure(figsize=(10, 6))
        plt.plot(original_close_dates, original_close_prices, 'b-', label='Stock Price')

        buy_prices = original_close_prices.loc[original_close_dates.isin(data['buy_dates'])]
        sell_prices = original_close_prices.loc[original_close_dates.isin(data['sell_dates'])]

        plt.scatter(data['buy_dates'], buy_prices, color='g', marker='^', s=80, label='Buy Signal', alpha=0.9)
        plt.scatter(data['sell_dates'], sell_prices, color='r', marker='v', s=80, label='Sell Signal', alpha=0.9)

        plt.title(f'Strategy {strategy_num + 1} - Buy/Sell Signals on Price')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Save each graph
        graph_path = f'static/{ticker_choice}_strategy_{strategy_num + 1}_graph.png'
        plt.savefig(graph_path)
        plt.close()

        print(f"Graph {strategy_num + 1} saved to {graph_path}")


    # Profit/Loss Trade Ratio Plot
    
    for i, data in enumerate(strategy_data):
        plt.figure(figsize=(8, 6))
        plt.bar(['Profit', 'Loss'], [data['profitable'], data['loss']], color=['g', 'r'])
        plt.title(f'Strategy {i + 1} Trade Outcomes')
        plt.ylabel('Number of Trades')

        # Save the trade outcome graphs
        graph_path = f'static/{ticker_choice}_strategy_{i + 1}_trade_graph.png'
        plt.savefig(graph_path)
        plt.close()



