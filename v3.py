import numpy as np
import pandas as pd
import random
from typing import List, Union, Optional, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime

# [Previous technical indicators and Node class remain the same]
from typing import List, Union, Optional
from dataclasses import dataclass

# Technical indicators for feature generation
def moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate moving average of price series."""
    return prices.rolling(window=window, min_periods=1).mean()

def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-6)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def exponential_ma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate exponential moving average."""
    return prices.ewm(span=window, adjust=False).mean()

def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    fast_ma = exponential_ma(prices, fast)
    slow_ma = exponential_ma(prices, slow)
    macd_line = fast_ma - slow_ma
    signal_line = exponential_ma(macd_line, signal)
    return macd_line - signal_line
@dataclass
class Node:
    """Tree node class with type hints and improved operator handling."""
    value: Union[str, float, int]
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate the node's value using the provided data."""
        try:
            if isinstance(self.value, str):
                if self.value in data.columns:  # Direct feature access
                    return data[self.value]
                
                # Operators
                left_val = self.left.evaluate(data) if self.left else pd.Series(0, index=data.index)
                right_val = self.right.evaluate(data) if self.right else pd.Series(0, index=data.index)
                
                # Ensure operands are pandas Series
                if isinstance(left_val, np.ndarray):
                    left_val = pd.Series(left_val, index=data.index)
                if isinstance(right_val, np.ndarray):
                    right_val = pd.Series(right_val, index=data.index)
                
                ops = {
                    '+': lambda x, y: x + y,
                    '-': lambda x, y: x - y,
                    '*': lambda x, y: x * y,
                    '/': lambda x, y: x / (y + 1e-6),
                    'IF': lambda x, y: pd.Series(
                        np.where(x > 0, y, -y),
                        index=data.index
                    )
                }
                
                if self.value in ops:
                    result = ops[self.value](left_val, right_val)
                    # Ensure result is a pandas Series
                    if isinstance(result, np.ndarray):
                        result = pd.Series(result, index=data.index)
                    return result
                
            # Constant value
            return pd.Series(float(self.value), index=data.index)
            
        except Exception as e:
            print(f"Error in node evaluation: {e}")
            return pd.Series(0, index=data.index)

class GeneticTradingStrategy:
    def __init__(self, 
                 max_depth: int = 4,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 5):
        """Initialize the genetic trading strategy with configurable parameters."""
        self.max_depth = max_depth
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.strategy_history = []  # Track best strategies
        
    # [Previous methods remain the same until evolve()]
    def generate_random_tree(self, depth: int, features: List[str]) -> Node:
        """Generate a random trading strategy tree."""
        if depth == 0 or (depth < self.max_depth and random.random() < 0.3):
            # Leaf node: either a feature or a constant
            if random.random() < 0.7:
                return Node(random.choice(features))
            else:
                return Node(random.uniform(-1, 1))
        
        # Internal node: operator
        operators = ['+', '-', '*', '/', 'IF']
        operator = random.choice(operators)
        left = self.generate_random_tree(depth - 1, features)
        right = self.generate_random_tree(depth - 1, features)
        
        return Node(operator, left, right)
    def calculate_fitness(self, strategy: Node, data: pd.DataFrame, 
                         returns: pd.Series) -> float:
        """Calculate the fitness (Sharpe ratio) of a strategy."""
        try:
            signals = np.sign(strategy.evaluate(data))
            pnl = (signals.shift(1) * returns).fillna(0)
            
            # Calculate annualized Sharpe ratio
            sharpe_ratio = np.sqrt(252) * (pnl.mean() / (pnl.std() + 1e-6))
            
            # Add penalty for complexity
            node_count = self._count_nodes(strategy)
            complexity_penalty = 0.01 * node_count
            
            return sharpe_ratio - complexity_penalty
            
        except Exception as e:
            print(f"Fitness calculation error: {e}")
            return -np.inf
    
    def _count_nodes(self, node: Node) -> int:
        """Count the number of nodes in a tree."""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def tournament_select(self, population: List[Node], 
                         fitness_scores: List[float]) -> Node:
        """Select a strategy using tournament selection."""
        tournament = random.sample(list(zip(population, fitness_scores)), 
                                 self.tournament_size)
        return max(tournament, key=lambda x: x[1])[0]
    
    def crossover(self, parent1: Node, parent2: Node) -> Node:
        """Perform crossover between two parent strategies."""
        if random.random() < 0.5:
            return self._copy_tree(parent1)
        
        # Find random crossover points
        p1_nodes = self._get_nodes(parent1)
        p2_nodes = self._get_nodes(parent2)
        
        if not p1_nodes or not p2_nodes:
            return self._copy_tree(parent1)
        
        new_tree = self._copy_tree(parent1)
        crossover_point = random.choice(self._get_nodes(new_tree))
        replacement = self._copy_tree(random.choice(p2_nodes))
        
        # Replace subtree
        if crossover_point.value == new_tree.value:
            new_tree = replacement
        else:
            self._replace_subtree(new_tree, crossover_point, replacement)
        
        return new_tree
    
    def _get_nodes(self, node: Node) -> List[Node]:
        """Get all nodes in a tree."""
        if node is None:
            return []
        return [node] + self._get_nodes(node.left) + self._get_nodes(node.right)
    
    def _copy_tree(self, node: Node) -> Optional[Node]:
        """Create a deep copy of a tree."""
        if node is None:
            return None
        return Node(node.value,
                   self._copy_tree(node.left),
                   self._copy_tree(node.right))
    
    def _replace_subtree(self, root: Node, old_node: Node, new_node: Node) -> None:
        """Replace a subtree in the strategy."""
        if root is None:
            return
        
        if root.left is old_node:
            root.left = new_node
        elif root.right is old_node:
            root.right = new_node
        else:
            self._replace_subtree(root.left, old_node, new_node)
            self._replace_subtree(root.right, old_node, new_node)
    
    def mutate(self, strategy: Node, features: List[str]) -> Node:
        """Mutate a strategy with a certain probability."""
        if random.random() < self.mutation_rate:
            nodes = self._get_nodes(strategy)
            if not nodes:
                return strategy
            
            mutation_point = random.choice(nodes)
            
            if random.random() < 0.5:
                # Change node value
                if isinstance(mutation_point.value, (int, float)):
                    mutation_point.value = random.uniform(-1, 1)
                elif mutation_point.value in features:
                    mutation_point.value = random.choice(features)
                else:
                    mutation_point.value = random.choice(['+', '-', '*', '/', 'IF'])
            else:
                # Replace subtree
                new_subtree = self.generate_random_tree(2, features)
                if mutation_point.value == strategy.value:
                    return new_subtree
                self._replace_subtree(strategy, mutation_point, new_subtree)
        
        return strategy
    def _strategy_to_string(self, node: Node) -> str:
        """Convert strategy tree to string representation."""
        if node is None:
            return ""
        if node.left is None and node.right is None:
            return str(node.value)
        left_str = self._strategy_to_string(node.left)
        right_str = self._strategy_to_string(node.right)
        return f"({node.value} {left_str} {right_str})"
    
    def evolve(self, data: pd.DataFrame, returns: pd.Series, 
               features: List[str]) -> tuple[Node, float]:
        """Evolve the population to find the best trading strategy."""
        # Initialize population
        population = [
            self.generate_random_tree(self.max_depth, features)
            for _ in range(self.population_size)
        ]
        
        best_strategy = None
        best_fitness = float('-inf')
        generation_history = []
        
        for generation in range(self.generations):
            # Calculate fitness for all strategies
            fitness_scores = [
                self.calculate_fitness(strategy, data, returns)
                for strategy in population
            ]
            
            # Store top 3 strategies for this generation
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_strategies = []
            for idx in sorted_indices[:3]:
                strategy = population[idx]
                fitness = fitness_scores[idx]
                signals = np.sign(strategy.evaluate(data))
                pnl = (signals.shift(1) * returns).fillna(0)
                cumulative_returns = (1 + pnl).cumprod()
                
                strategy_info = {
                    'generation': generation,
                    'strategy': self._strategy_to_string(strategy),
                    'fitness': fitness,
                    'cumulative_return': (cumulative_returns.iloc[-1] - 1) * 100,
                    'sharpe_ratio': np.sqrt(252) * (pnl.mean() / (pnl.std() + 1e-6)),
                    'strategy_object': strategy
                }
                top_strategies.append(strategy_info)
            
            generation_history.append(top_strategies)
            
            # Update best strategy
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_strategy = self._copy_tree(population[max_fitness_idx])
            
            # [Rest of the evolution logic remains the same]
            
        self.strategy_history = generation_history
        return best_strategy, best_fitness
    
    def save_strategy_history(self, filename: str = 'strategy_history.xlsx'):
        """Save strategy history to Excel file."""
        data = []
        for gen_strategies in self.strategy_history:
            for strategy in gen_strategies:
                data.append({
                    'Generation': strategy['generation'],
                    'Strategy': strategy['strategy'],
                    'Fitness': strategy['fitness'],
                    'Cumulative Return (%)': strategy['cumulative_return'],
                    'Sharpe Ratio': strategy['sharpe_ratio']
                })
        
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"Strategy history saved to {filename}")
    
    def visualize_top_strategies(self, data: pd.DataFrame, returns: pd.Series,
                               n_strategies: int = 3):
        """Visualize the performance of top strategies."""
        if not self.strategy_history:
            print("No strategy history available. Run evolve() first.")
            return
        
        # Get the last generation's top strategies
        top_strategies = self.strategy_history[-1][:n_strategies]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        for i, strategy_info in enumerate(top_strategies):
            strategy = strategy_info['strategy_object']
            signals = np.sign(strategy.evaluate(data))
            pnl = (signals.shift(1) * returns).fillna(0)
            cumulative_returns = (1 + pnl).cumprod()
            
            ax1.plot(cumulative_returns.index, cumulative_returns.values,
                    label=f"Strategy {i+1} (Return: {strategy_info['cumulative_return']:.1f}%)")
        
        ax1.set_title('Cumulative Returns of Top Strategies')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True)
        
        # Plot strategy signals
        ax2 = fig.add_subplot(gs[1, 0])
        for i, strategy_info in enumerate(top_strategies):
            strategy = strategy_info['strategy_object']
            signals = np.sign(strategy.evaluate(data))
            ax2.plot(signals.index[-100:], signals.values[-100:],
                    label=f"Strategy {i+1}", alpha=0.7)
        
        ax2.set_title('Strategy Signals (Last 100 Periods)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Signal')
        ax2.legend()
        ax2.grid(True)
        
        # Plot fitness history
        ax3 = fig.add_subplot(gs[1, 1])
        fitness_history = []
        for gen_strategies in self.strategy_history:
            fitness_history.append([s['fitness'] for s in gen_strategies])
        
        fitness_data = pd.DataFrame(fitness_history,
                                  columns=[f'Strategy {i+1}' for i in range(n_strategies)])
        
        for column in fitness_data.columns:
            ax3.plot(fitness_data.index, fitness_data[column],
                    label=column, alpha=0.7)
        
        ax3.set_title('Fitness Evolution')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Fitness')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'strategy_visualization_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
def prepare_data(prices: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Prepare feature data for strategy development."""
    data = pd.DataFrame({
        'close': prices,
        'volume': volume,
        'rsi': rsi(prices),
        'ma_20': moving_average(prices, 20),
        'ma_50': moving_average(prices, 50),
        'macd': macd(prices)
    })
    return data.fillna(0)
def test_strategy():
    """Test the genetic trading strategy implementation."""
    # Generate sample data
    np.random.seed(42)
    prices = pd.Series(np.random.randn(500).cumsum() + 100,
                      index=pd.date_range('2023-01-01', periods=500))
    volume = pd.Series(np.random.randint(100, 1000, 500),
                      index=prices.index)
    
    # Prepare features
    data = prepare_data(prices, volume)
    returns = prices.pct_change().fillna(0)
    
    # Initialize and run genetic algorithm
    features = list(data.columns)
    genetic_strategy = GeneticTradingStrategy(
        max_depth=4,
        population_size=50,
        generations=20,
        mutation_rate=0.1
    )
    
    best_strategy, best_fitness = genetic_strategy.evolve(data, returns, features)
    
    # Save strategy history to Excel
    genetic_strategy.save_strategy_history()
    
    # Visualize top strategies
    genetic_strategy.visualize_top_strategies(data, returns)
    
    # Print final results
    print(f"\nFinal Strategy Fitness: {best_fitness:.4f}")
    signals = np.sign(best_strategy.evaluate(data))
    pnl = (signals.shift(1) * returns).fillna(0)
    cumulative_returns = (1 + pnl).cumprod()
    print(f"Cumulative Return: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {np.sqrt(252) * (pnl.mean() / (pnl.std() + 1e-6)):.4f}")

if __name__ == "__main__":
    test_strategy()