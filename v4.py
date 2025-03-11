import numpy as np
import pandas as pd
import random
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import os

# Technical Indicator Functions
def calculate_ma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Moving Average"""
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=window, adjust=False).mean()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    fast_ema = calculate_ema(prices, fast)
    slow_ema = calculate_ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal)
    return macd_line - signal_line

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    ma = calculate_ma(prices, window)
    std = prices.rolling(window=window).std()
    return {
        'upper': ma + (std * num_std),
        'middle': ma,
        'lower': ma - (std * num_std)
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()
class StrategyType(Enum):
    TECHNICAL = "TA"
    FUNDAMENTAL = "FA"
    SENTIMENT = "SA"
    HYBRID = "HYBRID"

@dataclass
class Node:
    """Enhanced tree node class supporting strategy composition"""
    value: Union[str, float, int]
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    strategy_type: Optional[StrategyType] = None
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        try:
            if isinstance(self.value, str):
                if self.value in data.columns:
                    return data[self.value]
                
                left_val = self.left.evaluate(data) if self.left else pd.Series(0, index=data.index)
                right_val = self.right.evaluate(data) if self.right else pd.Series(0, index=data.index)
                
                if isinstance(left_val, np.ndarray):
                    left_val = pd.Series(left_val, index=data.index)
                if isinstance(right_val, np.ndarray):
                    right_val = pd.Series(right_val, index=data.index)
                
                ops = {
                    '+': lambda x, y: x + y,
                    '-': lambda x, y: x - y,
                    '*': lambda x, y: x * y,
                    '/': lambda x, y: x / (y + 1e-6),
                    'AND': lambda x, y: pd.Series(np.where((x > 0) & (y > 0), 1, -1), index=data.index),
                    'OR': lambda x, y: pd.Series(np.where((x > 0) | (y > 0), 1, -1), index=data.index),
                    'CROSS_ABOVE': lambda x, y: pd.Series(np.where((x.shift(1) <= y.shift(1)) & (x > y), 1, 0), index=data.index),
                    'CROSS_BELOW': lambda x, y: pd.Series(np.where((x.shift(1) >= y.shift(1)) & (x < y), 1, 0), index=data.index)
                }
                
                if self.value in ops:
                    return ops[self.value](left_val, right_val)
            
            return pd.Series(float(self.value), index=data.index)
            
        except Exception as e:
            print(f"Error in node evaluation: {e}")
            return pd.Series(0, index=data.index)
class StrategyStorage:
    """Class to manage storage and retrieval of trading strategies"""
    
    def __init__(self, storage_dir: str = "trading_strategies"):
        self.storage_dir = storage_dir
        self.known_strategies_file = os.path.join(storage_dir, "known_strategies.json")
        self.generated_strategies_file = os.path.join(storage_dir, "generated_strategies.json")
        self.excel_file = "trading_strategies.xlsx"  # Now saved in root directory
        self.initialize_storage()
    
    def initialize_storage(self):
        """Create storage directory and files if they don't exist"""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize known strategies file if it doesn't exist
        if not os.path.exists(self.known_strategies_file):
            self.save_known_strategies({
                "technical": {},
                "fundamental": {},
                "sentiment": {}
            })
        
        # Initialize generated strategies file if it doesn't exist
        if not os.path.exists(self.generated_strategies_file):
            self.save_generated_strategies({
                "strategies": [],
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_strategies": 0
                }
            })
    
    def strategy_to_dict(self, node: Node) -> Dict[str, Any]:
        """Convert strategy tree to dictionary format"""
        if node is None:
            return None
        
        return {
            "value": str(node.value),
            "strategy_type": node.strategy_type.value if node.strategy_type else None,
            "left": self.strategy_to_dict(node.left),
            "right": self.strategy_to_dict(node.right)
        }
    
    def dict_to_strategy(self, data: Dict[str, Any]) -> Optional[Node]:
        """Convert dictionary format back to strategy tree"""
        if data is None:
            return None
        
        strategy_type = StrategyType(data["strategy_type"]) if data["strategy_type"] else None
        
        # Convert value to appropriate type
        try:
            value = float(data["value"])
        except ValueError:
            value = data["value"]
        
        return Node(
            value=value,
            left=self.dict_to_strategy(data["left"]),
            right=self.dict_to_strategy(data["right"]),
            strategy_type=strategy_type
        )
    
    def save_known_strategies(self, strategies: Dict[str, Dict[str, Any]]):
        """Save known strategies to file"""
        with open(self.known_strategies_file, 'w') as f:
            json.dump(strategies, f, indent=2)
    
    def save_generated_strategies(self, strategies: Dict[str, Any]):
        """Save generated strategies to file"""
        with open(self.generated_strategies_file, 'w') as f:
            json.dump(strategies, f, indent=2)
    
    def load_known_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load known strategies from file"""
        with open(self.known_strategies_file, 'r') as f:
            return json.load(f)
    
    def load_generated_strategies(self) -> Dict[str, Any]:
        """Load generated strategies from file"""
        with open(self.generated_strategies_file, 'r') as f:
            return json.load(f)
    
    def add_known_strategy(self, name: str, strategy: Node, category: str, description: str = ""):
        """Add a new known strategy"""
        strategies = self.load_known_strategies()
        
        strategies[category][name] = {
            "strategy": self.strategy_to_dict(strategy),
            "description": description,
            "added_date": datetime.now().isoformat()
        }
        
        self.save_known_strategies(strategies)
    
    def add_generated_strategy(self, strategy: Node, fitness: float, 
                             performance_metrics: Dict[str, float]):
        """Add a new generated strategy"""
        strategies = self.load_generated_strategies()
        
        strategy_data = {
            "strategy": self.strategy_to_dict(strategy),
            "fitness": fitness,
            "performance_metrics": performance_metrics,
            "generation_date": datetime.now().isoformat()
        }
        
        strategies["strategies"].append(strategy_data)
        strategies["metadata"]["total_strategies"] += 1
        strategies["metadata"]["last_updated"] = datetime.now().isoformat()
        
        self.save_generated_strategies(strategies)
    
    def get_best_generated_strategies(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the top N generated strategies by fitness"""
        strategies = self.load_generated_strategies()
        sorted_strategies = sorted(
            strategies["strategies"],
            key=lambda x: x["fitness"],
            reverse=True
        )
        return sorted_strategies[:n]
    
    def export_to_excel(self):
        """Export all strategies to an Excel file with multiple sheets"""
        # Create Excel writer
        with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
            # Export known strategies
            known_strats = self.load_known_strategies()
            known_data = []
            
            for category, strategies in known_strats.items():
                for name, strat_info in strategies.items():
                    known_data.append({
                        'Category': category,
                        'Name': name,
                        'Description': strat_info['description'],
                        'Added Date': strat_info['added_date'],
                        'Strategy Tree': json.dumps(strat_info['strategy'], indent=2)
                    })
            
            if known_data:
                known_df = pd.DataFrame(known_data)
                known_df.to_excel(writer, sheet_name='Known Strategies', index=False)
            
            # Export generated strategies
            generated_strats = self.load_generated_strategies()
            generated_data = []
            
            for strat in generated_strats['strategies']:
                metrics = strat['performance_metrics']
                generated_data.append({
                    'Generation Date': strat['generation_date'],
                    'Fitness Score': strat['fitness'],
                    'Cumulative Return (%)': metrics.get('cumulative_return', 'N/A'),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 'N/A'),
                    'Max Drawdown (%)': metrics.get('max_drawdown', 'N/A'),
                    'Strategy Tree': json.dumps(strat['strategy'], indent=2)
                })
            
            if generated_data:
                generated_df = pd.DataFrame(generated_data)
                generated_df.to_excel(writer, sheet_name='Generated Strategies', index=False)
            
            # Add metadata sheet
            metadata = generated_strats['metadata']
            metadata_df = pd.DataFrame([{
                'Last Updated': metadata['last_updated'],
                'Total Strategies': metadata['total_strategies']
            }])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = min(adjusted_width, 50)
class StrategyLibrary:
    """Library of predefined trading strategies"""
    
    @staticmethod
    def create_technical_strategies() -> List[Node]:
        """Create common technical analysis strategies"""
        strategies = []
        
        # Moving Average Crossover
        ma_cross = Node('CROSS_ABOVE',
            Node('ma_20'),
            Node('ma_50'),
            strategy_type=StrategyType.TECHNICAL
        )
        strategies.append(ma_cross)
        
        # RSI Oversold/Overbought
        rsi_strategy = Node('OR',
            Node('CROSS_ABOVE',
                Node('rsi'),
                Node(30)
            ),
            Node('CROSS_BELOW',
                Node('rsi'),
                Node(70)
            ),
            strategy_type=StrategyType.TECHNICAL
        )
        strategies.append(rsi_strategy)
        
        # MACD Strategy
        macd_strategy = Node('CROSS_ABOVE',
            Node('macd'),
            Node(0),
            strategy_type=StrategyType.TECHNICAL
        )
        strategies.append(macd_strategy)
        
        # Bollinger Band Strategy
        bb_strategy = Node('OR',
            Node('CROSS_ABOVE',
                Node('close'),
                Node('bb_lower')
            ),
            Node('CROSS_BELOW',
                Node('close'),
                Node('bb_upper')
            ),
            strategy_type=StrategyType.TECHNICAL
        )
        strategies.append(bb_strategy)
        
        return strategies
    
    @staticmethod
    def create_fundamental_strategies() -> List[Node]:
        """Create common fundamental analysis strategies"""
        strategies = []
        
        # P/E Ratio Strategy
        pe_strategy = Node('CROSS_BELOW',
            Node('pe_ratio'),
            Node(15),
            strategy_type=StrategyType.FUNDAMENTAL
        )
        strategies.append(pe_strategy)
        
        # Price to Book Strategy
        pb_strategy = Node('AND',
            Node('CROSS_BELOW',
                Node('pb_ratio'),
                Node(1.5)
            ),
            Node('CROSS_ABOVE',
                Node('profit_margin'),
                Node(0.15)
            ),
            strategy_type=StrategyType.FUNDAMENTAL
        )
        strategies.append(pb_strategy)
        
        return strategies
    
    @staticmethod
    def create_sentiment_strategies() -> List[Node]:
        """Create common sentiment analysis strategies"""
        strategies = []
        
        # News Sentiment Strategy
        news_strategy = Node('AND',
            Node('CROSS_ABOVE',
                Node('news_sentiment'),
                Node(0.6)
            ),
            Node('CROSS_ABOVE',
                Node('sentiment_volume'),
                Node('avg_sentiment_volume')
            ),
            strategy_type=StrategyType.SENTIMENT
        )
        strategies.append(news_strategy)
        
        # Social Media Sentiment
        social_strategy = Node('AND',
            Node('CROSS_ABOVE',
                Node('social_sentiment'),
                Node(0.7)
            ),
            Node('CROSS_ABOVE',
                Node('social_volume'),
                Node('avg_social_volume')
            ),
            strategy_type=StrategyType.SENTIMENT
        )
        strategies.append(social_strategy)
        
        return strategies
class AdvancedStrategyBuilder:
    def create_technical_strategy(self):
        """Create advanced technical analysis strategy"""
        return {
            "value": "AND",
            "strategy_type": "TA",
            "left": {
                # Trend following component
                "value": "CROSS_ABOVE",
                "strategy_type": "TA",
                "left": {
                    "value": "ema_20",
                    "strategy_type": None
                },
                "right": {
                    "value": "ema_50",
                    "strategy_type": None
                }
            },
            "right": {
                # Volume confirmation
                "value": "AND",
                "strategy_type": "TA",
                "left": {
                    "value": "volume_sma",
                    "strategy_type": None,
                    "left": {
                        "value": "20",
                        "strategy_type": None
                    }
                },
                "right": {
                    "value": "rsi",
                    "strategy_type": None,
                    "left": {
                        "value": "30",
                        "strategy_type": None
                    },
                    "right": {
                        "value": "70",
                        "strategy_type": None
                    }
                }
            }
        }

    def create_hybrid_strategy(self):
        """Create a hybrid strategy combining technical, fundamental, and sentiment analysis"""
        return {
            "value": "AND",
            "strategy_type": "TA",
            "left": {
                # Technical Component
                "value": "AND",
                "strategy_type": "TA",
                "left": {
                    "value": "CROSS_ABOVE",
                    "strategy_type": "TA",
                    "left": {
                        "value": "macd",
                        "strategy_type": None
                    },
                    "right": {
                        "value": "signal",
                        "strategy_type": None
                    }
                },
                "right": {
                    "value": "bollinger_breakout",
                    "strategy_type": None,
                    "period": "20",
                    "std": "2"
                }
            },
            "right": {
                # Fundamental and Sentiment Component
                "value": "AND",
                "strategy_type": "FA",
                "left": {
                    "value": "pe_ratio",
                    "strategy_type": "FA",
                    "threshold": "20"
                },
                "right": {
                    "value": "sentiment_score",
                    "strategy_type": "SA",
                    "threshold": "0.6"
                }
            }
        }

    def create_momentum_strategy(self):
        """Create a momentum-based strategy"""
        return {
            "value": "AND",
            "strategy_type": "TA",
            "left": {
                # Momentum indicators
                "value": "AND",
                "strategy_type": "TA",
                "left": {
                    "value": "rsi",
                    "strategy_type": None,
                    "period": "14",
                    "threshold": "30"
                },
                "right": {
                    "value": "macd_histogram",
                    "strategy_type": None,
                    "threshold": "0"
                }
            },
            "right": {
                # Volume and trend confirmation
                "value": "AND",
                "strategy_type": "TA",
                "left": {
                    "value": "volume_surge",
                    "strategy_type": None,
                    "threshold": "1.5"
                },
                "right": {
                    "value": "higher_highs",
                    "strategy_type": None,
                    "period": "20"
                }
            }
        }

def prepare_data(prices: pd.Series, volume: pd.Series, 
                high: pd.Series = None, low: pd.Series = None) -> pd.DataFrame:
    """Prepare feature data including TA, FA, and SA indicators"""
    high = high if high is not None else prices
    low = low if low is not None else prices
    
    data = pd.DataFrame({
        'close': prices,
        'volume': volume,
        'rsi': calculate_rsi(prices),
        'ma_20': calculate_ma(prices, 20),
        'ma_50': calculate_ma(prices, 50),
        'macd': calculate_macd(prices),
        'atr': calculate_atr(high, low, prices),
    })
    
    # Add Bollinger Bands
    bb = calculate_bollinger_bands(prices)
    data['bb_upper'] = bb['upper']
    data['bb_lower'] = bb['lower']
    
    # Add mock fundamental and sentiment data for testing
    data['pe_ratio'] = np.random.uniform(10, 30, len(prices))
    data['pb_ratio'] = np.random.uniform(0.5, 3, len(prices))
    data['profit_margin'] = np.random.uniform(0.05, 0.25, len(prices))
    data['news_sentiment'] = np.random.uniform(-1, 1, len(prices))
    data['social_sentiment'] = np.random.uniform(-1, 1, len(prices))
    data['sentiment_volume'] = np.random.uniform(0, 100, len(prices))
    data['social_volume'] = np.random.uniform(0, 100, len(prices))
    data['avg_sentiment_volume'] = 50
    data['avg_social_volume'] = 50
    
    return data.fillna(0)
def test_strategy():
    # Initialize the strategy
    strategy = HybridGeneticStrategy()

    # Access stored strategies
    storage = strategy.strategy_storage

    # Get best performing generated strategies
    best_strategies = storage.get_best_generated_strategies(n=10)

    # Add a new known strategy
    strategy.strategy_storage.export_to_excel()
    # Prepare data with all indicators
    np.random.seed(42)
    prices = pd.Series(np.random.randn(500).cumsum() + 100,
                       index=pd.date_range('2023-01-01', periods=500))
    volume = pd.Series(np.random.randint(100, 1000, 500),
                       index=prices.index)
    high_series = prices + np.random.rand(500) * 5  # Simulated high prices
    low_series = prices - np.random.rand(500) * 5   # Simulated low prices

    data = prepare_data(
        prices=prices,
        volume=volume,
        high=high_series,
        low=low_series
    )

    # Backtest each strategy
    results = {}
    for strat in best_strategies:
        performance = backtest_strategy(strategy=strat, data=data)
        results[strat.name] = performance

    # Evaluate and print performance
    for name, perf in results.items():
        print(f"Strategy: {name}, Performance: {perf}")

    return results

def backtest_strategy(strategy, data):
    """
    Simulate a backtest of the provided strategy on the given data.
    Returns performance metrics like total return or Sharpe ratio.
    """
    # Placeholder for backtest logic
    return np.random.rand()  # Simulated performance metric

# Example call
test_results = test_strategy()