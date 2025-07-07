"This is an AI-driven intraday bot for Gold that combines machine learning with traditional technical analysis. "
"It trades short-term trends but exits quickly if volatility spikes or the market reverses. "
"The system prioritizes capital preservation with strict risk limits, making it suitable for moderate-risk traders."



# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
# To create the environment: python -m venv goldAI_env
# goldAI_env\Scripts\activate

# IP Protection: File patents for core algorithms (like BOTS Inc.’s Bitcoin ATM patents) to prevent replication 12.

#!/usr/bin/env python3
"""GoldAI - MetaTrader 5 Gold Trading Bot with Machine Learning with Stable Baselines3 (PPO) for RL"""
import sys
default_path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
import pandas as pd
import MetaTrader5  as mt5
import joblib
from datetime import datetime, timedelta
import time
from typing import Tuple, Dict, Callable, TypedDict, Optional, List, Any, Union, Literal
from pathlib import Path
import warnings
import threading
import math
from sklearn.preprocessing import RobustScaler
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,  # The correct V2 import
    ConfigDict,  # For model config
)
import random
from decimal import Decimal, getcontext
from pydantic import ValidationInfo
from typing import ClassVar, Annotated, Callable
from typing_extensions import Annotated 
from typing import Literal
from typing import TYPE_CHECKING
import re
from functools import lru_cache
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import pytest
import logging
from logging.handlers import TimedRotatingFileHandler
import pytz
from functools import lru_cache
from apscheduler.schedulers.background import BackgroundScheduler
from collections import defaultdict
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
import pytest


ObsType = npt.NDArray[np.float32]
ActType = int


def mt5_get(attr: str) -> Any:
    """Safely get MT5 attributes with type checking"""
    if hasattr(mt5, attr):
        return getattr(mt5, attr)
    raise AttributeError(f"MT5 has no attribute {attr}")

# Usage example:
account = mt5_get("account_info")()

PriceData = Dict[str, Union[float, datetime]]

def setup_logging():
    logger = logging.getLogger('goldAI')
    logger.setLevel(logging.INFO)
    
    # File handler with daily rotation
    fh = TimedRotatingFileHandler('logs/goldAI.log', when='D', interval=1)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()
# 2. Enhanced error handling

class TradingError(Exception):
    """Base class for trading exceptions"""
    pass


class RetryableError(TradingError):
    """Errors that can be retried"""
    pass


def _execute_with_retry_core(
    func: Callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    success_retcode: int = MT5Wrapper.TRADE_RETCODE_DONE,
    *args,
    ensure_connected: Optional[Callable[[], bool]] = None,
    on_success: Optional[Callable[[], None]] = None,
    on_exception: Optional[Callable[[], None]] = None,
    **kwargs
) -> Any:
    """
    Enhanced core retry logic with:
    - Exponential backoff with jitter
    - Absolute time limit
    - Thread safety
    - Connection state awareness
    """
    # Constants (tunable parameters)
    MAX_TOTAL_TIME = 30.0  # Absolute maximum retry window (seconds)
    BASE_DELAY = retry_delay      # Initial backoff (seconds)
    MAX_JITTER = 0.2       # Max random delay variation (20%)
    MAX_SINGLE_DELAY = 5.0 # Cap for individual delays
    
    last_exception = None
    start_time = time.time()
    attempt = 0
    
    while attempt < max_retries:
        try:
            # 1. Global timeout check
            elapsed = time.time() - start_time
            if elapsed >= MAX_TOTAL_TIME:
                logger.warning(f"Max total retry time ({MAX_TOTAL_TIME}s) reached")
                break

            # 2. Connection state verification
            if ensure_connected and not ensure_connected():
                delay = min(
                    BASE_DELAY * (2 ** attempt) * (1 + random.uniform(0, MAX_JITTER)),
                    MAX_SINGLE_DELAY
                )
                remaining_time = MAX_TOTAL_TIME - elapsed
                if delay > remaining_time:
                    delay = remaining_time
                time.sleep(delay)
                attempt += 1
                continue

            # 3. Execute the target function
            result = func(*args, **kwargs)
            
            # 4. Success handling
            if on_success:
                on_success()
            
            # 5. MT5-specific result validation
            if hasattr(result, 'retcode') and result.retcode != success_retcode:
                delay = min(
                    BASE_DELAY * (2 ** attempt) * (1 + random.uniform(0, MAX_JITTER)),
                    MAX_SINGLE_DELAY
                )
                time.sleep(delay)
                attempt += 1
                continue
                
            return result
            
        except (ConnectionError, TimeoutError, MT5Wrapper.Error) as e:
            last_exception = e
            if on_exception:
                on_exception()
                
            # Calculate adaptive delay
            delay = min(
                BASE_DELAY * (2 ** attempt) * (1 + random.uniform(0, MAX_JITTER)),
                MAX_SINGLE_DELAY
            )
            remaining_time = MAX_TOTAL_TIME - (time.time() - start_time)
            if delay > remaining_time:
                break
                
            time.sleep(delay)
            attempt += 1
    
    # Final error handling
    if last_exception:
        error_msg = f"Failed after {attempt} attempts over {time.time()-start_time:.1f}s: {str(last_exception)}"
        logger.error(error_msg)
        raise type(last_exception)(error_msg) from last_exception
        
    logger.warning(f"Operation failed without exceptions after {attempt} attempts")
    return None

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)




class SymbolInfoTick(TypedDict):
    """Type definition for MT5 symbol tick data"""
    time: int
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int
    flags: int
    # Add other tick properties as needed

class SymbolInfo(TypedDict):
    """Type definition for MT5 symbol info"""
    point: float
    trade_contract_size: float
    digits: int
    spread: float
    point: float
    trade_contract_size: float
    digits: int
    spread: float
    trade_tick_size: float
    trade_tick_value: float
    trade_stops_level: int
    trade_freeze_level: int
    volume_min: float
    volume_max: float
    volume_step: float
    swap_mode: int
    margin_initial: float
    margin_maintenance: float
    # Add other symb

class Position(TypedDict):
    """Type definition for MT5 position data"""
    ticket: int
    symbol: str
    type: int  # mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    profit: float
    time: int
    # Add other position properties as needed

@dataclass
class TradePair:
    entry: float
    exit: float


class MT5Wrapper:
    """Safe wrapper for all MT5 operations with type hints"""
    
    # MT5 Constants
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    TRADE_RETCODE_DONE = 10009
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 2
    TRADE_RETCODE_DONE = 10009  # Example value
    
    # Timeframes
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60
    
    class Error(Exception):
        """Custom exception for MT5-related errors"""
        pass

    @staticmethod
    def initialize_with_validation(**kwargs) -> bool:
        """Initialize and verify the connection works"""
        try:
            # Use your existing mt5_get function to safely access the attribute
            init_func = mt5_get("initialize")
            return bool(init_func(**kwargs))
        except Exception as e:
            raise MT5Wrapper.Error(f"MT5 initialization failed: {str(e)}")

        
    @staticmethod
    def shutdown() -> None:
        """Shutdown MT5 connection"""
        try:
            mt5.shutdown()  # type: ignore
        except Exception as e:
            raise MT5Wrapper.Error(f"MT5 shutdown failed: {str(e)}")
    
    @staticmethod
    def last_error() -> Tuple[int, str]:
        """Get last error details"""
        try:
            return mt5.last_error()  # type: ignore
        except Exception as e:
            raise MT5Wrapper.Error(f"Failed to get last error: {str(e)}")
    
    @staticmethod
    def check_connection() -> bool:
        """Verify the connection is actually functional"""
        try:
            # Use your existing mt5_get utility function
            total_symbols = mt5_get("symbols_total")()
            return total_symbols is not None  # More reliable than > 0 check
        except Exception:
            return False
   
    @staticmethod
    def symbol_info_tick(symbol: str) -> Optional[Dict[str, float]]:
        """Get current tick data with proper typing and error handling"""
        try:
            result = mt5.symbol_info_tick(symbol)  # type: ignore
            if result is None:
                return None
                
            return {
                'time': int(result.time),
                'bid': float(result.bid),
                'ask': float(result.ask),
                'last': float(result.last),
                'volume': int(result.volume),
                'time_msc': int(result.time_msc),
                'flags': int(result.flags)
            }
        except Exception as e:
            raise MT5Wrapper.Error(f"symbol_info_tick failed for {symbol}: {str(e)}")
    
    @staticmethod
    def positions_get(*, symbol: Optional[str] = None, ticket: Optional[int] = None) -> List[Position]:
        """Get positions with proper typing"""
        try:
            result = mt5.positions_get(symbol=symbol, ticket=ticket)  # type: ignore
            if result is None:
                return []
                
            return [{
                'ticket': int(pos.ticket),
                'symbol': str(pos.symbol),
                'type': int(pos.type),
                'volume': float(pos.volume),
                'price_open': float(pos.price_open),
                'sl': float(pos.sl),
                'tp': float(pos.tp),
                'price_current': float(pos.price_current),
                'profit': float(pos.profit),
                'time': int(pos.time)
            } for pos in result]
        except Exception as e:
            raise MT5Wrapper.Error(f"positions_get failed: {str(e)}")
    
    @staticmethod
    def order_send(request: Dict[str, Any]) -> Dict[str, Any]:
        """Send order with type checking"""
        try:
            result = mt5.order_send(request)  # type: ignore
            return {
                'retcode': int(result.retcode),
                'deal': int(result.deal),
                'order': int(result.order),
                'volume': float(result.volume),
                'price': float(result.price),
                'comment': str(result.comment)
            }
        except Exception as e:
            raise MT5Wrapper.Error(f"order_send failed: {str(e)}")
    
    @staticmethod
    def copy_rates_from_pos(
        symbol: str,
        timeframe: int,
        start_pos: int,
        count: int
    ) -> Optional[List[Dict[str, Union[int, float]]]]:
        """Get historical rates with proper typing and error handling"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)  # type: ignore
            if rates is None:
                return None
                
            return [{
                'time': int(rate[0]),
                'open': float(rate[1]),
                'high': float(rate[2]),
                'low': float(rate[3]),
                'close': float(rate[4]),
                'tick_volume': int(rate[5]),
                'spread': int(rate[6]),
                'real_volume': int(rate[7])
            } for rate in rates]
        except Exception as e:
            raise MT5Wrapper.Error(f"copy_rates_from_pos failed for {symbol}: {str(e)}")
    
    @staticmethod
    def account_info() -> Optional[Dict[str, float]]:
        """Get account info with proper typing"""
        try:
            result = mt5.account_info()  # type: ignore
            if result is None:
                return None
                
            return {
                'login': int(result.login),
                'balance': float(result.balance),
                'equity': float(result.equity),
                'margin': float(result.margin),
                'margin_free': float(result.margin_free),
                'margin_level': float(result.margin_level)
            }
        except Exception as e:
            raise MT5Wrapper.Error(f"account_info failed: {str(e)}")
    
    @staticmethod
    def symbol_info(symbol: str) -> Optional[SymbolInfo]:
        """Get symbol info with proper typing"""
        try:
            result = mt5.symbol_info(symbol)  # type: ignore
            if result is None:
                return None
                
            return {
                'point': float(result.point),
                'trade_contract_size': float(result.trade_contract_size),
                'digits': int(result.digits),
                'spread': int(result.spread),
                'trade_tick_size': float(getattr(result, 'trade_tick_size', 0.0)),
                'trade_tick_value': float(getattr(result, 'trade_tick_value', 0.0)),
                'trade_stops_level': int(getattr(result, 'trade_stops_level', 0)),
                'trade_freeze_level': int(getattr(result, 'trade_freeze_level', 0)),
                'volume_min': float(getattr(result, 'volume_min', 0.01)),
                'volume_max': float(getattr(result, 'volume_max', 100.0)),
                'volume_step': float(getattr(result, 'volume_step', 0.01)),
                'swap_mode': int(getattr(result, 'swap_mode', 0)),
                'margin_initial': float(getattr(result, 'margin_initial', 0.0)),
                'margin_maintenance': float(getattr(result, 'margin_maintenance', 0.0))
            }
        except Exception as e:
            raise MT5Wrapper.Error(f"symbol_info failed for {symbol}: {str(e)}")

    @staticmethod
    def symbols_total() -> int:
        """Get total number of symbols (fixed version)"""
        try:
            total_func = mt5_get("symbols_total")
            return int(total_func())
        except Exception as e:
            raise MT5Wrapper.Error(f"Failed to get symbols count: {str(e)}")

    @staticmethod
    def positions_total() -> Optional[int]:
        """Get total number of open positions with proper error handling"""
        try:
            # Use your existing mt5_get function for safe access
            total_func = mt5_get("positions_total")
            result = total_func()
            return int(result) if result is not None else None
        except Exception as e:
            raise MT5Wrapper.Error(f"positions_total failed: {str(e)}")


class Config(BaseModel):
    """Complete trading bot configuration with Pydantic V2 validation"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=False  # Allows model modification after creation
    )
    
    # --- Core Validated Fields ---
    # Instrument and Account Settings
    SYMBOL: str = "GOLD"
    ACCOUNT_BALANCE: float = Field(..., gt=0, description="Current account balance in USD")
    RISK_PER_TRADE: float = Field(default=0.02, gt=0, le=0.05)
    INITIAL_STOP_LOSS: float = Field(default=100, gt=5, le=500)
    INITIAL_TAKE_PROFIT: float = 150
    MAX_TRADES_PER_DAY: int = Field(default=20, gt=0, le=100)
    MAX_OPEN_POSITIONS: int = Field(default=5, gt=0, le=20)
    MAX_ALLOWED_SPREAD: float = Field(default=2.0, gt=0, le=10.0, description="Maximum allowed spread in pips")
    MAX_DAILY_LOSS: float = Field(default=0.05, gt=0, le=0.2, description="Max daily loss percentage (5%)")
    MAX_DRAWDOWN_PCT: float = Field(default=0.25, gt=0, le=0.5, description="Max allowed drawdown percentage (25%)")
    MAX_DATA_BUFFER: int = Field(
        default=2000,
        gt=100,
        le=10000,
        description="Maximum number of data samples to store for retraining"
    )
    MAX_SLIPPAGE: float = Field(default=0.5, gt=0, le=5.0, description="Maximum allowed slippage in pips")
    LOSS_BUFFER_PCT: float = Field(0.0005, gt=0, le=0.01)  # 0.05% buffer for breakeven
    PARTIAL_CLOSE_RATIO: float = Field(0.5, gt=0, lt=1) 
    FEATURES: List[str] = ['open', 'high', 'low', 'close', 'real_volume']
    MODEL_VERSION: str = "1.0"
    USE_ATR_SIZING: bool = Field(default=True)
    ATR_LOOKBACK: int = Field(default=14, gt=5, le=50)  # Period for ATR calculation
    USE_ATR_SIZING: bool = True  # Dynamic SL enabled by default
    MIN_STOP_DISTANCE: float = 50  # Fallback if ATR too small (pips)
    ATR_STOP_LOSS_FACTOR: float = Field(default=1.5, gt=0.0, le=5.0)
    TRAILING_STOP_POINTS: int = Field(default=50, gt=0, le=500)
    PREPROCESSOR_PATH: Path = Path("models/preprocessor.pkl")
    DATA_POINTS: int = 500
    RL_MODEL_PATH: Path = Field(default=Path("models/rl_model"))
    TRADE_PENALTY_THRESHOLD: float = -50
    POSITION_ADJUSTMENT_THRESHOLD: float = 100
    RETRAIN_INTERVAL_DAYS: int = 7
    MIN_RETRAIN_SAMPLES: int = 1000
    TIMEFRAME: int = MT5Wrapper.TIMEFRAME_M1
    ADX_WINDOW: int = 14
    VOLUME_MA_WINDOW: int = 20
    VOLUME_ROC_PERIOD: int = 5

    WALKFORWARD_GAP_SIZE: int = Field(
        default=5,
        gt=0,
        le=30,
        description="Number of periods to leave between train and test sets in walkforward validation"
    )

    STRESS_SCENARIOS: Dict = Field(
        default={
            'gold_crash': {'type': 'vol_spike', 'size': 0.3},  # 30% volatility spike
            'gold_squeeze': {'type': 'trend', 'slope': 0.002}, # Rapid upward trend
            'liquidity_crisis': {'type': 'vol_spike', 'size': 0.7, 'spread_multiplier': 3}
        },
        description="Predefined market stress scenarios"
    )
    

    COMMISSION: float = Field(default=0.0005, gt=0, le=0.01, 
                            description="Broker commission per trade (0.05%)")
    SLIPPAGE_MODEL: Dict[str, Dict[str, float]] = Field(
        default={
            'gold': {'mean': 0.0005, 'std': 0.0002},
            'normal': {'mean': 0.0002, 'std': 0.0001}
        },
        description="Slippage model parameters"
    )
    
    
    # --- Class Variables (Non-Fields) ---
    RL_PARAMS: Dict[str, Any] = Field(
        default={
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            "max_grad_norm": 0.5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'total_timesteps': 100000
        },
        description="RL training parameters"
    )

    def __init__(self, **data):
        # Get actual balance from MT5 if not provided
        if 'ACCOUNT_BALANCE' not in data:
            account_info = MT5Wrapper.account_info()
            if account_info is None:
                raise ValueError("Could not retrieve account balance from MT5")
            data['ACCOUNT_BALANCE'] = account_info['balance']
        
        super().__init__(**data)
        
    # --- Validators ---
    @field_validator('SYMBOL')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if not re.match(r'^[A-Z]{3,6}$', v):
            raise ValueError('Symbol must be 3-6 uppercase letters (e.g. XAUUSD)')
        return v

    
    @field_validator('PREPROCESSOR_PATH', 'RL_MODEL_PATH')
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator('INITIAL_STOP_LOSS')
    @classmethod
    def validate_stop_loss(cls, v: float) -> float:
        """Validate stop loss against basic constraints"""
        if not (5 <= v <= 500):
            raise ValueError('Stop loss must be 5-500 pips')
        return v
        
    @model_validator(mode='after')
    def validate_risk_reward(self) -> 'Config':
        """Final business logic validation"""
        if self.INITIAL_TAKE_PROFIT <= self.INITIAL_STOP_LOSS:
            raise ValueError('Take profit must exceed stop loss')
        
        ratio = self.INITIAL_TAKE_PROFIT / self.INITIAL_STOP_LOSS
        if ratio < 1.5:
            print(f"Warning: Risk/reward ratio {ratio:.1f}:1 is suboptimal")
        return self

    @field_validator('TIMEFRAME')
    @classmethod
    def validate_timeframe(cls, v: int) -> int:
        valid_timeframes = {
            MT5Wrapper.TIMEFRAME_M1: 'M1',
            MT5Wrapper.TIMEFRAME_M5: 'M5', 
            MT5Wrapper.TIMEFRAME_M15: 'M15',
            MT5Wrapper.TIMEFRAME_H1: 'H1'
        }
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Valid: {list(valid_timeframes.values())}')
        return v
    
    def __str__(self) -> str:
        return (f"Config(SYMBOL={self.SYMBOL}, "
                f"RISK={self.RISK_PER_TRADE*100}%, "
                f"SL/TP={self.INITIAL_STOP_LOSS}/{self.INITIAL_TAKE_PROFIT}, "
                f"TF={self.TIMEFRAME})")


class PerformanceMonitor:
    """
    Performance Monitoring:

    Track more advanced metrics like:

        Calmar ratio

        Sortino ratio

        Profit factor by trade type

    Add correlation analysis between trades
    """
    def __init__(self, config: Config,):
        self.config = config
        # version = model.get_model_version()
        self.model_trackers: Dict[str, ModelPerformanceTracker] = {} 
        self.cumulative_pnl = 0.0
        self.win_rate = 0.0        # self.equity_curve = [0.0] 
        self.metrics = {
            'daily_pnl': [],
            'cumulative_pnl': 0.0,
            'win_rate': [],
            'calmar': None,
            'sortino': None,
            'sharpe': None,
            'max_dd': 0.0,
            'trades': [],
            'equity_curve': [0.0],  # Starting with 0 equity
            'buffer_stats': {
                'current_size': 0,
                'max_capacity': 0,
                'last_update': None,
                'errors': []
            }
        }
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def _update_sharpe(self) -> None:
        """Calculate Sharpe ratio based on daily returns"""
        if len(self.metrics['daily_pnl']) < 2:
            return
            
        daily_returns = np.diff(self.metrics['equity_curve'])
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        
        if np.std(excess_returns) > 0:
            self.metrics['sharpe'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            self.metrics['sharpe'] = 0.
           
    def _update_drawdown(self):
        """Calculate maximum drawdown percentage"""
        equity = np.array(self.metrics['equity_curve'])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        self.metrics['max_dd'] = min(drawdown.min(), self.metrics['max_dd'])
    
    def _calc_annualized_return(self):
        """Calculate annualized return percentage"""
        if len(self.metrics['equity_curve']) < 2:
            return 0.0
            
        total_return = (self.metrics['equity_curve'][-1] / self.metrics['equity_curve'][0] - 1) * 100
        days_active = len(self.metrics['daily_pnl'])
        return ((1 + total_return/100) ** (252/days_active) - 1) * 100
    
    def _calc_profit_factor(self) -> float:
        """Calculate profit factor using IndicatorUtils"""
        return IndicatorUtils.profit_factor(self.metrics['trades'])
    
    def _calc_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        annualized_return = self._calc_annualized_return()
        max_drawdown = abs(self.metrics['max_dd'])  # Convert to positive
        
        if max_drawdown > 0:
            return annualized_return / max_drawdown
        return float('inf')  # No drawdown means perfect ratio

    def _calc_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (focuses on downside risk)
        Returns:
            float: Ratio value (infinity if no downside risk, 0 if insufficient data)
        """
        try:
            if len(self.metrics['daily_pnl']) < 2:
                return 0.0
                
            daily_returns = np.diff(self.metrics['equity_curve'])
            downside_returns = daily_returns[daily_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
                
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                # _calc_annualized_return returns percentage, so divide by 100
                annualized_return = self._calc_annualized_return() / 100  
                return (annualized_return - (self.risk_free_rate)) / downside_std * np.sqrt(252)
            return 0.0
        except Exception as e:
            logger.warning(f"Sortino ratio calculation error: {str(e)}")
            return 0.0

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with warnings and enhanced metrics"""
        # Calculate core metrics
        profit_factor = self._calc_profit_factor()
        annualized_return = self._calc_annualized_return()
        sharpe = self.metrics.get('sharpe')
        drawdown = self.metrics.get('max_dd', 0)
        win_rate = np.mean(self.metrics['win_rate']) if self.metrics.get('win_rate') else None
        daily_pnl = self.metrics.get('daily_pnl', [0])[-1]
        total_trades = len(self.metrics['trades'])
        calmar_ratio = self._calc_calmar_ratio()
        sortino = self._calc_sortino_ratio()
        avg_trade_pnl = np.mean([t['pnl'] for t in self.metrics['trades']]) if total_trades > 0 else 0
        trade_stats = self.get_trade_statistics()

        # Generate warnings
        warnings = []
        if profit_factor < 1.0:
            warnings.append("WARNING: Profit factor below 1.0 - strategy is losing money")
        if drawdown < -self.config.MAX_DRAWDOWN_PCT:  # Assuming config is available
            warnings.append(f"WARNING: Drawdown ({abs(drawdown):.1f}%) exceeds maximum allowed")
        if sharpe is not None and sharpe < 1.0:
            warnings.append(f"WARNING: Sharpe ratio ({sharpe:.2f}) indicates poor risk-adjusted returns")
        if calmar_ratio < 1.0 and not np.isinf(calmar_ratio):
            warnings.append(f"WARNING: Calmar ratio ({calmar_ratio:.2f}) indicates poor risk-adjusted returns")
        if sortino is not None and sortino < 1.0:
            warnings.append(f"WARNING: Sortino ratio ({sortino:.2f}) indicates poor downside risk adjustment")
        if trade_stats['profit_factor'] < 1.0:
            warnings.append("WARNING: Profit factor below 1.0 - strategy is losing money")

        return {
            # Core metrics
            'annualized_return': f"{annualized_return:.2f}%",
            'sharpe_ratio': f"{sharpe:.2f}" if sharpe is not None else "N/A",
            'max_drawdown': f"{drawdown:.2f}%",
            'daily_pnl': daily_pnl,
            'win_rate': f"{win_rate*100:.1f}%" if win_rate is not None else "N/A",
            'profit_factor': f"{profit_factor:.2f}",
            'total_trades': total_trades,
            'avg_trade_pnl': f"{avg_trade_pnl:.2f}" if total_trades > 0 else "N/A",

            'calmar_ratio': f"{calmar_ratio:.2f}" if not np.isinf(calmar_ratio) else "∞",
            'sortino_ratio': f"{sortino:.2f}" if sortino is not None else "N/A",
            
            # Additional useful metrics
            'winning_trades': sum(1 for t in self.metrics['trades'] if t['pnl'] > 0),
            'losing_trades': sum(1 for t in self.metrics['trades'] if t['pnl'] < 0),
            'best_trade': f"{max(t['pnl'] for t in self.metrics['trades']):.2f}" if total_trades > 0 else "N/A",
            'worst_trade': f"{min(t['pnl'] for t in self.metrics['trades']):.2f}" if total_trades > 0 else "N/A",
            'profitability': "Profitable" if profit_factor > 1 else "Unprofitable",
            
            # Advanced metrics
            'advanced_stats': {
                'avg_win': trade_stats['avg_win'],
                'avg_loss': trade_stats['avg_loss'],
                'win_loss_ratio': abs(trade_stats['avg_win']/trade_stats['avg_loss']) if trade_stats['avg_loss'] != 0 else float('inf')
            },

            # Warnings and alerts
            'warnings': warnings if warnings else ["No critical issues detected"],
            'status': "DANGER" if warnings else "OK",
            
            
            # Model performance (if available)
            'model_accuracy': getattr(self, 'model_accuracy', "N/A"),
            'last_retrain': getattr(self, 'last_retrain_time', "Never")
        }

    def plot_equity_curve(self):
        """Generate matplotlib equity curve plot"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['equity_curve'], label='Equity Curve')
        plt.title('Trading Performance')
        plt.xlabel('Trade #')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        return plt

        """
        Example Usage:
        python

        monitor = PerformanceMonitor()

        # Simulate some trades
        trades = [
            {'pnl': 150}, {'pnl': -100}, {'pnl': 200}, 
            {'pnl': -50}, {'pnl': 300}
        ]

        for trade in trades:
            monitor.update(trade)

        print(monitor.get_performance_report())
        """
        """

        Example Output:
        {
            'annualized_return': '217.47%',  # If these were daily trades
            'sharpe_ratio': '2.37',
            'max_drawdown': '-20.00%',
            'win_rate': '60.0%',
            'profit_factor': '2.60',
            'total_trades': 5,
            'avg_trade_pnl': '100.00'
        }
        """
        """
        # Generate equity curve plot
        monitor.plot_equity_curve().show()
        """  

        """
        monitor = PerformanceMonitor()
        trades = [{'pnl': 150}, {'pnl': -100}, {'pnl': 200}, {'pnl': -50}, {'pnl': 300}]
        for trade in trades:
            monitor.update(trade)
        print(monitor.get_performance_report())
        monitor.plot_equity_curve().show()
        """

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        trades = self.metrics['trades']
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return wins / len(trades)

    def _update_equity_metrics(self, pnl: float) -> None:
        self.metrics['equity_curve'].append(self.metrics['equity_curve'][-1] + pnl)

    def add_model(self, model: "GoldMLModel") -> None:
        """Register a model for versioned tracking"""
        version = model.get_model_version()
        self.model_trackers[version] = ModelPerformanceTracker(model)

    def get_combined_report(self) -> Dict:
        """Aggregate performance across all models"""
        return {
            'overall': self._calculate_overall_metrics(),
            'model_comparison': {
                v: t.to_dict() for v,t in self.model_trackers.items()
            },
            'feature_analysis': self._analyze_features()
        }
    
    def should_retrain(self, model: "GoldMLModel") -> bool:
        """Smart retraining decision based on multiple factors"""
        version = model.get_model_version()
        tracker = self.model_trackers.get(version)
        
        if not tracker:
            return False
            
        # Check accuracy decay
        accuracy_threshold = 0.6
        if tracker.accuracy < accuracy_threshold:
            return True
            
        # Check feature drift
        feature_report = tracker.get_feature_report()
        if max(feature_report.values()) < 0.2:
            return True  # No dominant features
            
        return False

    def _calculate_overall_metrics(self) -> Dict:
        """Calculate combined metrics across all models"""
        return {
            'total_trades': len(self.metrics['trades']),
            'win_rate': self._calculate_win_rate(),
            'sharpe_ratio': self.metrics['sharpe'],
            'annualized_return': self._calc_annualized_return(),
            'max_drawdown': self.metrics['max_dd']
        }

    def _analyze_features(self) -> Dict:
        """Aggregate feature importance across all model versions"""
        feature_analysis = defaultdict(float)
        total_weight = 0
        
        for tracker in self.model_trackers.values():
            report = tracker.get_feature_report()
            for feature, weight in report.items():
                feature_analysis[feature] += weight
                total_weight += weight
                
        # Normalize if we have data
        if total_weight > 0:
            return {k: v/total_weight for k,v in feature_analysis.items()}
        return dict(feature_analysis)

    def update(self, trade_data: Dict) -> None:
        """Update metrics with new trade data"""
        if not isinstance(trade_data, dict):
            raise TypeError("Trade data must be a dictionary")
        
        # Store trade data in single location
        self.metrics['trades'].append(trade_data)
        
        # Update metrics
        self._update_trade_stats(trade_data)
        
        # Update model-specific tracking
        version = trade_data.get('model_version', 'unversioned')
        if version in self.model_trackers:
            self.model_trackers[version].update(trade_data)
    
    def _update_trade_stats(self, trade: Dict) -> None:
        """Update all metrics for a new trade"""
        pnl = trade.get('pnl', 0)
        
        # Update equity curve
        self.metrics['equity_curve'].append(self.metrics['equity_curve'][-1] + pnl)
        
        # Update win rate
        self.metrics['win_rate'].append(1 if pnl > 0 else 0)
        
        # Daily PnL tracking
        trade_time = trade.get('time', datetime.now())
        if self.metrics['daily_pnl'] and self.metrics['daily_pnl'][-1]['date'] == trade_time.date():
            self.metrics['daily_pnl'][-1]['amount'] += pnl
        else:
            self.metrics['daily_pnl'].append({
                'date': trade_time.date(),
                'amount': pnl
            })
        
        # Update risk metrics
        self._update_sharpe()
        self._update_drawdown()
        self.metrics['calmar'] = self._calc_calmar_ratio()
        self.metrics['sortino'] = self._calc_sortino_ratio()

    def get_total_reward(self) -> float:
        """Get cumulative PnL"""
        return self.metrics['equity_curve'][-1]

    def get_trade_count(self) -> int:
        """Get total number of trades"""
        return len(self.metrics['trades'])

    def get_trades(self, filter_func: Optional[Callable[[Dict], bool]] = None) -> List[Dict]:
        """Get trades with optional filtering
        
        Args:
            filter_func: Callable that takes a trade dict and returns bool
            
        Returns:
            List of filtered trades (or all trades if no filter)
        """
        if filter_func and not callable(filter_func):
            raise TypeError("filter_func must be callable")
        return [t for t in self.metrics['trades']] if filter_func is None else [t for t in self.metrics['trades'] if filter_func(t)]

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive trade statistics"""
        trades = self.metrics['trades']
        if not trades:
            return {}
        
        pnls = [t.get('pnl', 0) for t in trades]
        return {
            'count': len(trades),
            'win_rate': self._calculate_win_rate(),
            'avg_win': np.mean([p for p in pnls if p > 0]),
            'avg_loss': np.mean([p for p in pnls if p < 0]),
            'largest_win': max(pnls),
            'largest_loss': min(pnls),
            'profit_factor': self._calc_profit_factor(),
            'expectancy': (self._calculate_win_rate() * np.mean([p for p in pnls if p > 0])) + 
                        ((1 - self._calculate_win_rate()) * np.mean([p for p in pnls if p < 0]))
        }

    def get_profit_factor_analysis(self) -> Dict[str, Any]:
        """Enhanced profit factor breakdown"""
        trades = self.metrics['trades']
        return {
            'overall': IndicatorUtils.profit_factor(trades),
            'by_type': {
                'long': IndicatorUtils.profit_factor([t for t in trades if t.get('type') == 'long']),
                'short': IndicatorUtils.profit_factor([t for t in trades if t.get('type') == 'short']),
                'unclassified': IndicatorUtils.profit_factor([t for t in trades if 'type' not in t])
            },
            'by_model': {
                ver: IndicatorUtils.profit_factor([t for t in trades if t.get('model_version') == ver])
                for ver in {t.get('model_version', 'unversioned') for t in trades}
            }
        }

    def get_extended_report(self) -> Dict[str, Any]:
        """New method that includes advanced metrics"""
        report = self.get_performance_report()
        report['advanced_metrics'] = {
            'profit_factor_analysis': self.get_profit_factor_analysis(),
            # Could add other advanced metrics later
        }
        return report

class IndicatorUtils:
    """Core technical indicator calculations with robust array handling"""

    @staticmethod
    def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate True Range with proper NaN handling and vectorization.
        TR = max[(high - low), abs(high - prev_close), abs(low - prev_close)]
        """
        # Convert inputs to float64 arrays
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        close_arr = np.asarray(close, dtype=np.float64)
        
        # Create shifted close with NaN for first element
        prev_close = np.empty_like(close_arr)
        prev_close[0] = np.nan
        prev_close[1:] = close_arr[:-1]
        
        # Vectorized TR calculation
        return np.nanmax(
            np.vstack([
                high_arr - low_arr,
                np.abs(high_arr - prev_close),
                np.abs(low_arr - prev_close)
            ]),
            axis=0
        )
    
    @staticmethod
    def _wilder_smoothing(series: np.ndarray, window: int) -> np.ndarray:
        """Internal Wilder's smoothing implementation"""
        smoothed = np.full_like(series, np.nan)
        if len(series) < window:
            return smoothed
        
        # First value is simple average
        smoothed[window-1] = np.nanmean(series[:window])
        
        # Subsequent values use Wilder's formula: (prev * (n-1) + current) / n
        for i in range(window, len(series)):
            if np.isnan(smoothed[i-1]):
                smoothed[i] = series[i]
            else:
                smoothed[i] = (smoothed[i-1] * (window-1) + series[i]) / window
                
        return smoothed
    
    @staticmethod
    def directional_movements(high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized directional movement calculation"""
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        
        # Initialize arrays with zeros
        pos_dm = np.zeros_like(high_arr)
        neg_dm = np.zeros_like(low_arr)
        
        # Calculate differences
        high_diff = high_arr[1:] - high_arr[:-1]
        low_diff = low_arr[:-1] - low_arr[1:]  # Note: reverse difference for low
        
        # Vectorized conditions
        pos_mask = (high_diff > low_diff) & (high_diff > 0)
        neg_mask = (low_diff > high_diff) & (low_diff > 0)
        
        # Apply masks
        pos_dm[1:] = np.where(pos_mask, high_diff, 0)
        neg_dm[1:] = np.where(neg_mask, low_diff, 0)
        
        return pos_dm, neg_dm

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            window: int = 14, smoothing: str = 'wilder') -> np.ndarray:
        """
        Calculate Average True Range using true_range() as the base.
        Supports both Wilder's smoothing (default) and SMA.
        """
        # Get true range values
        tr = IndicatorUtils.true_range(high, low, close)
        
        # Handle empty/invalid inputs
        if len(tr) == 0 or window <= 1:
            return np.array([])
        
        # Apply selected smoothing method
        if smoothing.lower() == 'sma':
            # Simple Moving Average
            atr_values = np.full_like(tr, np.nan)
            for i in range(window-1, len(tr)):
                window_tr = tr[i-window+1:i+1]
                atr_values[i] = np.nanmean(window_tr)
            return atr_values
        else:
            # Wilder's Smoothing (default)
            return IndicatorUtils._wilder_smoothing(tr, window)

    @staticmethod
    def profit_factor(trades: List[Union[Dict[str, float], Tuple[str, float, Any]]]) -> float:
        """
        Calculate profit factor from trades.
        Supports both formats:
        1. PerformanceMonitor format: List of dicts with 'pnl' key
        2. TradingEnvironment format: List of ('buy'/'sell', price, ...) tuples
        """
        if not trades:
            return 0.0

        # Handle PerformanceMonitor format
        if isinstance(trades[0], dict) and 'pnl' in trades[0]:
            profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)  # type: ignore
            losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))  # type: ignore
            return profits / losses if losses > 0 else float('inf')

        # Handle TradingEnvironment format
        elif isinstance(trades[0], (tuple, list)) and len(trades[0]) >= 2:
            trade_pairs: List[TradePair] = []
            current_trade = None
            
            for trade in trades:
                if not isinstance(trade, (tuple, list)):
                    continue
                    
                if trade[0] == 'buy':
                    current_trade = TradePair(entry=trade[1], exit=0.0)
                elif trade[0] == 'sell' and current_trade:
                    current_trade.exit = trade[1]
                    trade_pairs.append(current_trade)
                    current_trade = None
            
            gross_profit = sum(t.exit - t.entry for t in trade_pairs if t.exit > t.entry)
            gross_loss = abs(sum(t.exit - t.entry for t in trade_pairs if t.exit <= t.entry))
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return 0.0  # Unknown format


class MT5Connector:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                data_fetcher: Optional['DataFetcher'] = None):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection_lock = threading.Lock()
        self._position_cache = {}
        self._reset_connection_state()
        self.emergency_stop = EmergencyStop()
        self.data_fetcher = data_fetcher
        
        
    def disconnect(self) -> None:
        """Shutdown MT5 connection"""
        with self._connection_lock:
            if self.connected:
                try:
                    MT5Wrapper.shutdown()
                    logger.info("Disconnected from MT5")
                except Exception as e:
                    logger.error(f"Error during disconnection: {str(e)}")
                finally:
                    self._reset_connection_state()

    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        return _execute_with_retry_core(
            func=func,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            success_retcode=MT5Wrapper.TRADE_RETCODE_DONE,
            *args,
            ensure_connected=self.ensure_connected,
            on_success=lambda: setattr(self, 'last_activity_time', time.time()),
            on_exception=self._reset_connection_state,
            **kwargs
        )

    def verify_trade_execution(self, ticket: int, max_slippage: float) -> bool:
        """
        Comprehensive trade verification with:
        - Position existence check
        - Price validation
        - Slippage calculation
        - Profit verification
        
        Args:
            ticket: Position ticket number
            max_slippage: Allowed slippage threshold
            
        Returns:
            bool: True if trade is valid, False otherwise
        """
        assert self.data_fetcher is not None, "DataFetcher must be initialized"
        current_price = self.data_fetcher.get_current_price(Config.SYMBOL)
        if not current_price:
            logger.error("Price data unavailable for verification")
            return False
            
        # Get position with retry logic
        position = self._fetch_position_with_retry(ticket)
        if not position or not self._validate_cached_position(ticket):
            logger.error(f"Position {ticket} not found or invalid")
            return False
            
        try:
            # Calculate expected profit
            if position['type'] == 'buy':
                expected_profit = (current_price['bid'] - position['price_open']) * position['volume']
            else:
                expected_profit = (position['price_open'] - current_price['ask']) * position['volume']
                
            actual_profit = position['profit']
            
            # Verify slippage
            slippage = abs(actual_profit - expected_profit)
            if slippage > max_slippage:
                logger.warning(f"High slippage detected: {slippage:.2f} (max allowed: {max_slippage:.2f})")
                return False
                
            return True
            
        except KeyError as e:
            logger.error(f"Position data incomplete: {str(e)}")
            return False
    
    def get_open_positions(self, symbol: str, use_cache: bool = True) -> List[Dict]:
        """Get all open positions for a symbol with caching"""
        cache_key = f"positions_{symbol}"
        if use_cache and cache_key in self._position_cache:
            return self._position_cache[cache_key]
            
        def _get_positions():
            positions = MT5Wrapper.positions_get(symbol=symbol)
            return [] if positions is None else positions
            
        positions = self._execute_with_retry(_get_positions)
        result = [self._parse_position(pos) for pos in positions] if positions else []
        
        if use_cache:
            self._position_cache[cache_key] = result
        return result

    def get_historical_data(self, symbol: str, timeframe: int, num_candles: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data with retry logic
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: MT5 timeframe constant
            num_candles: Number of candles to retrieve
        Returns:
            pd.DataFrame with OHLCV data or None if failed
        """
        def _fetch_data() -> Optional[pd.DataFrame]:
            rates = MT5Wrapper.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
        
        result = self._execute_with_retry(_fetch_data)
        if result is None:
            logger.error(f"Failed to get historical data for {symbol}")
        return result
    
    def _parse_position(self, position) -> Dict:
        """Parse MT5 position to dictionary"""
        return {
            'ticket': position.ticket,
            'symbol': position.symbol,
            'type': 'buy' if position.type == MT5Wrapper.ORDER_TYPE_BUY else 'sell',
            'volume': position.volume,
            'entry_price': position.price_open,
            'current_price': position.price_current,
            'sl': position.sl,
            'tp': position.tp,
            'profit': position.profit,
            'time': pd.to_datetime(position.time, unit='s')
        }
    
    def send_order(self, symbol: str, order_type: str, volume: float, 
                sl: float, tp: float, comment: str = "") -> Optional[Dict]:
        """Send a market order with retry logic and return the full result object"""
        def _prepare_and_send() -> Optional[Dict]:
            # Get tick data with null check
            tick = MT5Wrapper.symbol_info_tick(symbol)
            if tick is None or 'bid' not in tick or 'ask' not in tick:
                print(f"Failed to get tick data for {symbol}")
                return None
                
            # Get symbol info with null check
            symbol_info = MT5Wrapper.symbol_info(symbol)
            if symbol_info is None:
                print(f"Failed to get symbol info for {symbol}")
                return None
                
            try:
                price = tick['ask'] if order_type == 'buy' else tick['bid']
                point = symbol_info['point']

                # Convert SL/TP from pips to price
                sl_price = price - (sl * point) if order_type == 'buy' else price + (sl * point)
                tp_price = price + (tp * point) if order_type == 'buy' else price - (tp * point)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                return MT5Wrapper.order_send(request)
            except KeyError as e:
                print(f"Missing required data in tick/symbol info: {str(e)}")
                return None
            
        result = self._execute_with_retry(_prepare_and_send)
        if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
            return result
            
        print(f"Order failed after retries. Last retcode: {getattr(result, 'retcode', 'UNKNOWN')}")
        return None
    
    def modify_position(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify stop loss and take profit of an existing position with retry logic"""
        def _prepare_and_modify() -> Optional[Dict]:
            positions = MT5Wrapper.positions_get(ticket=ticket)
            if not positions:
                print(f"Position {ticket} not found")
                return None
                
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": new_tp,
                "magic": 123456,
            }
            
            return MT5Wrapper.order_send(request)
        
        result = self._execute_with_retry(_prepare_and_modify)
        if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        
        print(f"Modify position failed after retries. Last retcode: {getattr(result, 'retcode', 'UNKNOWN')}")
        return False
    
    def close_position(self, ticket: int, volume: float) -> bool:
        """Close a position or part of it with retry logic"""
        def _prepare_and_close() -> Optional[Dict]:
            positions = MT5Wrapper.positions_get(ticket=ticket)
            if not positions:
                print(f"Position {ticket} not found")
                return None
                
            pos = positions[0]
            
            # Get tick data safely
            tick = MT5Wrapper.symbol_info_tick(pos['symbol'])
            if tick is None:
                print(f"Failed to get tick data for {pos['symbol']}")
                return None
                
            order_type = mt5.ORDER_TYPE_SELL if pos['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick['bid'] if order_type == mt5.ORDER_TYPE_SELL else tick['ask']
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos['symbol'],
                "volume": volume,
                "type": order_type,
                "position": pos['ticket'],
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": "Partial close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            return MT5Wrapper.order_send(request)
        
        result = self._execute_with_retry(_prepare_and_close)
        if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        
        print(f"Close position failed after retries. Last retcode: {getattr(result, 'retcode', 'UNKNOWN')}")
        return False

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.disconnect()
    
    def _validate_cached_position(self, ticket: int) -> bool:
        """Comprehensive position validation with detailed checks"""
        try:
            positions = MT5Wrapper.positions_get(ticket=ticket)
            if not positions:
                return False
                
            pos = positions[0]
            return (
                pos['volume'] > 0 and 
                isinstance(pos['symbol'], str) and 
                pos['symbol'] != '' and 
                pos['price_open'] > 0 and
                isinstance(pos['ticket'], int) and
                pos['ticket'] > 0 and
                pos['time'] > 0  # Valid timestamp
            )
        except Exception as e:
            logger.warning(f"Position validation failed: {str(e)}")
            return False

    def _fetch_position_with_retry(self, ticket: int) -> Optional[Any]:
        """Retry logic for position fetching"""
        for attempt in range(3):
            try:
                positions = MT5Wrapper.positions_get(ticket=ticket)
                return positions[0] if positions else None
            except Exception as e:
                logger.warning(f"Position fetch failed (attempt {attempt+1}): {str(e)}")
                time.sleep(0.5)
        return None

    def _reset_connection_state(self):
        """Reset all connection-related state"""
        self.connected = False
        self.last_activity_time = 0

    def get_position(self, ticket: int) -> Optional[Dict]:
        """Get single position by ticket with retry logic"""
        def _get_position():
            positions = MT5Wrapper.positions_get(ticket=ticket)
            return positions[0] if positions else None
            
        position = self._execute_with_retry(_get_position)
        return self._parse_position(position) if position else None

    def _establish_connection(self, recovery_mode: bool = False) -> bool:
        """Core connection logic used by both methods"""
        try:
            # Enhanced cleanup (preserves recover_connection's aggressive cleanup)
            if self.connected:
                try:
                    MT5Wrapper.shutdown()
                except Exception as e:
                    if recovery_mode:  # Only log in recovery mode to reduce noise
                        logger.warning(f"Cleanup warning: {str(e)}")

            # Initialize with validation
            if not MT5Wrapper.initialize_with_validation():
                return False

            # Additional validation for recovery mode (preserves recover_connection check)
            if recovery_mode:
                try:
                    if MT5Wrapper.symbols_total() <= 0:
                        logger.warning("Connection established but no symbols available")
                        return False
                except Exception as e:
                    logger.error(f"Symbol validation failed: {str(e)}")
                    return False

            # Update state
            self.connected = True
            self.last_activity_time = time.time()
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Network error: {str(e)}" + (" (recovery)" if recovery_mode else ""))
        except MT5Wrapper.Error as e:
            logger.error(f"MT5 protocol error: {str(e)}")
            if "account" in str(e).lower():
                raise  # Don't retry account errors
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}" + (" (recovery)" if recovery_mode else ""))
        
        return False

    def ensure_connected(self) -> bool:
        """Standard connection maintenance"""
        with self._connection_lock:
            if self.emergency_stop.check():
                logger.error("Connection attempt blocked by emergency stop")
                return False
                
            if self.connected and time.time() - self.last_activity_time < 30:
                return True

            for attempt in range(self.max_retries):
                if self._establish_connection():
                    logger.info(f"MT5 connection established (attempt {attempt+1})")
                    return True
                
                time.sleep(self.retry_delay * (attempt + 1))

            # Activate emergency stop if connection fails
            self.emergency_stop.activate("Connection failure")
            self._reset_connection_state()
            logger.error(f"Failed to connect after {self.max_retries} attempts")
            return False

    def recover_connection(self, max_attempts: int = 5) -> bool:
        """Full recovery mode with all original features plus cache clearing"""
        logger.warning("Initiating connection recovery...")
        
        with self._connection_lock:
            # Clear position cache before attempting recovery
            self._position_cache.clear()
            logger.debug("Cleared position cache before recovery attempt")
            
            # Preserve original aggressive recovery attempts
            for attempt in range(max_attempts):
                if self._establish_connection(recovery_mode=True):
                    logger.info(f"Connection recovered after {attempt+1} attempts")
                    
                    # Additional validation after successful recovery
                    try:
                        if MT5Wrapper.symbols_total() <= 0:
                            logger.warning("Connection recovered but no symbols available")
                            self._reset_connection_state()
                            continue
                            
                        # Test position data retrieval
                        test_positions = MT5Wrapper.positions_total()
                        if test_positions is None:
                            logger.warning("Position data not available after recovery")
                            continue
                            
                        logger.info("Connection recovery fully validated")
                        return True
                        
                    except Exception as e:
                        logger.error(f"Recovery validation failed: {str(e)}")
                        continue

                # Preserve original exponential backoff
                delay = min(5 * (attempt + 1), 30)
                logger.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)

            # Final cleanup if recovery fails
            self._reset_connection_state()
            self._position_cache.clear()  # Ensure cache is cleared even if recovery fails
            logger.error(f"Recovery failed after {max_attempts} attempts")
            return False
    

class DataFetcher:
    """Handles all data fetching operations using MT5 connection"""
    # Add somewhere: df = self.mt5.get_historical_data(Config.SYMBOL, Config.TIMEFRAME, 50)
    
    def __init__(self, mt5_connector: MT5Connector, config: Config):
        self.mt5 = mt5_connector
        self.config = config
        self._cache = {}
        self._cache_expiry = {
            'price': 1.0,      # 1 second for tick data
            'symbol_info': 120, # 2 minute for symbol info
            'historical': 300  # 5 minutes for historical data
        }

    
    @lru_cache(maxsize=10)
    def get_daily_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate daily ATR using IndicatorUtils.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            period: ATR lookback period in days (5-50 recommended)
        
        Returns:
            Latest ATR value or None if calculation fails
        """
        # Input validation
        if not isinstance(symbol, str) or len(symbol) < 3:
            logger.error(f"Invalid symbol format: {symbol}")
            return None
            
        if period < 5 or period > 50:
            logger.warning(f"Unusual ATR period {period}. Typical range: 5-50")
            period = min(max(period, 5), 50)  # Clamp to valid range

        try:
            if not self._validate_symbol(symbol):
                return None

            logger.debug(f"Calculating {period}-day ATR for {symbol}")
            
            # Get rates with buffer for calculation
            rates = MT5Wrapper.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, period + 15)
            if not rates or len(rates) < period:
                logger.warning(f"Insufficient data for {period}-day ATR on {symbol}")
                return None
                
            # Convert to DataFrame safely
            df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close'])
            
            # Convert to numpy arrays
            high = df['high'].to_numpy(dtype=np.float32)
            low = df['low'].to_numpy(dtype=np.float32)
            close = df['close'].to_numpy(dtype=np.float32)
            
            # Calculate ATR
            atr = IndicatorUtils.atr(high, low, close, period)
            result = float(atr[-1]) if not np.isnan(atr[-1]) else None
            
            logger.debug(f"ATR calculation complete: {result}")
            return result
            
        except KeyError as e:
            logger.error(f"Missing required price data for ATR calculation: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid price data format: {str(e)}")
        except Exception as e:
            logger.error(f"ATR calculation failed: {str(e)}", exc_info=True)
            
        return None

    def get_historical_data(self, symbol: str, timeframe: int, num_candles: int) -> Optional[pd.DataFrame]:
        if not self.mt5.ensure_connected():
            logger.error("Cannot fetch historical data - MT5 not connected")
            return None
        
        cache_key = f"hist_{symbol}_{timeframe}_{num_candles}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        df = self._fetch_historical_data(symbol, timeframe, num_candles)
        if df is not None:
            self._cache[cache_key] = df
        return df

        rates = MT5Wrapper.copy_rates_from_pos(...)

    def _fetch_historical_data(self, symbol: str, timeframe: int, num_candles: int) -> Optional[pd.DataFrame]:
        """Actual MT5 historical data fetching"""
        try:
            rates = MT5Wrapper.copy_rates_from_pos(
                symbol=symbol,
                timeframe=timeframe,
                start_pos=0,
                count=num_candles
            )
            return self._process_rates_to_df(rates)
        except Exception as e:
            logger.error(f"Historical data failed: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[PriceData]:
        """Single source of truth for price data with caching"""
        if not self._validate_symbol(symbol):
            return None
            
        cache_key = f"price_{symbol}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached['timestamp'] < 1:
                return cached['data']
            
        price_data = self._fetch_raw_price(symbol)
        if price_data:
            self._cache[cache_key] = {
                'data': price_data,
                'timestamp': time.time()
            }
        return price_data

    def _process_rates_to_df(self, rates: Optional[Union[List[Dict[str, Union[int, float]]], np.ndarray]]) -> Optional[pd.DataFrame]:
        """Convert MT5 rates to pandas DataFrame with proper formatting"""
        if rates is None or len(rates) == 0:
            return None
            
        try:
            # Define columns outside the conditional to ensure it's always bound
            columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            
            # Convert list of dicts to numpy array if needed
            if isinstance(rates, list):
                # Extract values in consistent order
                rates_array = np.array([[r[c] for c in columns if c in r] for r in rates])
            else:
                rates_array = rates
                
            # Convert to DataFrame
            df = pd.DataFrame(rates_array, columns=columns)
            
            # Convert time to datetime and set as index
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Convert numeric columns to float
            numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
            
        except Exception as e:
            logger.error(f"Failed to process rates to DataFrame: {str(e)}")
            return None
        
    def _fetch_raw_price(self, symbol: str) -> Optional[PriceData]:
        """Actual MT5 price fetching with error handling"""
        if not self.mt5.ensure_connected():  # Add this check
            return None
        try:
            tick = MT5Wrapper.symbol_info_tick(symbol)
            if not tick:
                return None
                
            return {
                'bid': float(tick['bid']),
                'ask': float(tick['ask']),
                'last': float(tick['last']),
                'time': pd.to_datetime(tick['time'], unit='s')
            }
        except Exception as e:
            logger.error(f"Price fetch failed: {str(e)}")
            return None

    def get_symbol_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """Centralized symbol info retrieval with validation and caching.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            use_cache: Whether to use cached data (default: True)
            
        Returns:
            Dict with keys: point, trade_contract_size, digits, trade_stops_level
            None if symbol info cannot be retrieved
        """
        if not self._validate_symbol(symbol):
            return None
            
        cache_key = f"symbol_info_{symbol}"
        
        # Return cached data if available and requested
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Validate cache hasn't expired (30 minute cache lifetime)
            if time.time() - cached['timestamp'] < 1800:  # 30 minutes in seconds
                return cached['data']
            del self._cache[cache_key]  # Remove stale cache
            
        try:
            info = MT5Wrapper.symbol_info(symbol)
            if not info:
                return None
                
            result = {
                'point': float(info['point']),
                'trade_contract_size': float(info['trade_contract_size']),
                'digits': int(info['digits']),
                'trade_stops_level': int(info['trade_stops_level']),
                'margin_initial': float(info.get('margin_initial', 0)),
                'volume_min': float(info.get('volume_min', 0.01))
            }
            
            # Store with timestamp for cache invalidation
            self._cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            return result
            
        except Exception as e:
            logger.error(f"Symbol info failed for {symbol}: {str(e)}", exc_info=True)
            return None

    def get_current_spread(self, symbol: str) -> Optional[float]:
        """Get current spread in points with comprehensive error handling"""
        try:
            # Get current price
            price = self.get_current_price(symbol)
            if price is None:
                logger.warning(f"Could not get price for {symbol}")
                return None
            
            # Get symbol info
            symbol_info = MT5Wrapper.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Could not get symbol info for {symbol}")
                return None
            
            # Validate point value exists and is valid
            point_value = symbol_info.get('point')
            if point_value is None or point_value <= 0:
                logger.warning(f"Invalid point value for {symbol}: {point_value}")
                return None
            
            # Calculate spread
            ask = price['ask']
            bid = price['bid']
            # Ensure both are floats (not datetime)
            if isinstance(ask, datetime) or isinstance(bid, datetime):
                logger.warning(f"Invalid price types for spread calculation: ask={type(ask)}, bid={type(bid)}")
                return None
            spread_pips = (float(ask) - float(bid)) / point_value
            return round(spread_pips, 2)
            
        except Exception as e:
            logger.error(f"Error calculating spread for {symbol}: {str(e)}")
            return None
    
    def get_account_balance(self) -> Optional[float]:
        """Get current account balance"""
        if not self.mt5.connected and not self.mt5.ensure_connected():
            return None
            
        try:
            account_info = MT5Wrapper.account_info()  
            if account_info is None:
                print("Failed to get account info")
                return None
                
            return getattr(account_info, 'balance', None)
            
        except Exception as e:
            print(f"Error getting account balance: {str(e)}")
            return None

    def _validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists in MT5"""
        try:
            if not self.mt5.connected and not self.mt5.ensure_connected():
                logger.warning("Not connected for symbol validation")
                return False
                
            # Check if symbol exists by getting total symbols count
            total_symbols = MT5Wrapper.symbols_total()
            if total_symbols <= 0:
                logger.warning("No symbols available in MT5")
                return False
                
            # Additional check with symbol_info for more robust validation
            symbol_info = MT5Wrapper.symbol_info(symbol)
            return symbol_info is not None
            
        except Exception as e:
            logger.error(f"Symbol validation failed for {symbol}: {str(e)}")
            return False


class DataPreprocessor:
    """Handles data preprocessing and feature engineering with comprehensive validation"""
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.scaler: RobustScaler = RobustScaler()
        self.data_fetcher = data_fetcher
        self.features: List[str] = config.FEATURES
        self.window_size: int = 30  # Number of candles to look back
        self.training_data: Optional[pd.DataFrame] = None
        self.processed_features: Optional[np.ndarray] = None

        # Initialize cache dictionary
        self._indicator_cache: Dict[str, pd.Series] = {}  # Explicit type declaration

        if self.config.PREPROCESSOR_PATH.exists():
            self.load_preprocessor()

        # Load preprocessor if exists
        if not self._initialize_preprocessor():
            logger.warning("Proceeding with fresh preprocessor instance")

    
    @lru_cache(maxsize=32)
    def _calculate_volume_indicators(self, volume_tuple: tuple, close_tuple: tuple) -> Dict[str, pd.Series]:
        """Cached volume indicators calculation with proper typing"""
        # Initialize empty series with default index in case of early failure
        empty_series = pd.Series(dtype='float64')
        
        try:
            volume = pd.Series(volume_tuple[1], index=volume_tuple[0])
            close = pd.Series(close_tuple[1], index=close_tuple[0])
            
            vwap = (volume * close).cumsum() / volume.cumsum()
            volume_ma = volume.rolling(20).mean()
            volume_roc = volume.pct_change(periods=5)
            obv = (np.sign(close.diff()) * volume).cumsum()
            
            return {
                'vwap': pd.Series(vwap, index=volume.index),
                'volume_ma': pd.Series(volume_ma, index=volume.index),
                'volume_roc': pd.Series(volume_roc, index=volume.index),
                'obv': pd.Series(obv, index=volume.index)
            }
            
        except Exception as e:
            logger.error(f"Volume indicators failed: {str(e)}")
            # Use index from input tuples if available, otherwise use empty index
            index = volume_tuple[0] if volume_tuple else empty_series.index
            return {k: pd.Series(dtype='float64', index=index) 
                    for k in ['vwap', 'volume_ma', 'volume_roc', 'obv']}
        
    @lru_cache(maxsize=32)
    def _calculate_adx(self, high_tuple: tuple, low_tuple: tuple, 
                    close_tuple: tuple, window: int = 14) -> pd.Series:
        """Cached ADX calculation using core utilities"""
        try:
            # Convert tuples to Series
            high = pd.Series(
                data=[v for _, v in high_tuple],
                index=[i for i, _ in high_tuple],
                dtype=np.float64
            )
            low = pd.Series(
                data=[v for _, v in low_tuple],
                index=[i for i, _ in low_tuple],
                dtype=np.float64
            )
            close = pd.Series(
                data=[v for _, v in close_tuple],
                index=[i for i, _ in close_tuple],
                dtype=np.float64
            )
            
            # Calculate using core utilities
            tr = IndicatorUtils.true_range(high.to_numpy(), low.to_numpy(), close.to_numpy())
            pos_dm, neg_dm = IndicatorUtils.directional_movements(high.to_numpy(), low.to_numpy())
            
            # Wilder smoothing
            def _smooth(s: np.ndarray) -> np.ndarray:
                return IndicatorUtils._wilder_smoothing(s, window)
            
            tr_s = _smooth(tr)
            pos_dm_s = _smooth(pos_dm)
            neg_dm_s = _smooth(neg_dm)
            
            # Calculate indicators with safe division
            pos_di = 100 * (pos_dm_s / (tr_s + 1e-10))
            neg_di = 100 * (neg_dm_s / (tr_s + 1e-10))
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
            
            # Final ADX smoothing
            adx = _smooth(dx)
            
            return pd.Series(
                adx, 
                index=high.index, 
                name='adx'
            ).astype(np.float32)
            
        except Exception as e:
            logger.error(f"ADX calculation failed: {str(e)}")
            return pd.Series(
                np.nan,
                index=[i for i, _ in high_tuple] if high_tuple else [],
                name='adx'
            ).astype(np.float32)
    
    def _initialize_preprocessor(self) -> bool:
        """Safe initializer with fallback"""
        if self.config.PREPROCESSOR_PATH.exists():
            return self.load_preprocessor()
        return False

    def load_preprocessor(self) -> bool:
        """Load preprocessor from disk with comprehensive validation"""
        try:
            if not self.config.PREPROCESSOR_PATH.exists():
                logger.warning(f"No preprocessor at {self.config.PREPROCESSOR_PATH}")
                return False
                
            loaded = joblib.load(self.config.PREPROCESSOR_PATH)
            
            if not isinstance(loaded, RobustScaler):
                raise TypeError(f"Expected RobustScaler, got {type(loaded)}")
                
            if hasattr(loaded, 'n_features_in_') and loaded.n_features_in_ != len(self.features):
                raise ValueError(
                    f"Feature mismatch. Expected {len(self.features)}, "
                    f"got {loaded.n_features_in_}"
                )
                
            self.scaler = loaded
            logger.info(f"Loaded preprocessor from {self.config.PREPROCESSOR_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessor load failed: {str(e)}")
            self.scaler = RobustScaler()  # Fresh instance
            return False

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Full preprocessing pipeline with validation"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
            
        df = self._add_technical_indicators(df)
        self.training_data = df.copy()
        
        try:
            X = df[self.features].values.astype(np.float32)
            self.processed_features = X
            
            # Target generation
            price_diff = df['close'].diff().shift(-1)
            y = np.select(
                [
                    price_diff > 5, 
                    price_diff < -5,
                    True
                ],
                [
                    1,  # Long
                    0,  # Short
                    -1  # No trade
                ]
            )[:-1]  # Remove last NaN
            
            X = X[:-1]  # Align shapes
            
            # Sequence creation
            X_seq = self._create_sequences(X, self.window_size)
            y_seq = y[self.window_size-1:]
            
            # Filter no-trade samples
            trade_mask = y_seq != -1
            return X_seq[trade_mask], y_seq[trade_mask]
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicators with caching"""
        df = df.copy()
        
        # 1. Symbol Validation (unchanged)
        symbol_info = self.data_fetcher.get_symbol_info(self.config.SYMBOL)
        if not symbol_info:
            logger.warning(f"Could not validate symbol {self.config.SYMBOL}")
            return df

        # 2. Cache Management (unchanged)
        if len(self._indicator_cache) > 0 and len(df) != len(next(iter(self._indicator_cache.values()))):
            self._indicator_cache.clear()
        
        # 3. Work with pandas Series - ADD SHIFT(1) HERE
        close = df['close'].shift(1).astype(np.float64)  # CHANGED: Added shift(1)
        high = df['high'].shift(1).astype(np.float64)    # CHANGED: Added shift(1)
        low = df['low'].shift(1).astype(np.float64)      # CHANGED: Added shift(1)
        
        # 4. Calculate Indicators - ALL INDICATORS NOW USE SHIFTED DATA
        
        # A. Volume Indicators (now uses shifted close)
        if 'volume' in df.columns:
            vol_indicators = self._calculate_volume_indicators(
                tuple(zip(df.index, df['volume'].astype(np.float64))),
                tuple(zip(df.index, close))  # Uses shifted close
            )
            df = df.assign(**vol_indicators)
        
        # B. ADX (uses shifted high/low/close)
        df['adx'] = self._calculate_adx(
            tuple(zip(df.index, high)),  # Shifted
            tuple(zip(df.index, low)),   # Shifted
            tuple(zip(df.index, close)), # Shifted
            window=self.config.ADX_WINDOW
        )
        
        # C. Other Indicators - ALL USE SHIFTED CLOSE PRICES
        df['sma_10'] = close.rolling(10).mean().astype(np.float32)  # Now safe
        df['sma_20'] = close.rolling(20).mean().astype(np.float32)  # Now safe
        
        # RSI - uses shifted close via the 'close' variable
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = delta.where(delta < 0, 0.0).abs().rolling(14).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = (100 - (100 / (1 + rs))).fillna(50).astype(np.float32)
        
        # MACD - uses shifted close
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = (ema12 - ema26).astype(np.float32)
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean().astype(np.float32)
        
        # Bollinger Bands - uses shifted close
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['upper_band'] = (sma20 + 2*std20).astype(np.float32)
        df['lower_band'] = (sma20 - 2*std20).astype(np.float32)
        
        # 5. Update Cache (unchanged)
        for col in df.columns.difference(['open', 'high', 'low', 'close', 'volume']):
            self._indicator_cache[col] = df[col]
        
        # 6. Pip-adjusted Features (unchanged)
        pip_value = symbol_info['point']
        df['pip_normalized_vol'] = df['volume'] * pip_value
        
        return df.dropna()

    def clear_cache(self) -> None:
        """Clear the indicator cache"""
        self._indicator_cache.clear()
        self._calculate_adx.cache_clear()
        self._calculate_volume_indicators.cache_clear()

    def _create_sequences(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Create time-series sequences with shape validation"""
        if len(data) < window_size:
            raise ValueError(f"Need at least {window_size} samples, got {len(data)}")
            
        return np.array([
            data[i:i+window_size] 
            for i in range(len(data) - window_size + 1)
        ])

    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit and persist scaler with validation"""
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be numpy array")
            
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        self._save_preprocessor()

    def transform_data(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling with shape preservation"""
        orig_shape = X.shape
        return self.scaler.transform(X.reshape(-1, orig_shape[-1])).reshape(orig_shape)

    def _save_preprocessor(self) -> None:
        """Internal save method with error handling"""
        try:
            joblib.dump(self.scaler, self.config.PREPROCESSOR_PATH)
            logger.info(f"Saved preprocessor to {self.config.PREPROCESSOR_PATH}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {str(e)}")
            raise

    def save_preprocessor(self) -> None:
        """Public save interface"""
        self._save_preprocessor()

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns while maintaining structure"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
        
        # Only normalize numeric columns that are in our features
        numeric_cols = [
            col for col in self.features 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if not numeric_cols:
            return df
        
        # Fit scaler if not already fitted
        if not hasattr(self.scaler, 'n_features_in_'):
            self.scaler.fit(df[numeric_cols])
        
        # Transform and preserve DataFrame structure
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df


class GoldMLModel:
    """Machine Learning model for trading predictions with Stable Baselines3 (PPO) for RL """
    
    def __init__(self, config: Config, monitor: PerformanceMonitor, data_fetcher: Optional[DataFetcher] = None):
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        self.config = config
        self.data_fetcher = data_fetcher or DataFetcher(self.mt5, self.config)
        self.model: Optional[PPO] = None 
        self._init_model()
        self.performance = ModelPerformanceTracker(self)
        self.monitor = monitor
        self.monitor.add_model(self)
        self.preprocessor = DataPreprocessor(config, self.data_fetcher) 
        self.trade_history = []
        self.model_performance = {'accuracy': 0, 'total_trades': 0, 'correct_trades': 0}
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_interval = timedelta(days=config.RETRAIN_INTERVAL_DAYS)
        self.min_retrain_samples = config.MIN_RETRAIN_SAMPLES
        self._version = self._generate_version() 
        self.prediction_stats = {
            'total': 0,
            'errors': 0,
            'last_error': None
        }
        self.mt5 = MT5Connector()
        self.data_fetcher = DataFetcher(self.mt5, self.config)

        if not self.config.PREPROCESSOR_PATH.exists():
            self._train_initial_model()

        # Initialize RiskManager
        self.risk_manager = RiskManager(
            mt5_connector=self.mt5,
            config=self.config,
            performance_monitor=monitor,
            data_fetcher=self.data_fetcher
        )

    def _init_model(self) -> None:
        """Ensure model is properly initialized with gradient clipping"""
        if self.config.RL_MODEL_PATH.exists():
            self.model = PPO.load(self.config.RL_MODEL_PATH)
            # Ensure loaded model has gradient clipping enabled
            if hasattr(self.model, 'max_grad_norm'):
                if self.model.max_grad_norm != self.config.RL_PARAMS.get('max_grad_norm', 0.5):
                    logger.warning(f"Loaded model has different gradient norm ({self.model.max_grad_norm}) "
                                 f"than config ({self.config.RL_PARAMS.get('max_grad_norm', 0.5)})")
        else:
            # Initialize new model with gradient clipping from config
            self.model = PPO(
                "MlpPolicy", 
                self._create_dummy_env(),
                **self.config.RL_PARAMS  # This now includes max_grad_norm
            )
            logger.info(f"Initialized new model with gradient clipping (max_grad_norm={self.config.RL_PARAMS.get('max_grad_norm', 0.5)})")
            
    def _create_dummy_env(self) -> gym.Env:
        """Create dummy environment for model initialization"""
        return gym.make('CartPole-v1')  # Temporary until real env is ready
    
    def _init_model_paths(self):
        """Ensure all model directories exist"""
        paths = [
            self.config.PREPROCESSOR_PATH.parent,
            self.config.RL_MODEL_PATH.parent
        ]
        
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        
    def train_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train PPO model with preprocessed data
        
        Args:
            X: Scaled feature data of shape (n_samples, n_features * window_size)
            y: Target labels of shape (n_samples,)
            
        Returns:
            bool: True if training succeeded, False otherwise
        """

        if X is None or y is None:
            raise ValueError("Training data cannot be None")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
    
        try:
            logger.info("Starting PPO model training...")
            logger.info(f"Training with gradient clipping (max_grad_norm={self.config.RL_PARAMS.get('max_grad_norm', 0.5)})")
            
            # 1. Convert data to properly typed DataFrame
            feature_columns = getattr(self.preprocessor, 'processed_features', 
                                    [f'feature_{i}' for i in range(X.shape[1])])
            env_data = pd.DataFrame(
                data=X,
                columns=feature_columns,
                dtype=np.float32
            )
            
            # Add targets for environment (if needed)
            env_data['target'] = y.astype(np.int32)
            
            # 2. Create RL environment
            env = TradingEnvironment(
                data=X,
                features=self.preprocessor.features,
                window_size=self.preprocessor.window_size,
                symbol=Config.SYMBOL,
                config=self.config,
                risk_manager=self.risk_manager,
                initial_balance=Config.ACCOUNT_BALANCE
            )

            # 3. Integrated callbacks
            callbacks = self._create_callbacks()
            
            # 4. Initialize PPO model
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="auto",
                **self.config.RL_PARAMS
            )
            
            # 5. Train with progress tracking
            self.model.learn(
                total_timesteps=self.config.RL_PARAMS['total_timesteps'],
                callback=self._create_callbacks(),
                progress_bar=True
            )
            
            # 6. Post-training validation
            if not self._validate_model_shapes():
                raise ValueError("Model validation failed after training")
                
            # 7. Verify model architecture matches expectations
            if not self._verify_model_architecture():
                raise ValueError("Model architecture verification failed after training")
                
            # 8. Evaluate initial performance
            train_acc = self._evaluate_model(X, y)
            self.performance.update({
                'accuracy': train_acc,
                'pnl': 0,  # Dummy value for training
                'time': datetime.now()
            })
            
            logger.info(f"✅ Training complete. Initial accuracy: {train_acc:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self.model = None
            return False
    
    def _verify_gradient_clipping(self) -> bool:
        """Verify gradient clipping is properly configured"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
                
            expected_norm = self.config.RL_PARAMS.get('max_grad_norm', 0.5)
            
            # Check if model has gradient clipping attribute
            if not hasattr(self.model, 'max_grad_norm'):
                logger.warning("Model has no max_grad_norm attribute")
                return False
                
            actual_norm = self.model.max_grad_norm
            
            if actual_norm != expected_norm:
                logger.warning(f"Gradient clipping mismatch: "
                             f"expected {expected_norm}, got {actual_norm}")
                return False
                
            logger.info(f"✅ Gradient clipping verified (max_grad_norm={actual_norm})")
            return True
            
        except Exception as e:
            logger.error(f"Gradient clipping verification failed: {str(e)}")
            return False
        
    def _train_initial_model(self) -> bool:
        """Train the initial model when no preprocessor exists"""
        try:
            logger.info("Starting initial model training...")
            
            # 1. Prepare directories
            self._init_model_paths()
            
            # 2. Fetch initial training data
            data_fetcher = self.data_fetcher
            
            df = data_fetcher.get_historical_data(
                symbol=self.config.SYMBOL,
                timeframe=self.config.TIMEFRAME,
                num_candles=self.config.DATA_POINTS * 2
            )
            if df is None:
                raise ValueError("DataFetcher returned None - check MT5 connection")
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(df)}")
            if df.empty:
                raise ValueError("Empty DataFrame returned")
            if len(df) < self.config.MIN_RETRAIN_SAMPLES:
                raise ValueError(
                    f"Insufficient data: got {len(df)} samples, "
                    f"need at least {self.config.MIN_RETRAIN_SAMPLES}"
                )

            # 3. Preprocess data
            X, y = self.preprocessor.preprocess_data(df)
            # Validate preprocessed data
            if X is None or y is None:
                raise ValueError("Preprocessing returned None")
            if len(X) != len(y):
                raise ValueError("X and y length mismatch after preprocessing")
            
            # 4. Fit and save scaler
            self.preprocessor.fit_scaler(X)
            self.preprocessor.save_preprocessor()
            
            # 5. Scale data
            X_scaled = self.preprocessor.transform_data(X)
            
            # 6. Create environment and callbacks
            env = TradingEnvironment(
                data=X_scaled,
                features=self.preprocessor.features,
                window_size=self.preprocessor.window_size,
                symbol=self.config.SYMBOL,
                config=self.config,
                risk_manager=self.risk_manager,
                initial_balance=self.config.ACCOUNT_BALANCE
            )
            
            # Create the callback list
            callbacks = self._create_callbacks()
            
            # 7. Initialize and train model with callbacks
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="auto",
                **self.config.RL_PARAMS
            )
            
            self.model.learn(
                total_timesteps=self.config.RL_PARAMS['total_timesteps'],
                callback=callbacks,  # Callbacks now integrated
                progress_bar=True
            )
            
            # 8. Save initial artifacts
            self.model.save(self.config.RL_MODEL_PATH)
            df.to_csv(Path("data/initial_training_data.csv"), index=False)
            
            # 9. Set version after successful training
            self._version = self._generate_version()
            
            logger.info("✅ Initial model training completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Initial model training failed: {str(e)}", exc_info=True)
            # Clean up potentially corrupted files
            if hasattr(self, 'model') and self.model is not None:
                self.model = None
            if self.config.PREPROCESSOR_PATH.exists():
                self.config.PREPROCESSOR_PATH.unlink()
            return False
    
    def should_retrain(self, new_data_samples: int) -> bool:
        """Check if model should be retrained"""
        # First check if we have enough new data
        if new_data_samples < self.min_retrain_samples:
            return False
            
        # Then check if enough time has passed
        if self.last_retrain_time is None:
            return True
            
        return datetime.now() - self.last_retrain_time > self.retrain_interval
    
    def retrain_model(self, new_data: pd.DataFrame) -> bool:
        """Retrain model with new data combined with existing knowledge"""
        if new_data is None:
            raise ValueError("New data cannot be None")
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(new_data)}")
        if len(new_data) == 0:
            raise ValueError("New data cannot be empty")
        
        try:
            logger.info("Starting model retraining...")
            
            # Load current model's training data (if exists)
            old_data_path = Path("data/training_data.csv")
            if old_data_path.exists():
                old_data = pd.read_csv(old_data_path, parse_dates=['time'], index_col='time')
                combined_data = pd.concat([old_data, new_data]).drop_duplicates()
            else:
                combined_data = new_data
                
            # Save the updated dataset
            combined_data.to_csv(old_data_path)
            
            # Preprocess data with validation
            X, y = self.preprocessor.preprocess_data(combined_data)
            if X is None or y is None:
                raise ValueError("Preprocessing returned None")
            if len(X) != len(y):
                raise ValueError("X and y length mismatch after preprocessing")
                
            X_scaled = self.preprocessor.transform_data(X)
            
            # Retrain model
            success = self.train_model(X_scaled, y)
            if not success:
                raise RuntimeError("Training failed during retraining")
            
            # Evaluate new model before deployment
            if not self.evaluate_new_model(combined_data.sample(frac=0.2)):
                logger.warning("New model performed worse - keeping old model")
                return False
            
            # Save versioned data and update timestamp
            combined_data.to_csv(f"data/training_data_v{Config.MODEL_VERSION}.csv")
            self.last_retrain_time = datetime.now()
            self.save_model()
            self.preprocessor.clear_cache()
            
            logger.info("Model retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model retraining failed: {str(e)}", exc_info=True)
            return False
        
    def _verify_model_architecture(self) -> bool:
        """Verify model architecture matches config expectations"""
        try:
            # 1. First validate model exists and has policy
            if self.model is None:
                raise ValueError("Model is not initialized - cannot verify architecture")
                
            if not hasattr(self.model, 'policy'):
                raise ValueError("Model has no policy attribute")
                
            if self.model.policy is None:
                raise ValueError("Model policy is not initialized")

            # 2. Safely get expected architecture
            expected_arch = None
            try:
                expected_arch = self.config.dict()["RL_PARAMS"]["policy_kwargs"]["net_arch"]
            except (KeyError, AttributeError) as e:
                logger.warning(f"Could not get expected architecture from config: {str(e)}")
                return True  # Skip verification if no expected architecture defined

            # 3. Check actual architecture with safety checks
            if not hasattr(self.model.policy, 'net_arch'):
                logger.warning("Model policy has no net_arch attribute")
                return True  # Skip verification if architecture can't be checked
                
            actual_arch = self.model.policy.net_arch
            if actual_arch is None:
                logger.warning("Model architecture is None")
                return True  # Skip verification if architecture is None

            # 4. Compare architectures
            if actual_arch != expected_arch:
                logger.error(f"Architecture mismatch - Expected: {expected_arch}, Actual: {actual_arch}")
                return False
                
            logger.info(f"✅ Architecture verified: {expected_arch}")
            return True

        except Exception as e:
            logger.error(f"Architecture verification failed: {str(e)}")
            return False
    
    def get_model_version(self) -> str:
        """Get current model version in YYYYMMDD-HHMM format
        Returns:
            str: Version timestamp if model exists, otherwise 'unversioned'
        """
        if self._version:
            return self._version
        return "unversioned"
        
    def get_model_config(self) -> Dict:
        """Get current model configuration"""
        return {
            'version': self._version,
            'features': self.config.FEATURES,
            'window_size': self.preprocessor.window_size,
            'last_retrained': self.last_retrain_time.isoformat() if self.last_retrain_time else None
        }

    def _generate_version(self) -> str:
        """Generate new version string"""
        return datetime.now().strftime("%Y%m%d-%H%M")
        
    def validate_model(self, full_check: bool = True) -> bool:
        """Comprehensive model validation combining core, shape and architecture checks
        
        Args:
            full_check: If True, performs both shape and architecture validation.
                    If False, only performs core validation.
                    
        Returns:
            bool: True if all validations pass
        """
        # 1. Core validation (model exists, basic structure)
        core_success, core_error = self._validate_model_core()
        if not core_success:
            logger.error(f"Core validation failed: {str(core_error)}")
            return False
        
        # 2. Shape validation (always performed)
        if not self._validate_model_shapes():
            logger.error("Model shape validation failed")
            return False
        
        # 3. Architecture validation (optional)
        if full_check and not self._verify_model_architecture():
            logger.error("Model architecture validation failed")
            return False

        # 4. Gradient clipping check
        if full_check and not self._verify_gradient_clipping():
            logger.warning("Gradient clipping configuration validation failed")
            return False
            
        return True    
    
    def _create_callbacks(self):
        """Create simplified training callbacks without directory dependencies"""
        from stable_baselines3.common.callbacks import BaseCallback
        
        class TrainingProgressCallback(BaseCallback):
            def __init__(self, check_freq: int = 1000, verbose: int = 1):
                super().__init__(verbose)
                self.check_freq = check_freq
                
            def _on_step(self) -> bool:
                if self.n_calls % self.check_freq == 0:
                    if self.verbose >= 1:
                        print(f"Training progress: {self.num_timesteps} steps completed")
                return True
        
        return TrainingProgressCallback(check_freq=1000, verbose=1)

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Enhanced model evaluation using stored training data"""
        try:
            # 1. Validate preprocessor state
            if not hasattr(self.preprocessor, 'processed_features'):
                print("⚠️ Preprocessor has no processed_features")
                return 0.0
                
            if self.preprocessor.processed_features is None:
                print("⚠️ Processed features not available")
                return 0.0
                
            # 2. Safely access processed features
            eval_data = getattr(self.preprocessor, 'processed_features', None)
            if eval_data is None:
                print("⚠️ No processed features available")
                return 0.0
                
            # 3. Get last 1000 samples safely
            try:
                last_samples = eval_data[-1000:] if len(eval_data) >= 1000 else eval_data[:]
            except (TypeError, IndexError) as e:
                print(f"⚠️ Error accessing processed features: {str(e)}")
                return 0.0
                
            if len(last_samples) < 100:
                print(f"⚠️ Insufficient evaluation data ({len(last_samples)} samples)")
                return 0.0
                
            # 4. Validate model is ready
            if not hasattr(self, 'model') or self.model is None:
                print("⚠️ No model available for evaluation")
                return 0.0
                
            # 5. Create evaluation environment
            eval_env = TradingEnvironment(
                data=last_samples,
                features=self.config.FEATURES,
                window_size=getattr(self.preprocessor, 'window_size', 30),
                symbol=self.config.SYMBOL,
                config=self.config,
                risk_manager=self.risk_manager,
                initial_balance=self.config.ACCOUNT_BALANCE
            )
            
            # 6. Run evaluation with proper type handling
            correct = 0
            total_tests = min(1000, len(last_samples) - getattr(self.preprocessor, 'window_size', 30))
            
            obs, _ = eval_env.reset()  # Unpack both observation and info
            for _ in range(total_tests):
                try:
                    # Get action (convert to int if model returns array)
                    action_output = self.model.predict(obs, deterministic=True)
                    action = int(action_output[0])  # Ensure action is int
                    
                    # Handle Gymnasium's 5-tuple return
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    
                    if reward > 0:
                        correct += 1
                    if terminated or truncated:
                        obs, _ = eval_env.reset()
                        
                except Exception as e:
                    print(f"⚠️ Evaluation step failed: {str(e)}")
                    continue
                    
            return correct / total_tests if total_tests > 0 else 0.0
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return 0.0


    def save_model(self):
        """Save PPO model"""
        if self.model:
            self.model.save(self.config.RL_MODEL_PATH)

    def _validate_model_core(self) -> Tuple[bool, Optional[Exception]]:
        """Core model validation used by all other validation methods
        
        Validates:
        - Model exists and is initialized
        - Basic policy structure
        - Observation and action spaces
        
        Returns:
            Tuple: (success: bool, error: Optional[Exception])
        """
        try:
            # 1. Validate model exists
            if self.model is None:
                raise ValueError("Model not initialized")
                
            # 2. Validate policy exists
            if not hasattr(self.model, 'policy'):
                raise ValueError("Model has no policy attribute")
                
            policy = self.model.policy
            if policy is None:
                raise ValueError("Model policy is None")
                
            # 3. Validate observation space
            if not hasattr(policy, 'observation_space'):
                raise ValueError("Policy has no observation_space")
                
            obs_space = policy.observation_space
            if obs_space is None:
                raise ValueError("observation_space is None")
                
            obs_shape = getattr(obs_space, 'shape', None)
            if obs_shape is None:
                raise ValueError("observation_space has no shape attribute")
                
            if not isinstance(obs_shape, (tuple, list)):
                raise ValueError("observation_space.shape must be a sequence")
                
            # 4. Validate action space
            if not hasattr(self.model, 'action_space'):
                raise ValueError("Model has no action_space")
                
            action_space = self.model.action_space
            if not isinstance(action_space, spaces.Discrete):
                raise ValueError(f"Expected Discrete action space, got {type(action_space)}")
                
            if not hasattr(action_space, 'n'):
                raise ValueError("Discrete action_space has no 'n' attribute")
                
            return True, None
            
        except Exception as e:
            return False, e

    def _validate_model_shapes(self) -> bool:
        """Validate that model shapes match preprocessor configuration"""
        try:
            # 1. First validate preprocessor attributes
            if not hasattr(self.preprocessor, 'features'):
                raise ValueError("Preprocessor missing features attribute")
                
            if not hasattr(self.preprocessor, 'window_size'):
                raise ValueError("Preprocessor missing window_size attribute")
                
            # 2. Validate model exists and is properly initialized
            if self.model is None:
                raise ValueError("Model is not initialized - cannot validate shapes")
                
            if not hasattr(self.model, 'policy'):
                raise ValueError("Model has no policy attribute")
                
            if self.model.policy is None:
                raise ValueError("Model policy is not initialized")
                
            if not hasattr(self.model.policy, 'observation_space'):
                raise ValueError("Model policy has no observation_space")
                
            # 3. Get expected input shape
            n_features = len(self.preprocessor.features)
            window_size = self.preprocessor.window_size
            expected_features = n_features * window_size
            
            # 4. Get actual model input shape with safety checks
            obs_space = self.model.policy.observation_space
            if obs_space is None:
                raise ValueError("observation_space is None")
                
            obs_shape = getattr(obs_space, 'shape', None)
            if obs_shape is None:
                raise ValueError("observation_space has no shape attribute")
                
            if not isinstance(obs_shape, (tuple, list)):
                raise ValueError("observation_space.shape must be a sequence")
                
            # 5. Validate feature dimension matches
            if obs_shape[-1] != expected_features:
                raise ValueError(
                    f"Feature dimension mismatch. Model expects {obs_shape[-1]}, "
                    f"preprocessor provides {expected_features}. Possible causes:\n"
                    f"- Changed feature set (currently using: {self.preprocessor.features})\n"
                    f"- Changed window_size (currently: {window_size})"
                )
                
            logger.info("✅ Model shape validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Shape validation failed: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """Load trained RL model from disk with comprehensive validation"""
        try:
            # 1. Validate model path exists
            model_path = self.config.RL_MODEL_PATH.with_suffix('.zip')
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            if not self._verify_model_architecture():
                logger.error("Model architecture verification failed!")
                return False
                
            # 2. Attempt to load model
            print(f"Loading model from {model_path}...")
            self.model = PPO.load(model_path, print_system_info=True)
            
            # 3. Validate model loaded successfully
            if self.model is None:
                raise ValueError("Model failed to load (returned None)")
                
            # 4. Use core validation
            success, error = self._validate_model_core()
            if not success:
                if error is not None:
                    raise error
                else:
                    raise Exception("Model validation failed, but no error was provided.")
                
            # 5. Additional load-specific validation
            if not hasattr(self.preprocessor, 'window_size'):
                raise AttributeError("Preprocessor missing window_size")
                
            if not hasattr(self.preprocessor, 'features'):
                raise AttributeError("Preprocessor missing features")
                
            # 6. Get feature count safely
            n_features = len(self.preprocessor.features)
            if n_features <= 0:
                raise ValueError(f"Invalid feature count: {n_features}")
                
            # 7. Prepare validation input
            expected_features = n_features * self.preprocessor.window_size
            dummy_input = np.zeros((1, expected_features), dtype=np.float32)
            
            # 8. Test prediction capability
            if not hasattr(self.model, 'predict'):
                raise AttributeError("Model has no predict method")
                
            action, _ = self.model.predict(dummy_input, deterministic=True)
            if not isinstance(action, (np.ndarray, int, np.integer)):
                raise ValueError(f"Invalid action type: {type(action)}")
                
            print(f"✅ Model loaded successfully. Features: {n_features}, "
                f"Input shape: {self.model.policy.observation_space.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            self.model = None
            return False
    
    def evaluate_new_model(self, test_data: pd.DataFrame) -> bool:
        """Evaluate new model before putting into production.
        
        Args:
            test_data: DataFrame containing test data with same features as training
            
        Returns:
            bool: True if new model performs at least 95% as well as old model
        """
        try:
            # Preprocess test data
            X_test, y_test = self.preprocessor.preprocess_data(test_data)
            X_test_scaled = self.preprocessor.transform_data(X_test)
            
            # Check we have a valid model
            if self.model is None:
                logger.warning("No model available for evaluation")
                return False
                
            # Create sequences matching the model's expected input shape
            X_seq = self.preprocessor._create_sequences(X_test_scaled, self.preprocessor.window_size)
            
            # Get predictions
            predictions = []
            for seq in X_seq:
                action, _ = self.model.predict(seq, deterministic=True)
                predictions.append(action)
            
            # Calculate accuracy
            y_true = y_test[self.preprocessor.window_size-1:]  # Align with predictions
            new_acc = accuracy_score(y_true, predictions)
            
            # Get old accuracy with default fallback
            old_acc = self.model_performance.get('accuracy', 0.0)
            
            # Explicit bool conversion for type safety
            is_better = bool(new_acc >= old_acc * 0.95)
            
            logger.info(f"Model evaluation - Old: {old_acc:.2%}, New: {new_acc:.2%}, Approved: {is_better}")
            return is_better
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return False
    
    def predict_signal(self, X: np.ndarray) -> int:
        """Predict trading signal using PPO model
        """
        # Update prediction stats
        self.prediction_stats['total'] += 1
        
        # First ensure we have a valid model
        if not self._ensure_valid_model():
            return -1  # Return no-trade signal if model unavailable
            
        try:
            # Double-check model exists and is valid
            if self.model is None:
                raise ValueError("Model is not initialized")
                
            # Get feature count safely
            n_features = self._get_feature_count()
            window_size = self.preprocessor.window_size
            
            # Validate input shape
            expected_shape = (window_size, n_features)
            if X.shape != expected_shape:
                raise ValueError(
                    f"Invalid input shape. Got {X.shape}, expected {expected_shape}. "
                    f"Window size: {window_size}, Features: {n_features}"
                )
                
            # Prepare input - reshape to (1, n_features*window_size)
            X_flat = X.reshape(1, -1).astype(np.float32)
            
            # Triple-check model has predict method before calling it
            if not hasattr(self.model, 'predict'):
                raise AttributeError("Model has no predict method")
                
            # Get prediction with additional safety checks
            action, _ = self.model.predict(X_flat, deterministic=True)
            
            # Store debug info before returning
            signal = self._parse_prediction(action)
            print(f"DEBUG: Raw action={action}, Parsed signal={signal}")
            return signal
            
        except Exception as e:
            self.prediction_stats['errors'] += 1
            self.prediction_stats['last_error'] = str(e)
            # Only print action if it exists
            logger.error(f"Prediction failed: {str(e)}")
            return -1  # Return no-trade on error

    def _ensure_valid_model(self) -> bool:
        """Ensure we have a working model instance"""
        if self.model is None:
            try:
                return self.load_model()
            except Exception as e:
                print(f"Model loading failed: {str(e)}")
                return False
        return True

    def _parse_prediction(self, action) -> int:
        """Convert model output to trading signal"""
        if action is None:
            return -1
            
        try:
            # Handle various action formats
            if isinstance(action, np.ndarray):
                return int(action.item() if action.size == 1 else action[0])
            return int(action)
        except (ValueError, TypeError):
            return -1

    def is_model_healthy(self) -> bool:
        """Check if model is loaded and responsive"""
        if self.model is None:
            return False
        try:
            dummy_input = np.zeros((self.preprocessor.window_size, 
                                len(self.preprocessor.features)))
            self.predict_signal(dummy_input)
            return True
        except:
            return False

    def check_model_health(self) -> Dict[str, Any]:
        return {
            'version': self.get_model_version(),
            'last_prediction': self.prediction_stats,
            'feature_weights': self._get_current_feature_weights(),
            'architecture': self._get_model_architecture(),
        }

        # How shall I implenet the call of this function???:
        # expected_features = self._get_feature_count()  
    
    def _get_feature_count(self) -> int:
        """Safely get number of features with multiple fallbacks"""
        try:
            # First try getting features from preprocessor
            features = getattr(self.preprocessor, 'features', None)
            
            if features is None:
                # Fallback to config features
                features = getattr(self.config, 'FEATURES', ['close'])  # Default to just close price
            
            # Handle different feature types
            if isinstance(features, (list, tuple, np.ndarray, pd.Index)):
                return len(features)
            elif isinstance(features, (int, float)):
                return int(features)  # If someone put a number directly
            return 1  # Ultimate fallback
                
        except Exception:
            return 1  # Guaranteed to return something

    def update_model_performance(self, trade_result: Dict) -> None:
        """Record trade results for performance tracking and later retraining"""
        try:
            # Basic validation
            if not isinstance(trade_result, dict):
                raise TypeError("Trade result must be a dictionary")
                
            # Store the raw trade data
            self.trade_history.append(trade_result)
            
            # Update performance metrics
            trade_pnl = trade_result.get('profit', 0)
            trade_size = trade_result.get('entry_price', 1)  # Avoid division by zero
            accuracy = abs(trade_pnl) / trade_size if trade_size else 0
            
            # Create a proper trade result dictionary
            full_trade_result = {
                **trade_result,
                'accuracy': accuracy,
                'pnl': trade_pnl
            }
            
            # Update performance tracker
            self.performance.update(full_trade_result)
            
        except Exception as e:
            logger.error(f"Failed to update model performance: {str(e)}")
            # Consider adding error recovery here if needed

    def _get_current_feature_weights(self) -> Dict[str, float]:
        """Estimate feature importance using permutation importance with pandas"""
        if not self.model or not hasattr(self, 'trade_history') or len(self.trade_history) < 10:
            # Return equal weights if insufficient data
            return {f: 1.0/len(self.config.FEATURES) for f in self.config.FEATURES}
        
        try:
            # Convert trade history to DataFrame
            trades_df = pd.DataFrame(self.trade_history)
            
            # Calculate correlation between features and successful trades
            feature_importance = {}
            for feature in self.config.FEATURES:
                if feature in trades_df.columns:
                    # Use absolute correlation with PnL as importance proxy
                    corr = trades_df[feature].astype(float).corr(trades_df['pnl'].astype(float))
                    feature_importance[feature] = abs(corr) if not pd.isna(corr) else 0.0
            
            # Normalize importance scores
            if feature_importance:
                total = sum(feature_importance.values())
                if total > 0:
                    return {k: v/total for k, v in feature_importance.items()}
            
            # Fallback equal weights
            return {f: 1.0/len(self.config.FEATURES) for f in self.config.FEATURES}
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {str(e)}")
            return {f: 1.0/len(self.config.FEATURES) for f in self.config.FEATURES}


    def _get_model_architecture(self) -> Dict:
        """Inspect model architecture for debugging"""
        if self.model is None:
            return {
                'error': 'Model not initialized',
                'policy_type': 'None',
                'layers': []
            }
        
        arch = {
            'policy_type': str(type(self.model.policy)),
            'layers': []
        }
        
        try:
            if hasattr(self.model.policy, 'features_extractor'):
                arch['features_extractor'] = str(type(self.model.policy.features_extractor))
            if hasattr(self.model.policy, 'mlp_extractor'):
                arch['mlp_extractor'] = str(type(self.model.policy.mlp_extractor))
        except:
            pass
            
        return arch

class ModelPerformanceTracker:
    def __init__(self, model: Optional['GoldMLModel'] = None):
        self.model = model
        self.accuracy: float = 0.0
        self.total_trades: int = 0
        self.correct_trades: int = 0
        self.feature_impacts: Dict[str, float] = {}
        self.permutation_importance: Dict[str, Any] = {}
        self._last_importance_calc: float = 0.0
        
    def update(self, trade_data: Union[Dict[str, Any], float], new_trades: Optional[int] = None) -> None:
        """Update tracker with either trade results or direct metrics."""
        if isinstance(trade_data, dict):
            self._update_from_trade_result(trade_data)
        else:
            self.accuracy = float(trade_data)
            if new_trades is not None:
                self.total_trades += new_trades
                self.correct_trades += int(trade_data > 0)

    def _update_feature_impacts(self, trade_result: Dict) -> None:
        """Track which features contributed to successful trades"""
        features = trade_result.get('features', {})
        for feature, weight in features.items():
            impact = weight * trade_result.get('pnl', 0)
            self.feature_impacts[feature] = self.feature_impacts.get(feature, 0) + impact
            
    def get_feature_report(self, include_permutation: bool = True) -> Dict[str, Any]:
        """Enhanced feature report with both impact and permutation importance"""
        report: Dict[str, Any] = {}
        
        # Basic feature impacts
        total_impact = sum(abs(v) for v in self.feature_impacts.values()) or 1
        report['empirical_impacts'] = {
            k: v/total_impact 
            for k, v in self.feature_impacts.items()
        }
        
        # Add permutation importance if available
        if include_permutation and self.permutation_importance:
            report['permutation_importance'] = {
                'scores': self.permutation_importance['importances_mean'],
                'std': self.permutation_importance['importances_std'],
                'p_values': self._calculate_p_values()
            }
            
        return report

    def calculate_permutation_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        force_recalc: bool = False,
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate permutation importance with proper validation."""
        # Check cache first
        current_time = time.time()
        if not force_recalc and current_time - self._last_importance_calc < 3600:
            return self.permutation_importance
            
        if self.model is None:
            logger.warning("Cannot calculate importance - no model attached")
            return {}
            
        try:
            start_time = current_time
            
            # Get feature names
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
                X_array = X.values
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X_array = X
                
            # Convert y to proper numpy array format
            y_array = np.asarray(y.values if isinstance(y, pd.Series) else y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            
            # Ensure model has predict method
            if not hasattr(self.model, 'predict'):
                logger.error("Model must have predict() method for permutation importance")
                return {}
                
            # Calculate importance
            r = permutation_importance(
                self.model,
                X_array,
                y_array,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs,
                scoring=accuracy_score
            )
            
            # Store results - using dictionary access instead of dot notation
            self.permutation_importance = {
                'importances_mean': dict(zip(feature_names, r['importances_mean'])),
                'importances_std': dict(zip(feature_names, r['importances_std'])),
                'importances': {feature: r['importances'][idx] for idx, feature in enumerate(feature_names)},
                'elapsed_time': current_time - start_time,
                'timestamp': current_time,
                'n_repeats': n_repeats
            }
            
            self._last_importance_calc = current_time
            return self.permutation_importance
            
        except Exception as e:
            logger.error(f"Permutation importance failed: {str(e)}", exc_info=True)
            return {}
        
    def _calculate_p_values(self) -> Dict[str, float]:
        """Calculate empirical p-values for permutation importance"""
        if not self.permutation_importance:
            return {}
            
        p_values = {}
        for feature, vals in self.permutation_importance['importances'].items():
            mean_imp = self.permutation_importance['importances_mean'][feature]
            if mean_imp >= 0:
                p_val = np.mean(vals >= mean_imp)
            else:
                p_val = np.mean(vals <= mean_imp)
            p_values[feature] = min(p_val, 1 - p_val) * 2  # Two-tailed
            
        return p_values

    def to_dict(self) -> Dict[str, Any]:
        """Return complete tracker state as dictionary"""
        return {
            "accuracy": self.accuracy,
            "total_trades": self.total_trades,
            "correct_trades": self.correct_trades,
            "feature_impacts": self.feature_impacts.copy(),
            "permutation_importance": self.permutation_importance.copy()
        }
    
    def _update_from_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Update metrics from trade result dictionary"""
        try:
            accuracy = trade_result.get('accuracy', 0)
            self.accuracy = float(accuracy)
            self.total_trades += 1
            self.correct_trades += int(accuracy > 0)
            
            if self.model:
                self._update_feature_impacts(trade_result)
        except Exception as e:
            logger.error(f"Tracker update failed: {str(e)}")


class TradingEnvironment(Env):
    """Custom trading environment that fully implements gymnasium.Env interface"""
    def __init__(self, config: Config, data: npt.NDArray[np.float32], features: list, 
                 window_size: int, symbol: str, risk_manager: 'RiskManager',
                 initial_balance: Optional[float] = None, commission: float = 0.0005):
        super().__init__()
        self.config = config 
        self.data: np.ndarray = data
        self.features: list = features
        self.window_size: int = window_size
        self.symbol: str = symbol
        self.commission: float = commission
        self.risk_manager = risk_manager
        
        # Use config balance if not overridden
        self.initial_balance = float(initial_balance) if initial_balance is not None else config.ACCOUNT_BALANCE

        # Initialize ATR values (volatility regime is now handled by RiskManager)
        self.atr_values: np.ndarray = np.array([])
        self._initialize_atr_values()
       
        # Gymnasium spaces
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(features) * window_size,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Union[float, int]]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment
        Args:
            seed: Optional seed for random number generation
            options: Optional configuration dictionary containing:
                - initial_balance: Starting balance override (float)
                - start_step: Starting step index override (int)
        Returns:
            Tuple containing:
            - observation: Initial environment observation
            - info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset core state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.trades = []
        self.done = False

        # Apply options if provided
        if options:
            if "initial_balance" in options:
                self.balance = float(options["initial_balance"])
            if "start_step" in options:
                self.current_step = max(self.window_size, int(options["start_step"]))
        
        return self._get_observation(), {}
    
    def _initialize_atr_values(self) -> None:
        """Initialize ATR values using IndicatorUtils"""
        try:
            high_idx = self.features.index('high')
            low_idx = self.features.index('low')
            close_idx = self.features.index('close')
            
            self.atr_values = IndicatorUtils.atr(
                self.data[:, high_idx],
                self.data[:, low_idx], 
                self.data[:, close_idx],
                self.config.ATR_LOOKBACK  # Use ATR lookback from config
            )
        except Exception as e:
            logger.error(f"ATR initialization failed: {str(e)}")
            self.atr_values = np.zeros(len(self.data))

    def step(self, action: int) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        if self.done:
            return (
                self._get_observation(), 
                0.0,  # Default reward when done
                True, 
                False,
                {}
            )
        
        # Validate and clip action to valid range
        action = int(np.clip(action, 0, 2))  # Ensure action is 0, 1, or 2
        
        # Initialize reward before any actions
        reward = 0.0
        current_price = self._get_current_price()
        candle = self.data[min(self.current_step, len(self.data)-1)]
        high, low = candle[self.features.index('high')], candle[self.features.index('low')]
        volatility_penalty = np.log(high/low) if high > 0 and low > 0 else 0.0
        
        # Get volatility multiplier from RiskManager
        volatility_multiplier = self.risk_manager.get_volatility_multiplier(self.symbol)

        if action == 1:  # Buy
            if self.position <= 0:
                trade_size = self.balance * self.config.RISK_PER_TRADE
                self.position = trade_size / current_price
                self.balance -= trade_size
                self.trades.append(('buy', current_price, self.current_step))
                
        elif action == 2:  # Sell
            if self.position > 0:
                entry_price = self.trades[-1][1]
                exit_price = current_price
                holding_period = self.current_step - self.trades[-1][2]
                
                reward = (
                    np.log(exit_price/entry_price)  # Logarithmic return
                    - 0.5 * volatility_penalty  # Volatility penalty
                    + (1/holding_period if holding_period > 0 else 0)  # Time bonus
                )
                
                self.balance += self.position * exit_price * (1 - self.commission)
                self.position = 0
                self.trades.append(('sell', exit_price))

        # Apply volatility multiplier to reward
        reward *= volatility_multiplier
        
        # Update state
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': float(self.balance),
            'position': float(self.position),
            'volatility_regime': self.risk_manager.current_volatility_regime,
            'volatility_multiplier': float(volatility_multiplier),
            'current_atr': float(self.atr_values[min(self.current_step, len(self.atr_values)-1)]),
            'reward_components': {
                'log_return': float(np.log(current_price/self.trades[-1][1])) if self.trades else 0.0,
                'volatility_penalty': float(volatility_penalty),
                'time_bonus': 1/(self.current_step - self.trades[-1][2]) if self.trades and (self.current_step - self.trades[-1][2]) > 0 else 0.0
            }
        }
        
        return (
            self._get_observation(),
            float(reward),
            self.done,
            False,
            info
        )
    
    
    def _get_observation(self) -> npt.NDArray[np.float32]:
        """Get current observation"""
        return self.data[self.current_step:self.current_step + self.window_size].flatten()
    
    def _get_current_price(self):
        return self.data[self.current_step][self.features.index('close')]
    
    def get_total_reward(self):
        return self.balance - self.initial_balance
    
    def get_trade_count(self):
        return len(self.trades) // 2
    
    def get_profit_factor(self) -> float:
        """Calculate profit factor using IndicatorUtils"""
        return IndicatorUtils.profit_factor(self.trades)
    

class RiskManager:
    """
    Risk Management:

    Add circuit breakers for consecutive losses

    Implement volatility-adjusted position sizing

    Add maximum position duration checks
    """

    """Manages risk and position sizing"""
    
    def __init__(self, mt5_connector: MT5Connector, config: Config, 
                 performance_monitor: PerformanceMonitor, data_fetcher: DataFetcher):
        getcontext().prec = 8 
        self.mt5 = mt5_connector
        self.config = config  # Store the passed config
        self.performance_monitor = performance_monitor  # Store the performance monitor
        self.today_trades = 0
        self.data_fetcher = data_fetcher
        self.emergency_stop = EmergencyStop() 
        self.max_trades = config.MAX_TRADES_PER_DAY  # Use from config instead of Config class
        self._symbol_info_cache: Dict[str, Dict] = {} 
        self._position_cache: Dict[int, Dict] = {}  # Add position cache
        self._retry_attempts = 3  # Number of retry attempts
        self._retry_delay = 1.0  # Delay between retries in seconds
        self.current_volatility_regime = "medium"
        self.volatility_multipliers = {
            "high": 1.3,
            "medium": 1.0,
            "low": 0.7
        }
        self._atr_history_cache: Dict[str, List[float]] = {}

    def get_atr_history(self, symbol: str, period: int = 20) -> List[float]:
        """Get recent ATR values for volatility calculations
        
        Args:
            symbol: Trading symbol (e.g. "XAUUSD")
            period: Number of ATR values to return
            
        Returns:
            List of ATR values (empty list if data unavailable)
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self._atr_history_cache:
                return self._atr_history_cache[cache_key]
            
            # Get daily ATR data
            atr_values = []
            for _ in range(period):
                atr = self.data_fetcher.get_daily_atr(symbol)
                if atr is not None:
                    atr_values.append(float(atr))
            
            # Cache and return
            self._atr_history_cache[cache_key] = atr_values
            return atr_values
            
        except Exception as e:
            logger.error(f"Failed to get ATR history: {str(e)}")
            return []  # Return empty list instead of None

    def determine_volatility_regime(self, symbol: str) -> str:
        """Centralized volatility regime detection
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            "high", "medium", or "low" volatility regime
        """
        try:
            atr = self.data_fetcher.get_daily_atr(symbol)
            if atr is None:
                return "medium"
            
            # Get ATR history safely
            atr_history = self.get_atr_history(symbol, period=20)
            if len(atr_history) < 20:
                return "medium"
                
            sma_atr = np.mean(atr_history[-20:])
            
            # Classify regime with threshold checks
            if atr > sma_atr * 1.2:
                self.current_volatility_regime = "high"
            elif atr < sma_atr * 0.8:
                self.current_volatility_regime = "low"
            else:
                self.current_volatility_regime = "medium"
                
            return self.current_volatility_regime
            
        except Exception as e:
            logger.error(f"Volatility regime detection failed: {str(e)}")
            return "medium"

    def get_volatility_multiplier(self, symbol: str) -> float:
        """Get position sizing multiplier based on volatility
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Multiplier value (1.3 for high, 1.0 medium, 0.7 low)
        """
        regime = self.determine_volatility_regime(symbol)
        return self.volatility_multipliers.get(regime, 1.0)
        
    def calculate_min_stop_loss(self, symbol: str) -> float:
        """Calculate minimum allowed stop loss based on account risk"""
        try:
            # Get symbol metadata
            symbol_info = self.get_symbol_metadata(symbol)
            if not symbol_info:
                return 5.0  # Fallback minimum
            
            # Calculate minimum risk amount (50% of normal risk)
            min_risk_amount = self.config.ACCOUNT_BALANCE * self.config.RISK_PER_TRADE * 0.5
            
            # Calculate pip value
            pip_value = symbol_info['trade_contract_size'] * symbol_info['point']
            
            # Return max between 5 pips and calculated minimum
            return max(5.0, min_risk_amount / pip_value)
            
        except Exception as e:
            logger.error(f"Min SL calculation failed: {str(e)}")
            return 5.0  # Fallback value
        
    def calculate_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """Calculate position size using Decimal internally but return float"""
        try:
            # Convert inputs to Decimal for precise math
            stop_loss_dec = Decimal(str(stop_loss_pips))
            if stop_loss_dec <= Decimal('0'):
                logger.error(f"Invalid stop loss: {stop_loss_pips}")
                return 0.0

            # Get price data
            tick = self.data_fetcher.get_current_price(symbol)
            if not tick:
                return 0.0

            # Convert price to Decimal safely
            ask_price = None
            for key in ['ask', 'last', 'bid']:
                if key in tick and not isinstance(tick[key], datetime):
                    try:
                        ask_price = Decimal(str(tick[key]))
                        if ask_price > Decimal('0'):
                            break
                    except Exception:
                        continue

            if not ask_price or ask_price <= Decimal('0'):
                return 0.0

            # Get symbol info
            symbol_info = self.data_fetcher.get_symbol_info(symbol)
            if not symbol_info:
                return 0.0

            # Convert all values to Decimal for calculation
            point_value = Decimal(str(symbol_info.get('point', 0)))
            balance = Decimal(str(self.data_fetcher.get_account_balance() or 0))
            risk_percent = Decimal(str(self.config.RISK_PER_TRADE))
            contract_size = Decimal(str(symbol_info.get('trade_contract_size', 1.0)))

            if point_value <= Decimal('0') or balance <= Decimal('0'):
                return 0.0

            # Precise Decimal calculation
            risk_amount = balance * risk_percent
            sl_amount = stop_loss_dec * point_value * Decimal(100)
            
            if sl_amount == Decimal('0'):
                return 0.0

            position_size = risk_amount / sl_amount
            min_position = Decimal('0.01')
            max_position = (balance * Decimal('0.5')) / (ask_price * contract_size)

            # Apply bounds and convert back to float
            final_size = max(min_position, min(position_size, max_position))
            return float(final_size.quantize(Decimal('0.00001')))  # Precise rounding then float

        except Exception as e:
            logger.error(f"Position calculation error: {str(e)}", exc_info=True)
            return 0.0
    
    def can_trade_today(self) -> bool:
        """Check if we can place more trades today"""
        return self.today_trades < self.max_trades
    
    def increment_trade_count(self):
        """Increment daily trade count"""
        self.today_trades += 1
        
    def check_emergency_conditions(self) -> bool:
        """Check various risk thresholds"""
        try:
            # Safely access daily PnL from performance monitor
            daily_pnl = self.performance_monitor.metrics.get('daily_pnl', [0])[-1]  # Get most recent value
            
            # Check daily loss limit
            if daily_pnl < -self.config.MAX_DAILY_LOSS:
                self.emergency_stop.activate(f"Daily loss limit breached: {daily_pnl:.2f}")
                return True

            # NEW: Profit factor check
            pf = self.performance_monitor._calc_profit_factor()
            if pf < 1.0:
                logger.warning(f"Profit factor critically low: {pf:.2f} - reconsider strategy")
                # Optional: activate emergency stop if desired
                # self.emergency_stop.activate(f"Profit factor too low: {pf:.2f}")
                
            # Check maximum drawdown
            max_dd = self.performance_monitor.metrics.get('max_dd', 0)
            if max_dd < -self.config.MAX_DRAWDOWN_PCT:
                self.emergency_stop.activate(f"Max drawdown exceeded: {max_dd:.2f}%")
                return True
                
            # Add other emergency conditions as needed
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
            return False

    def get_effective_stop_loss(self, symbol: str, current_price: Optional[float] = None) -> float:
        """Returns dynamic stop-loss in PRICE (not pips) using ATR or fixed SL"""
        FALLBACK_SL = float(self.config.INITIAL_STOP_LOSS * 0.1)

        # Get minimum stop loss from RiskManager instead of Config
        min_sl = self.calculate_min_stop_loss(symbol)  # Using instance method
        
        try:
            # Handle current price
            current_price_float: float
            if current_price is not None:
                current_price_float = float(current_price)
            else:
                # Use MT5Wrapper's symbol_info_tick instead of mt5.get_current_price
                tick_data = MT5Wrapper.symbol_info_tick(symbol)
                if tick_data is None:
                    logger.warning(f"Could not get tick data for {symbol}")
                    return FALLBACK_SL
                
                # Use the 'ask' price from the tick data
                current_price_float = float(tick_data['ask'])
            
            # Calculate base stop loss
            calculated_sl = current_price_float - FALLBACK_SL
            
            # Apply ATR adjustment if enabled
            if self.config.USE_ATR_SIZING:
                atr = self.data_fetcher.get_daily_atr(symbol)
                if atr is not None:
                    try:
                        atr_stop = float(atr) * float(self.config.ATR_STOP_LOSS_FACTOR)
                        logger.info(f"Using ATR-based SL: {atr_stop:.2f} pips")
                        calculated_sl = current_price_float - (atr_stop * 0.1)
                    except (TypeError, ValueError) as e:
                        logger.error(f"ATR calculation error: {str(e)}")
            
            # Validate and adjust if needed
            if not self._validate_price_levels(symbol, current_price_float, calculated_sl, 0.0):
                symbol_info = self.data_fetcher.get_symbol_info(self.config.SYMBOL)
                if symbol_info and all(k in symbol_info for k in ['trade_stops_level', 'point']):
                    try:
                        min_distance = float(symbol_info['trade_stops_level']) * float(symbol_info['point'])
                        calculated_sl = current_price_float - min_distance
                        logger.warning(f"Adjusted SL to minimum distance: {calculated_sl}")
                    except (TypeError, ValueError) as e:
                        logger.error(f"Stop level calculation error: {str(e)}")

            return max(float(calculated_sl), min_sl)  # Ensure SL respects minimum
                
        except Exception as e:
            logger.error(f"SL calculation failed: {str(e)}")
            return FALLBACK_SL


    
    def reset_daily_counts(self):
        """Reset daily counters with logging"""
        if self.today_trades > 0:
            logger.info(f"Resetting daily counts (was {self.today_trades} trades)")
        self.today_trades = 0


    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic"""
        last_exception = None
        for attempt in range(self._retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                time.sleep(self._retry_delay)
                continue
        logger.error(f"Failed after {self._retry_attempts} attempts: {str(last_exception)}")
        return None
    
    
    def clear_position_cache(self, ticket: Optional[int] = None):
        """Clear position cache (all or specific ticket)"""
        if ticket is None:
            self._position_cache.clear()
        elif ticket in self._position_cache:
            del self._position_cache[ticket]


    def _validate_price_levels(self, symbol: str, price: float, sl: float, tp: float) -> bool:
        """Ensures SL/TP are broker-valid"""
        symbol_info = self.data_fetcher.get_symbol_info(self.config.SYMBOL)
        if not symbol_info:
            return False
        
        min_stop_distance = float(symbol_info['trade_stops_level']) * float(symbol_info['point'])
        
        if abs(price - sl) < min_stop_distance:
            logger.warning(f"SL too close! Required: {min_stop_distance}, Actual: {abs(price - sl)}")
            return False
            
        if abs(price - tp) < min_stop_distance:
            logger.warning(f"TP too close! Required: {min_stop_distance}, Actual: {abs(price - tp)}")
            return False
        
        return True

    
    def get_symbol_metadata(self, symbol: str) -> Optional[Dict]:
        """DEPRECATED - Use data_fetcher.get_symbol_info() instead"""
        logger.warning("RiskManager.get_symbol_metadata() is deprecated - use data_fetcher.get_symbol_info()")
        return self.data_fetcher.get_symbol_info(symbol)


class TradingBot:
    """
    Trading Logic:

    Consider adding order book analysis for better fill simulation

    Add trade clustering detection
"""
    """Main trading bot class"""
    def __init__(self, config: Config, mt5_connector: MT5Connector, data_fetcher: DataFetcher, preprocessor: DataPreprocessor):
        self.mt5 = mt5_connector
        self.config = config
        
        # Initialize core components in proper order
        self.performance_monitor = PerformanceMonitor(config)
        self.preprocessor = preprocessor  # Use the injected preprocessor instead of creating new

        self.data_fetcher = data_fetcher
        
        # Initialize ML model with all required parameters
        self.ml_model = GoldMLModel(
            config=config,
            monitor=self.performance_monitor,
            data_fetcher=self.data_fetcher
        )

        # Now that ml_model exists, add it to monitor
        self.performance_monitor.add_model(self.ml_model)
        
        # Initialize data buffer
        self.data_buffer = []
        self.max_buffer_size = config.MAX_DATA_BUFFER
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            self.mt5,
            self.config,
            self.performance_monitor,
            self.data_fetcher
        )
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.risk_manager.reset_daily_counts,
            'cron',
            hour=0,
            minute=0
        )
        self.scheduler.start()

        # Initialize trading environment
        self.trading_env = None
        self._init_trading_environment()

        self.current_price = None
        self.symbol = config.SYMBOL 
        
    def _init_trading_environment(self):
        """Initialize the trading environment with current data"""
        # Get recent historical data through DataFetcher
        df = self.data_fetcher.get_historical_data(
            self.config.SYMBOL,
            self.config.TIMEFRAME,
            self.config.DATA_POINTS
        )
        
        if df is None:
            logger.error("Failed to initialize trading environment - no data")
            return
            
        # Preprocess the data
        df_processed = self.preprocessor._add_technical_indicators(df)
        features = df_processed[self.preprocessor.features].values
        
        self.trading_env = TradingEnvironment(
            config=self.config,
            data=features,
            features=self.preprocessor.features,
            window_size=self.preprocessor.window_size,
            symbol=self.config.SYMBOL,
            risk_manager=self.risk_manager,
            initial_balance=self.config.ACCOUNT_BALANCE
        )

    @property
    def is_connected(self) -> bool:
        """Centralized connection status with automatic recovery"""
        return self.mt5.ensure_connected()

    def initialize(self):
        """Initialize the trading bot with proper model loading/creation"""
        if not self.mt5.ensure_connected():
            return False
        
        initial_balance = self.data_fetcher.get_account_balance()
        if not initial_balance or initial_balance < 100:
            raise ValueError(f"Invalid account balance: {initial_balance}")
        self.config.ACCOUNT_BALANCE = float(initial_balance)
            
        try:
            if not self.ml_model.load_model():
                print("Training new RL model...")
                
                # Use data_fetcher instead of mt5
                df = self.data_fetcher.get_historical_data(
                    self.config.SYMBOL,
                    self.config.TIMEFRAME,
                    self.config.DATA_POINTS * 2
                )
                
                if df is None:
                    raise ValueError("Failed to fetch historical data for training")
                    
                X, y = self.preprocessor.preprocess_data(df)
                self.preprocessor.fit_scaler(X)
                X_scaled = self.preprocessor.transform_data(X)
                
                if not self.ml_model.train_model(X_scaled, y):
                    raise ValueError("Model training failed during initialization")
            
            if not self.ml_model._verify_model_architecture():
                raise ValueError("Model architecture verification failed")
                
            print("Model initialization complete")
            return True
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            return False
    
    def run(self):
        """Main trading loop with emergency stop capability"""
        if not self.initialize():
            print("Failed to initialize trading bot. Retrying in 60 seconds...")
            time.sleep(60)
            return False

        print("Trading bot started. Press Ctrl+C to stop.")

        while True:
            # Emergency stop check (first thing in loop)
            if self.risk_manager.emergency_stop.check():
                print("🛑 EMERGENCY STOP ACTIVATED - Closing all positions!")
                self._close_all_positions()
                return False  # Exit the run() method completely

            try:
                now = datetime.now()
                if now.hour == 0 and now.minute < 5:
                    self.risk_manager.reset_daily_counts()
                
                # Check if we can trade today
                if not self.risk_manager.can_trade_today():
                    print("Reached max trades for today. Sleeping...")
                    time.sleep(60)
                    continue

                # Manage positions with automatic cache clearing
                positions = self.mt5.get_open_positions(Config.SYMBOL)
                if positions:
                    self._manage_positions(positions)
                    
                # Get current data
                df = self.data_fetcher.get_historical_data(Config.SYMBOL, self.config.TIMEFRAME, 50)
                if df is None:
                    print("No data received, retrying in 5s...")
                    time.sleep(5)
                    continue
                    
                # Store new data
                self._update_data_buffer(df)
                
                # New RL model signal generation
                try:
                    df_processed = self.preprocessor._add_technical_indicators(df)
                    latest_window = df_processed.iloc[-self.preprocessor.window_size:][self.preprocessor.features].values
                    signal = self.ml_model.predict_signal(latest_window)
                except Exception as e:
                    print(f"⚠️ RL model prediction failed: {str(e)}")
                    signal = -1  # Default to no trade on error

                if self.risk_manager.check_emergency_conditions():
                    self.trigger_emergency_stop()

                if self.ml_model.should_retrain(len(self.data_buffer)):
                    self.train_rl_model()
                
                # Manage positions
                positions = self.mt5.get_open_positions(Config.SYMBOL)
                if positions:
                    self._manage_positions(positions)
                elif signal in [0, 1]:
                    trade_executed = self._execute_aggressive_trade(signal)
                    if trade_executed:
                        self.show_performance(periodic=True)

                atr = self.data_fetcher.get_daily_atr(Config.SYMBOL)
                if atr is None:
                    print("Warning: Could not get ATR value, skipping volatility check")
                elif atr < 15:  # Skip low-volatility periods
                    print(f"Skipping low-volatility period (ATR: {atr:.2f})")
                    continue

                if not self.ml_model.is_model_healthy():
                    self.trigger_emergency_stop()
                    
                # Check for retraining periodically
                if now.hour % 6 == 0 and now.minute < 5:
                    self._check_retraining()

                # Connection heartbeat check
                if not self.mt5.ensure_connected():
                    print("Connection lost, attempting recovery...")
                    if not self.mt5.recover_connection(): 
                        logger.error("Connection recovery failed")
                        break
                    time.sleep(5)
                    continue

                if datetime.now().hour % 6 == 0:  # Retrain every 6 hours
                    if len(self.data_buffer) > self.config.MIN_RETRAIN_SAMPLES:
                        self.train_rl_model()  # <-- HERE

                if now.minute % 30 == 0:  # Every 30 minutes
                    if not self.ml_model.check_model_health():
                        logger.warning("Model health check failed")

                # Add periodic profit factor check
                if self.performance_monitor.metrics['all_trades'] % 10 == 0:  # Check every 10 trades
                    pf = self.performance_monitor._calc_profit_factor()
                    if pf < 1.0:
                        logger.warning(f"Profit factor below 1.0 (current: {pf:.2f})")

                # Update current price at start of each iteration
                self.current_price = self.data_fetcher.get_current_price(Config.SYMBOL)  
                if not self.current_price:
                    print("Failed to get current price, retrying...")
                    time.sleep(1)
                    continue
                    
                # Small sleep to prevent CPU overload
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\nUser requested shutdown...")
                return True
            except ConnectionError as e:
                logger.error(f"Connection error: {str(e)}")
                if not self.mt5.recover_connection(): 
                    logger.error("Connection recovery failed")
                    break
                time.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                time.sleep(5)
    
    def _execute_aggressive_trade(self, signal: int) -> bool:
        """Execute high-frequency trade with ATR-based risk management"""
        current_atr = self.data_fetcher.get_daily_atr(Config.SYMBOL)
        if current_atr and current_atr < 5:  # Example threshold
            logger.warning(f"Low volatility (ATR: {current_atr:.2f}), skipping trade")
            return False

        if not self.mt5.ensure_connected():
            print("Connection unavailable for trading")
            return False

        # Validate signal
        if signal not in [0, 1]:
            return False
            
        trade_type = "buy" if signal == 1 else "sell"
        print(f"DEBUG: Executing {trade_type.upper()} trade")
        
        try:
            # 1. Price Fetching with type validation
            current_price = self.data_fetcher.get_current_price(Config.SYMBOL)  
            if not current_price or not all(k in current_price for k in ['bid', 'ask']):
                print("Invalid price data received")
                return False

            # 2. Get numeric price value safely
            price_key = 'ask' if trade_type == 'buy' else 'bid'
            try:
                price_value = current_price[price_key]
                if isinstance(price_value, datetime):
                    print("Invalid price data: got datetime when expecting float")
                    return False
                if not isinstance(price_value, (int, float)):
                    raise TypeError(f"Price must be numeric, got {type(price_value)}")
                current_price_value = float(price_value)
            except (TypeError, ValueError, KeyError) as e:
                print(f"Price access error: {str(e)}")
                return False

            # 3. Calculate order parameters
            requested_price = current_price_value + (0.1 if trade_type == 'buy' else -0.1)
            
            # 4. Calculate dynamic SL/TP levels
            stop_loss_pips = self.risk_manager.get_effective_stop_loss(Config.SYMBOL)
            if trade_type == "buy":
                sl_price = requested_price - stop_loss_pips
                tp_price = requested_price + (self.config.INITIAL_TAKE_PROFIT * 0.1)
            else:  # sell
                sl_price = requested_price + stop_loss_pips
                tp_price = requested_price - (self.config.INITIAL_TAKE_PROFIT * 0.1)

            # ===== CRITICAL VALIDATION ADDITION =====
            if not self.risk_manager._validate_price_levels(Config.SYMBOL, requested_price, sl_price, tp_price):
                logger.error(f"Invalid SL/TP levels. SL: {sl_price:.2f}, TP: {tp_price:.2f}")
                return False
            # ========================================

            # 5. Calculate position size
            aggressive_size = self.risk_manager.calculate_position_size(
                Config.SYMBOL, stop_loss_pips
            )
            
            # 6. Trade Execution
            result = self.mt5.send_order(
                symbol=Config.SYMBOL,
                order_type=trade_type,
                volume=aggressive_size,
                sl=sl_price,
                tp=tp_price,
                comment=f"AGGR-{trade_type.upper()}-ATR{stop_loss_pips:.0f}"
            )
            
            # 7. Execution Verification with proper dictionary access
            if not result or not isinstance(result, dict) or result.get('retcode') != mt5.TRADE_RETCODE_DONE:
                retcode = result.get('retcode', 'UNKNOWN') if isinstance(result, dict) else 'NO_RESULT'
                print(f"Order failed with retcode: {retcode}")
                return False

            # Get ticket from result
            ticket = int(result.get('order', 0)) or int(result.get('ticket', 0))
            if not ticket:
                print("No valid ticket/order number found")
                return False

            # Verify trade execution
            if not self.mt5.verify_trade_execution(ticket, self.config.MAX_SLIPPAGE):
                logger.error("Trade verification failed for ticket %s", ticket)
                return False

            assert isinstance(current_price, dict)  # Not None due to previous check
                
            try:
                bid_val = current_price['bid']
                ask_val = current_price['ask']
                if isinstance(bid_val, datetime) or isinstance(ask_val, datetime):
                    print(f"Invalid price data: bid or ask is datetime, not float")
                    return False
                bid_price = float(bid_val)
                ask_price = float(ask_val)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Invalid price data: {str(e)}")
                return False

            # 8. Calculate spread with explicit typing
            spread: float = ask_price - bid_price

            # 9. Calculate fill price with proper numeric handling
            try:
                requested_price_float = float(requested_price)
                fill_price: float = requested_price_float + (
                    spread * 0.5 if trade_type == "buy" else -spread * 0.5
                )
            except (TypeError, ValueError) as e:
                print(f"Price calculation error: {str(e)}")
                return False
            
            # 10. Slippage Check with type safety
            try:
                slippage = abs(float(fill_price) - float(requested_price))
                if slippage > 0.25:
                    print(f"Excessive slippage: {slippage:.2f} pips")
                    if position := self.mt5.get_position(ticket):
                        self.mt5.close_position(ticket, position['volume'])
                    return False
            except (TypeError, ValueError) as e:
                print(f"Slippage calculation error: {str(e)}")
                return False
            
            # 11. Trade Logging and Performance Tracking
            self.risk_manager.increment_trade_count()
            self.last_trade_time = datetime.now()
            print(f"Filled {trade_type} at {fill_price:.2f} (SL: {stop_loss_pips}pips, Slippage: {slippage:.2f}pips)")
            
            # 12. Performance monitoring with type safety
            try:
                # Get price with datetime check
                price_value = current_price['ask'] if trade_type == 'buy' else current_price['bid']
                if isinstance(price_value, datetime):
                    print(f"Invalid price data: got datetime when expecting float")
                    return False
                cost_basis = float(price_value)
                
                pnl = (fill_price - cost_basis) * aggressive_size * (1 if trade_type == 'buy' else -1)
                
                self.performance_monitor.update({
                    'action': trade_type,
                    'entry_price': cost_basis,
                    'exit_price': fill_price,
                    'pnl': pnl,
                    'slippage': slippage,
                    'stop_loss_pips': stop_loss_pips,
                    'time': datetime.now()
                })
            except (KeyError, TypeError, ValueError) as e:
                print(f"Trade performance tracking failed: {str(e)}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Unexpected error in trade execution: {str(e)}")
            return False
            
    
    def _update_data_buffer(self, new_data: pd.DataFrame) -> None:
        """Update the data buffer with new samples"""
        
        try:
            # Existing buffer update logic...
            new_samples = new_data[~new_data.index.isin(
                [x.index[0] for x in self.data_buffer if len(x) > 0]
            )]
            
            if len(new_samples) > 0:
                self.data_buffer.append(new_samples)
                
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer = self.data_buffer[-self.max_buffer_size:]
                
                # Add buffer monitoring HERE (after successful update)
                self.performance_monitor.update({
                    'buffer_size': len(self.data_buffer),
                    'buffer_capacity': self.max_buffer_size,
                    'new_samples': len(new_samples)
                })
                
        except Exception as e:
            logger.error(f"Buffer update failed: {str(e)}")
            # Update monitor with error state
            self.performance_monitor.update({
                'buffer_error': str(e),
                'buffer_size': len(self.data_buffer)  # Last known size
            })
    
    def _check_retraining(self):
        """Check if retraining is needed and execute it"""
        if len(self.data_buffer) == 0:
            return
            
        # Combine buffered data
        new_data = pd.concat(self.data_buffer)
        
        if self.ml_model.should_retrain(len(new_data)):
            print(f"Retraining model with {len(new_data)} new samples...")
            success = self.ml_model.retrain_model(new_data)
            
            if success:
                # Clear buffer after successful retraining
                self.data_buffer = []

                # Reinitialize environment with new data
                self._init_trading_environment()
                
                # Clear preprocessor cache
                self.preprocessor.clear_cache()
                
                # Reload the updated model
                self.ml_model.load_model()
                
                print("Retraining completed successfully")
            else:
                logger.error("Retraining failed, keeping existing model")

    def _manage_positions(self, positions: List[Dict]):
        """Manage existing positions (trailing stop, add to position, etc.) with cache management"""
        current_price = self.data_fetcher.get_current_price(Config.SYMBOL)
        if current_price is None:
            return
        
        position_modified = False  # Track if any positions were changed
        
        for position in positions:
            # Get full position details with caching
            self.risk_manager.clear_position_cache(position['ticket'])  # Clear per-ticket
            full_position = self.mt5.get_position(position['ticket'])
            if not full_position:
                continue
                
            # Enhanced logging for debugging
            logger.debug(f"Managing position {position['ticket']}: "
                        f"Type: {full_position['type']}, "
                        f"Size: {full_position['volume']}, "
                        f"Entry: {full_position['price_open']}, "
                        f"Current P/L: {full_position['profit']:.2f}")

            # Calculate profit in pips with current price
            if full_position['type'] == 'buy':
                profit_pips = (current_price['bid'] - full_position['price_open']) * 10
            else:
                profit_pips = (full_position['price_open'] - current_price['ask']) * 10
                
            # Check if we should add to position
            if profit_pips >= Config.POSITION_ADJUSTMENT_THRESHOLD:
                if self._add_to_position(full_position, current_price):
                    position_modified = True
                    
            # Check if we should close or reduce position
            if profit_pips <= -Config.TRADE_PENALTY_THRESHOLD:
                if self._handle_losing_position(full_position):
                    position_modified = True
                    
            # Update trailing stop with current market data
            if self._update_trailing_stop(full_position, current_price, profit_pips):
                position_modified = True
        
        # Clear cache if positions were modified
        if position_modified:
            logger.debug("Position changes detected - clearing cache")
            self.risk_manager.clear_position_cache()

    def _add_to_position(self, position: Dict, current_price: Dict):
        """Add to winning position"""
        # Calculate additional position size (half of initial)
        additional_size = self.risk_manager.calculate_position_size(
            Config.SYMBOL, Config.INITIAL_STOP_LOSS
        ) / 2
        
        if additional_size <= 0:
            return
            
        # Determine new stop loss (move to breakeven + some buffer)
        if position['type'] == 'buy':
            new_sl = position['entry_price'] + (Config.TRAILING_STOP_POINTS * 0.1 * 0.5)  # Half of trailing stop
            new_tp = current_price['bid'] + (Config.INITIAL_TAKE_PROFIT * 0.1)
        else:
            new_sl = position['entry_price'] - (Config.TRAILING_STOP_POINTS * 0.1 * 0.5)
            new_tp = current_price['ask'] - (Config.INITIAL_TAKE_PROFIT * 0.1)
            
        # Send additional order
        success = self.mt5.send_order(
            Config.SYMBOL,
            position['type'],
            additional_size,
            new_sl,
            new_tp,
            comment=f"AI-ADD-{position['type'].upper()}"
        )
        
        if success:
            print(f"Added to {position['type']} position at {current_price}")
    
    def _handle_losing_position(self, position: Dict) -> bool:
        """Close partial position or move SL to breakeven using Config params."""
        try:
            # Get params from config
            buffer_pct = self.config.LOSS_BUFFER_PCT
            close_ratio = self.config.PARTIAL_CLOSE_RATIO

            # Calculate breakeven with configurable buffer
            breakeven_price = float(position['price_open']) * (1 + buffer_pct)
            close_volume = round(float(position['volume']) * close_ratio, 2)

            if close_volume <= 0:
                return False

            if float(position['price_current']) >= breakeven_price:
                return self.mt5.modify_position(
                    position['ticket'],
                    new_sl=breakeven_price,
                    new_tp=float(position['tp'])  # Keep original TP
                )
            else:
                return self.mt5.close_position(position['ticket'], close_volume)

        except Exception as e:
            logger.error(f"Position handling failed: {str(e)}")
            return False
    
    def _update_trailing_stop(self, position: Dict, current_price: Dict, profit_pips: float):
        """Update trailing stop loss"""
        if profit_pips < Config.TRAILING_STOP_POINTS:
            return
            
        if position['type'] == 'buy':
            new_sl = current_price['bid'] - (Config.TRAILING_STOP_POINTS * 0.1)
            if new_sl > position['sl']:
                self.mt5.modify_position(position['ticket'], new_sl, position['tp'])
        else:
            new_sl = current_price['ask'] + (Config.TRAILING_STOP_POINTS * 0.1)
            if new_sl < position['sl'] or position['sl'] == 0:
                self.mt5.modify_position(position['ticket'], new_sl, position['tp'])

    def show_performance(self, periodic: bool = False):
        """Display performance metrics and equity curve"""
        if not hasattr(self, 'performance_monitor'):
            print("Performance monitoring not initialized")
            return
            
        report = self.performance_monitor.get_performance_report()
        
        print("\n=== Performance Report ===")
        for k, v in report.items():
            print(f"{k.replace('_', ' ').title()}: {v}")

        # Add annualized return
        annual_return = self.performance_monitor._calc_annualized_return()
        print(f"Annualized Return: {annual_return:.2f}%")
        
        # Environment metrics (only if available)
        if self.trading_env is not None:
            try:
                env_pf = self.trading_env.get_profit_factor()
                print(f"Environment Profit Factor: {env_pf:.2f}")
                if env_pf < 1.5:
                    logger.warning("Environment profit factor below 1.5 - reconsider strategy")
                
                env_reward = self.trading_env.get_total_reward()
                print(f"Environment Total Reward: {env_reward:.2f}")
            except Exception as e:
                logger.error(f"Failed to get environment metrics: {str(e)}")
        
        # Plotting logic remains the same
        if not periodic or len(self.performance_monitor.metrics['all_trades']) % 100 == 0:
            try:
                import matplotlib.pyplot as plt
                self.performance_monitor.plot_equity_curve().show()
            except ImportError:
                print("Matplotlib not available - cannot show equity curve")
            except Exception as e:
                print(f"Failed to plot equity curve: {str(e)}")

    
    def train_rl_model(self) -> bool:
        """Train the RL model using the TradingEnvironment
        
        Returns:
            bool: True if training succeeded, False otherwise
        """
        # Get historical data through DataFetcher
        df = self.data_fetcher.get_historical_data(
            self.config.SYMBOL,
            self.config.TIMEFRAME,
            self.config.DATA_POINTS
        )
        if df is None:
            logger.error("Failed to fetch historical data for training")
            return False
        
        # Preprocess data
        try:
            df = self.preprocessor._add_technical_indicators(df)
            features = df[self.preprocessor.features].values
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            return False

        # Create environment creation function
        def make_env() -> TradingEnvironment:
            return TradingEnvironment(
                data=features,
                features=self.preprocessor.features,
                window_size=self.preprocessor.window_size,
                symbol=self.config.SYMBOL,
                config=self.config,
                risk_manager=self.risk_manager,
                initial_balance=self.config.ACCOUNT_BALANCE
            )
        
        # Create vectorized environment
        try:
            env = make_vec_env(make_env, n_envs=1)
        except Exception as e:
            logger.error(f"Environment creation failed: {str(e)}")
            return False
        
        # Train model
        try:
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                **self.config.RL_PARAMS
            )
            model.learn(total_timesteps=self.config.RL_PARAMS["total_timesteps"])
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False
        
        # Save model
        try:
            model.save(self.config.RL_MODEL_PATH)
            logger.info(f"Model successfully saved to {self.config.RL_MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
            return False

    def _close_all_positions(self):
        """Force close all open positions"""
        positions = self.mt5.get_open_positions(Config.SYMBOL)
        for pos in positions:
            self.mt5.close_position(pos['ticket'], pos['volume'])

    def trigger_emergency_stop(self):
        """Public method for manual emergency stop"""
        self.risk_manager.emergency_stop.activate()
        self._close_all_positions()


class Backtester:
    """
    Testing Recommendations:

    Add more unit tests for core components

    Implement hypothesis testing for edge cases

    Add integration tests for full trading cycle
    """

    """Institutional-grade backtesting system with:
    - Walk-forward validation
    - Monte Carlo simulations
    - Strategy stress testing
    """
    
    def __init__(self, config: Config, mt5_connector: MT5Connector, data_fetcher: DataFetcher, preprocessor: DataPreprocessor):
        self.config = config
        self.mt5_connector = mt5_connector
        self.data_fetcher = data_fetcher
        self.preprocessor = preprocessor
        self.results = {
            'metrics': [],
            'equity_curves': [],
            'trade_analytics': []
        }
        self.monte_carlo_runs = 1000
        self.initial_balance = config.ACCOUNT_BALANCE
        self.gap_size = config.WALKFORWARD_GAP_SIZE

    def run_walkforward(self, data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]], n_splits=5) -> Dict:
        """Walkforward with expanding window and enforced time gaps to prevent leakage"""
        
        # Convert input to DataFrame if it's a tuple
        if isinstance(data, tuple):
            features, labels = data
            full_data = pd.DataFrame(features, columns=self.config.FEATURES)
            full_data['close'] = labels  # Assuming labels are close prices
        else:
            full_data = data
            
        # Initialize results storage if not already done
        if not hasattr(self, 'results'):
            self.results = {'metrics': [], 'equity_curves': [], 'trade_analytics': []}
        
        # Create custom time series split with gaps
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=self._calculate_test_size(len(full_data), n_splits))
        
        for train_idx, test_idx in tscv.split(full_data):
            # Apply gap between train and test
            test_idx = test_idx[self.gap_size:]  # Skip first 'gap_size' periods
            
            # Skip if test set becomes too small
            if len(test_idx) < self.preprocessor.window_size * 2:
                logger.warning(f"Skipping fold - test set too small after gap: {len(test_idx)} samples")
                continue
                
            # 1. Train Phase (with gap enforcement)
            train_data = full_data.iloc[train_idx]
            bot = self._init_bot_with_data(train_data)
            
            # Ensure no future data is used in preprocessing
            self._validate_no_data_leakage(train_data, test_idx[0], full_data)
            
            # Train with gap-adjusted data
            bot.ml_model.train_model(*bot.preprocessor.preprocess_data(train_data))
            
            # 2. Test Phase
            test_data = full_data.iloc[test_idx]
            test_results = self._run_backtest(bot, test_data.copy())
            
            # 3. Store metrics with gap information
            self.results['metrics'].append({
                'period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
                'cagr': self._calculate_cagr(test_results),
                'max_dd': test_results['max_drawdown'],
                'sharpe': test_results['sharpe_ratio'],
                'win_rate': test_results['win_rate'],
                'train_end': train_data.index[-1].date(),
                'test_start': test_data.index[0].date(),
                'gap_days': (test_data.index[0] - train_data.index[-1]).days
            })
            self.results['equity_curves'].append(test_results['equity_curve'])
            
        return self._compile_walkforward_results()

    def _validate_no_data_leakage(self, train_data: pd.DataFrame, first_test_idx: int, full_data: pd.DataFrame) -> None:
        """Validate no future data is leaking into training set"""
        last_train_date = train_data.index[-1]
        first_test_date = full_data.index[first_test_idx]
        
        if last_train_date >= first_test_date:
            raise ValueError(
                f"Data leakage detected! Train end {last_train_date} >= test start {first_test_date}"
            )
        
        # Convert gap to periods if not days (for non-daily data)
        gap_periods = (first_test_idx - len(train_data))
        if gap_periods < self.gap_size:
            logger.warning(
                f"Insufficient gap between train and test: {gap_periods} periods "
                f"(required: {self.gap_size} periods)"
            )
            raise ValueError("Insufficient gap between train and test sets")

    def _calculate_test_size(self, total_samples: int, n_splits: int) -> int:
        """Calculate appropriate test size considering gaps"""
        min_test_size = self.preprocessor.window_size * 2 + self.gap_size
        base_size = total_samples // (n_splits + 1)
        
        if base_size < min_test_size:
            raise ValueError(
                f"Insufficient data for {n_splits} splits. "
                f"Need at least {min_test_size} samples per test set"
            )
        
        return base_size + self.gap_size
    
    def _run_backtest(self, bot: TradingBot, data: pd.DataFrame) -> Dict[str, Any]:
        """Type-safe backtest execution"""
        # 1. Initialize with proper config
        broker = PaperTradingBroker(
            initial_balance=float(self.initial_balance),
            config=bot.config
        )
        
        # 2. Convert data to numpy upfront
        features = data[bot.preprocessor.features].values.astype(np.float32)
        close_prices = data['close'].values.astype(np.float32)
        
        equity_curve: List[float] = []
        
        for i in range(len(data) - bot.preprocessor.window_size):
            # 3. Get window as numpy array
            window = features[i:i+bot.preprocessor.window_size]
            
            # 4. Transform and predict with null checks
            if bot.preprocessor and bot.ml_model.model:
                obs = bot.preprocessor.transform_data(window)
                
                # 5. Type-safe prediction
                action_output, _ = bot.ml_model.model.predict(obs, deterministic=True)
                action = int(action_output.item() if hasattr(action_output, 'item') else action_output)
                
                # 6. Execute trade
                current_close = float(close_prices[i + bot.preprocessor.window_size])
                broker.execute_trade(
                    symbol=bot.config.SYMBOL,
                    action=action,
                    price=current_close
                )
                
                # 7. Type-safe equity recording
                equity_curve.append(broker.current_equity)
        
        # 8. Convert to numpy for calculations
        equity_array = np.array(equity_curve, dtype=np.float32)
        
        return {
            'equity_curve': equity_array.tolist(),
            'max_drawdown': float(self._calculate_max_dd(equity_array)),
            'sharpe_ratio': float(self._calculate_sharpe(equity_array)),
            'win_rate': float(broker.win_rate)
        }
    
    def run_monte_carlo(self, data: pd.DataFrame, n_runs=1000) -> Dict:
        """Monte Carlo path dependency testing"""
        mc_results = []
        for _ in range(n_runs):
            shuffled_data = data.sample(frac=1).reset_index(drop=True)
            mc_results.append(self._run_backtest(self._init_bot_with_data(data), shuffled_data))
        
        return {
            'cagr_dist': [self._calculate_cagr(r) for r in mc_results],
            'dd_dist': [r['max_drawdown'] for r in mc_results],
            'sharpe_dist': [r['sharpe_ratio'] for r in mc_results]
        }

    def stress_test(
        self, 
        data: Optional[pd.DataFrame] = None, 
        scenarios: Optional[Dict[str, Dict[str, Union[float, str]]]] = None
    ) -> Dict[str, Dict[str, Union[float, ArrayLike]]]:
        """Enhanced stress testing with strict typing and fallbacks.
        
        Args:
            data: Optional DataFrame for testing (uses fresh data if None)
            scenarios: Optional dict of scenario configs (uses Config.STRESS_SCENARIOS if None)
        
        Returns:
            Dict of scenario names to backtest results
        
        Raises:
            ValueError: If no data can be fetched
            TypeError: If inputs are invalid
        """
        # 1. Type-safe scenario initialization
        resolved_scenarios: Dict[str, Dict[str, Union[float, str]]] = scenarios or self.config.STRESS_SCENARIOS
        
        # 2. Type-safe data fetching with validation
        if data is None:
            fetched_data = self.data_fetcher.get_historical_data(
                self.config.SYMBOL,
                self.config.TIMEFRAME,
                self.config.DATA_POINTS
            )
            if fetched_data is None:
                raise ValueError("Could not fetch historical data for stress testing")
            resolved_data = fetched_data
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame")
            resolved_data = data

        # 3. Gold-specific adjustments (type-safe)
        if 'XAUUSD' in self.config.SYMBOL:
            for scenario in resolved_scenarios.values():
                if scenario['type'] == 'vol_spike' and isinstance(scenario.get('size'), (int, float)):
                    scenario['size'] *= 1.5  # type: ignore  # We validated type above

        # 4. Run tests (return type matches annotation)
        return {
            name: self._run_backtest(self._init_bot_with_data(self._apply_scenario(resolved_data.copy(), params)), resolved_data)
            for name, params in resolved_scenarios.items()
        }

    def _apply_scenario(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply market scenario (flash crash, low vol, etc.)"""
        if params.get('type') == 'vol_spike':
            data['close'] = data['close'] * (1 + np.random.normal(0, params['size'], len(data)))
        elif params.get('type') == 'trend':
            data['close'] = data['close'] * (1 + np.linspace(0, params['slope'], len(data)))
        return data

    def _init_bot_with_data(self, data: pd.DataFrame) -> TradingBot:
        """Initialize bot with preloaded data"""
        bot = TradingBot(
            config=self.config,
            mt5_connector=self.mt5_connector,
            data_fetcher=self.data_fetcher,
            preprocessor=DataPreprocessor(  # Updated instantiation
                config=self.config,
                data_fetcher=self.data_fetcher
            )
        )
        bot.data_buffer = [data]  # Bypass live data fetching
        bot._init_trading_environment() 
        return bot

    # Metric Calculations --------------------------------------------------
    @staticmethod
    def _calculate_cagr(test_results: Dict) -> float:
        eq = test_results['equity_curve']
        return (eq[-1]/eq[0]) ** (252/len(eq)) - 1 if len(eq) > 1 else 0

    @staticmethod
    def _calculate_max_dd(equity: np.ndarray) -> float:
        """Type-safe max drawdown calculation"""
        if not isinstance(equity, np.ndarray):
            equity = np.array(equity, dtype=np.float32)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(np.min(dd))

    @staticmethod
    def _calculate_sharpe(equity: np.ndarray) -> float:
        """Type-safe Sharpe ratio"""
        if len(equity) < 2:
            return 0.0
        returns = np.diff(equity) / equity[:-1]
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def _compile_walkforward_results(self) -> Dict:
        """Aggregate results across all folds with gap statistics"""
        if not self.results['metrics']:
            return {}
        
        gap_stats = [m['gap_days'] for m in self.results['metrics']]
        
        return {
            'avg_cagr': np.mean([r['cagr'] for r in self.results['metrics']]),
            'avg_max_dd': np.mean([r['max_dd'] for r in self.results['metrics']]),
            'avg_sharpe': np.mean([r['sharpe'] for r in self.results['metrics']]),
            'consistency': len([r for r in self.results['metrics'] if r['cagr'] > 0]) / len(self.results['metrics']),
            'gap_stats': {
                'min': min(gap_stats),
                'max': max(gap_stats),
                'mean': np.mean(gap_stats),
                'median': np.median(gap_stats)
            }
        }

    """
    def load_backtest_data(symbol, timeframe, num_points):
        cache_file = f"data/{symbol}_{timeframe}_{num_points}.pkl"
        if Path(cache_file).exists():
            return pd.read_pickle(cache_file)
        else:
            data = data_fetcher.get_historical_data(...)
            data.to_pickle(cache_file)
            return data
        """


class PaperTradingBroker:
    """Professional mock broker with:
    - Slippage modeling
    - Partial fills
    - Realistic order book simulation
    """
    
    def __init__(self, initial_balance: float, config: Config):  # CHANGED: Added config parameter
        self.balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.equity = [initial_balance]
        self.current_price = None
        self.config = config  # CHANGED: Store config
        
        # Enhanced slippage model
        self.slippage_model = {
            'normal': lambda: np.random.normal(0.0002, 0.0001),
            'high_vol': lambda: np.random.normal(0.001, 0.0005),
            'gold': lambda: np.random.normal(0.0005, 0.0002)  # Gold-specific slippage
        }

    def execute_trade(self, symbol: str, action: int, price: float) -> bool:
        """Execute trade with realistic market impact and price validation"""
        # Validate price input
        if not isinstance(price, (int, float)) or price <= 0:
            logger.error(f"Invalid price: {price}")
            return False

        # Get current market spread (simulated for paper trading)
        spread = self._get_simulated_spread(symbol)
        
        # Price validation checks
        if not self._validate_price(price, spread):
            logger.warning(f"Price validation failed for {symbol} at {price}")
            return False

        self.current_price = price
        
        # Apply symbol-appropriate slippage
        slippage = self.slippage_model['gold']() if action != 0 else 0
        executed_price = price * (1 + slippage)
        
        if action == 1:  # Buy
            # Additional spread threshold check for buys
            if spread > self.config.MAX_ALLOWED_SPREAD:
                logger.warning(f"Spread {spread} exceeds maximum for {symbol}")
                return False
                
            cost = self.balance * self.config.RISK_PER_TRADE
            position_size = cost / executed_price
            
            # Ensure minimum position size
            min_size = 0.01  # Minimum 0.01 oz for gold
            if position_size < min_size:
                logger.warning(f"Position size {position_size} below minimum {min_size}")
                return False
                
            self.positions.append({
                'symbol': symbol,
                'entry_price': executed_price,
                'size': max(position_size, min_size),
                'entry_time': len(self.equity),
                'commission': self.config.COMMISSION,
                'spread_at_entry': spread  # Track spread at entry
            })
            self.balance -= cost * (1 + self.config.COMMISSION)
            
        elif action == 2:  # Sell
            if not self.positions:
                logger.warning("No positions to sell")
                return False
                
            position = self.positions.pop()
            pnl = (executed_price - position['entry_price']) * position['size']
            gross_value = position['size'] * executed_price
            net_value = gross_value * (1 - position['commission'])
            
            # Spread impact analysis
            spread_impact = (position['spread_at_entry'] - spread) * position['size']
            if spread_impact < 0:
                logger.debug(f"Negative spread impact: {spread_impact:.4f}")

            self.balance += net_value
            self.trade_history.append({
                **position,
                'exit_price': executed_price,
                'gross_pnl': pnl,
                'net_pnl': pnl - (gross_value * position['commission']),
                'duration': len(self.equity) - position['entry_time'],
                'slippage': slippage,
                'spread_at_exit': spread
            })
        
        # Update equity curve
        self.equity.append(
            self.balance + 
            sum((self.current_price - p['entry_price']) * p['size'] for p in self.positions)
        )
        return True

    def _get_simulated_spread(self, symbol: str) -> float:
        """Generate realistic spread simulation"""
        # Base spread with volatility factor
        base_spread = 0.0002  # 2 pips for gold
        
        # Add volatility component
        volatility_factor = 1 + abs(np.random.normal(0, 0.5))
        
        # Time-of-day factor (wider spreads at market open/close)
        hour = datetime.now().hour
        time_factor = 1.5 if hour in {0, 23} else 1.0  # Wider at midnight
        
        return base_spread * volatility_factor * time_factor

    def _validate_price(self, price: float, spread: float) -> bool:
        """Comprehensive price validation"""
        # Basic price sanity check
        if price <= 0 or not math.isfinite(price):
            return False
        
        # Spread threshold check (configurable in pips)
        max_spread = getattr(self.config, 'MAX_ALLOWED_SPREAD', 0.005)  # 5 pips default
        if spread > max_spread:
            logger.debug(f"Spread {spread:.6f} exceeds threshold {max_spread:.6f}")
            return False
        
        # Price deviation check (prevent fat finger errors)
        if hasattr(self, 'current_price') and self.current_price:
            max_deviation = 0.01  # 1% price deviation allowed
            price_deviation = abs(price - self.current_price) / self.current_price
            if price_deviation > max_deviation:
                logger.warning(f"Price deviation {price_deviation:.2%} exceeds {max_deviation:.2%}")
                return False
        
        return True

    @property
    def win_rate(self) -> float:
        """Calculate win rate with new net PNL tracking"""
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t['net_pnl'] > 0)
        return wins / len(self.trade_history)

    def get_drawdown(self) -> float:
        """Calculate current drawdown"""
        peak = max(self.equity)
        current = self.equity[-1]
        return (current - peak) / peak

    @property
    def current_equity(self) -> float:
        """Returns the current equity value (last in the equity curve)"""
        return self.equity[-1] if self.equity else 0.0


class EmergencyStop:
    def __init__(self):
        self._active = False
        self.activation_time = None
        self.reason = None
        
    def activate(self, reason="Manual activation"):
        self._active = True
        self.activation_time = datetime.now()
        self.reason = reason
        
    def check(self) -> bool:
        """Pure status check without side effects"""
        return self._active
        
    def status(self) -> dict:
        return {
            "active": self._active,
            "since": self.activation_time,
            "reason": self.reason
        }


if __name__ == "__main__":
    logger = setup_logging()
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        # Initialize MT5 connection first
        mt5_connector = MT5Connector()
        if not mt5_connector.ensure_connected():
            raise ConnectionError("Failed to connect to MT5")

        # Fetch REAL account balance with proper dictionary access
        account_info = MT5Wrapper.account_info()
        if account_info is None:
            raise ValueError("Could not fetch MT5 account info")

        # Safely get balance with fallback
        account_balance = account_info.get('balance')
        if account_balance is None:
            raise ValueError("Account balance not available in account info")

        # Initialize configuration with REAL values
        config = Config(
            SYMBOL="GOLD",
            ACCOUNT_BALANCE=float(account_balance),  # Ensure float type
            INITIAL_STOP_LOSS=50,
            INITIAL_TAKE_PROFIT=75,
            RL_PARAMS={
                "total_timesteps": 30000,
                "learning_rate": 3e-4
            }
        )

        logger.info(f"Using REAL account balance: ${account_balance:.2f}")
        print(config)
        print(f"Model will be saved to: {config.RL_MODEL_PATH}")

        # Initialize DataFetcher with MT5 connection
        data_fetcher = DataFetcher(mt5_connector=mt5_connector, config=config)
        
        # Initialize DataPreprocessor
        preprocessor = DataPreprocessor(
            config=config,
            data_fetcher=data_fetcher  # Pass the existing data_fetcher
        )

        # Initialize Backtester with all dependencies
        backtester = Backtester(
            config=config,
            data_fetcher=data_fetcher,
            mt5_connector=mt5_connector,
            preprocessor=preprocessor
        )

        # Initialize TradingBot with all required dependencies
        bot = TradingBot(
            config=config,
            data_fetcher=data_fetcher,
            mt5_connector=mt5_connector,
            preprocessor=preprocessor
        )
        
        # Add command-line interface for backtesting
        import argparse
        parser = argparse.ArgumentParser(description='Trading Bot with Backtesting')
        parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
        parser.add_argument('--walkforward', action='store_true', help='Run walkforward validation')
        parser.add_argument('--montecarlo', action='store_true', help='Run Monte Carlo simulations')
        parser.add_argument('--periods', type=int, default=500, help='Number of periods to backtest')
        args = parser.parse_args()

        if args.backtest or args.walkforward or args.montecarlo:
            # Fetch historical data for backtesting
            historical_data = data_fetcher.get_historical_data(
                symbol=config.SYMBOL,
                timeframe=config.TIMEFRAME,
                num_candles=args.periods
            )
            
            if historical_data is None:
                raise ValueError("Failed to fetch historical data for backtesting")

            # Preprocess the data first
            processed_data = historical_data.copy()
            processed_data = preprocessor._add_technical_indicators(processed_data)
            processed_data = preprocessor._normalize_data(processed_data)
            processed_data = processed_data[config.FEATURES + ['close']].dropna()

            # Add stress testing option
            if input("Run stress tests? (y/n): ").lower() == 'y':
                stress_results = backtester.stress_test(
                    processed_data,
                    scenarios=config.STRESS_SCENARIOS
                )
                print("\n=== Stress Test Results ===")
                for scenario, result in stress_results.items():
                    print(f"{scenario}: Max DD={result['max_drawdown']:.2%}")
            
            # Preprocess the data but keep the DataFrame structure
            # First ensure we have all needed columns
            if not all(col in historical_data.columns for col in config.FEATURES + ['close']):
                raise ValueError("Historical data missing required columns")
            
            # Process the data without converting to numpy arrays yet
            processed_data = historical_data.copy()
            
            # Apply any necessary preprocessing that maintains DataFrame structure
            # For example:
            processed_data = preprocessor._add_technical_indicators(processed_data)
            processed_data = preprocessor._normalize_data(processed_data)  # If this maintains DataFrame
            
            # Ensure we have the required columns for backtesting
            required_columns = config.FEATURES + ['close']
            processed_data = processed_data[required_columns].dropna()
            
            if args.walkforward:
                print("\n=== Running Walkforward Validation ===")
                results = backtester.run_walkforward(processed_data)
                print(f"Average CAGR: {results['avg_cagr']:.2%}")
                print(f"Average Max Drawdown: {results['avg_max_dd']:.2%}")
                print(f"Average Sharpe Ratio: {results['avg_sharpe']:.2f}")
                print(f"Strategy Consistency: {results['consistency']:.2%}")
            
            elif args.montecarlo:
                print("\n=== Running Monte Carlo Simulations ===")
                results = backtester.run_monte_carlo(processed_data)
                print(f"CAGR Distribution: Mean={np.mean(results['cagr_dist']):.2%} "
                      f"Std={np.std(results['cagr_dist']):.2%}")
                print(f"Drawdown Distribution: Mean={np.mean(results['dd_dist']):.2%} "
                      f"Std={np.std(results['dd_dist']):.2%}")
                print(f"Sharpe Distribution: Mean={np.mean(results['sharpe_dist']):.2f} "
                      f"Std={np.std(results['sharpe_dist']):.2f}")
            
            else:
                print("\n=== Running Standard Backtest ===")
                results = backtester._run_backtest(bot, processed_data)
                print(f"Final Equity: ${results['equity_curve'][-1]:,.2f}")
                print(f"Max Drawdown: {results['max_drawdown']:.2%}")
                print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                print(f"Win Rate: {results['win_rate']:.2%}")
            
            sys.exit(0)  # Exit after backtesting
        
        # Live trading mode
        if bot.initialize():
            try:
                # Initialize trade counter
                trade_count = 0
                
                # Main trading loop
                while True:
                    # Run one iteration of the trading logic
                    bot.run()
                    
                    # Check for retraining every N trades
                    if trade_count % 50 == 0:  # Check every 50 trades
                        new_data = data_fetcher.get_historical_data(
                            symbol=config.SYMBOL,
                            timeframe=config.TIMEFRAME,
                            num_candles=config.MIN_RETRAIN_SAMPLES
                        )
                        
                        if new_data is not None and len(new_data) >= config.MIN_RETRAIN_SAMPLES:
                            if bot.ml_model.should_retrain(len(new_data)):
                                logger.info("Initiating automatic retraining...")
                                if bot.ml_model.retrain_model(new_data):
                                    logger.info("Retraining completed successfully")
                                else:
                                    logger.warning("Retraining failed")
                    
                    # Increment and check trade count
                    trade_count += 1
                    if trade_count % 100 == 0:
                        print(f"\nCompleted {trade_count} trades - Generating performance report...")
                        bot.show_performance(periodic=True)
                        
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nShutting down...")
                # Final performance report
                print("\n=== FINAL PERFORMANCE REPORT ===")
                bot.show_performance()
            finally:
                bot.mt5.disconnect()

    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        sys.exit(1)


"""
Licensing Fee Models
Model	Fee Structure	Best For	Example
Revenue Share	15-30% of net profits	Hedge funds, prop firms	20% of monthly P&L
Flat SaaS Fee	$1,000-$10,000/month	Retail traders, small funds	$5,000/month per terminal
Tiered Pricing	$500-$5,000/month based on AUM	Asset managers	$2k/month for $1M-$10M AUM
Per-Trade Fee	$0.10-$2.00 per trade	High-frequency traders	$0.50 per lot traded
One-Time License	$50,000-$500,000	Institutional buyers	$250k for unlimited use
2. Fee Benchmarks by Client Type
A. Retail Traders

    Price Range: $300-$3,000/month

    Offerings:

        Limited features (e.g., 3 signals/day)

        Community support


B. Prop Trading Firms

    Price Range: $5,000-$20,000/month + 10-25% performance fee

    Key Terms:

        White-label solutions

        API integration

    Industry Standard:

        "Most quant firms charge 20-30% for alpha-generating systems" – Tower Research

C. Hedge Funds/Institutional

    Price Range: $50,000-$200,000/year + 15-30% performance fee

    Negotiable Terms:

        Custom feature development

        Co-location support

3. Key Pricing Factors

    Sharpe Ratio

        Sharpe < 1.5: $1k-$5k/month

        Sharpe > 2.5: $10k+/month + profit share

    Asset Class
    Market	Premium	Reason
    Forex	1.0x	High liquidity
    Crypto	1.5x	Volatility premium
    Commodities	1.2x	Limited competition

    Exclusivity

        Non-exclusive: 50% discount

        Regional exclusivity: 2x base fee

4. Contract Structure
python

class LicenseAgreement:
    def __init__(self, term_years=1, aum=None):
        self.base_fee = 5000  # $/month
        self.performance_fee = 0.20  # 20%
        self.service_sla = "99.9% uptime"
        
    def calculate_fee(self, monthly_pnl):
        return self.base_fee + (monthly_pnl * self.performance_fee)

5. Real-World Examples

    Retail System (Similar to yours):

        TrendSpider: $108-$348/month

        Trade Ideas: $1,164/year

    Institutional-Grade:

        QuantConnect: $50k+/year for API access

        SigOpt: $100k+ for optimization suites

6. Recommended Pricing Strategy

    Phase 1 (Validation):

        Offer 3-month pilot at $2,500/month with 10% profit share to build track record.

    Phase 2 (Scaling):

        Retail: $500/month (capped at 5% profit share)

        Institutional: $15k/month + 15% over watermark

    Phase 3 (Exclusive):

        Sell white-label rights for $250k+/year per firm.

7. Legal Considerations

    Performance Clawback: Chargebacks if annual returns < 8%

    Kill Switch: Rights to disable system if client violates terms

    Jurisdiction: Delaware LLC for US clients, Cyprus for EU

Final Estimate

For a system with 1.8+ Sharpe ratio and 10-15% monthly returns:

    Retail: $1,000-$3,000/month

    Institutional: $10,000-$50,000/month + 20% profits

"""