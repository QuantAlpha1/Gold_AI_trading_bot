
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
# To create the environment: python -m venv goldAI_env
# goldAI_env\Scripts\activate

# Legal protection:
# Add a LICENCE file
# Include Terms of Use in README.md

#!/usr/bin/env python3
"""GoldAI - MetaTrader 5 Gold Trading Bot with Machine Learning with Stable Baselines3 (PPO) for RL"""

# ===================== SYSTEM & OS =====================
import sys
import os
api_key = os.getenv('MT5_API_KEY') # Not changed anywhere else yet!
from cryptography.fernet import Fernet
key = Fernet.generate_key()  # Store this securely, not implemented further yet
import argparse
from pathlib import Path
import warnings
import threading
from threading import Lock
from collections import defaultdict
from dataclasses import dataclass
import time
import re
import copy

# ===================== LOGGING =====================
import logging
from logging.handlers import TimedRotatingFileHandler

# ===================== DATA & MATH =====================
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import math
from decimal import Decimal, getcontext
getcontext().prec = 8

# ===================== DATETIME =====================
from datetime import datetime, timedelta
import pytz

# ===================== METATRADER =====================
import MetaTrader5 as mt5
default_path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# ===================== MACHINE LEARNING =====================
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ===================== REINFORCEMENT LEARNING =====================
import gymnasium as gym
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime, timedelta
from collections import deque
import hashlib

# ===================== STATISTICS =====================
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# ===================== TYPING =====================
from typing import (
    Tuple, Dict, Callable, TypedDict, Optional, 
    List, Any, Union, Literal, ClassVar
)
from typing_extensions import Annotated
from typing import TYPE_CHECKING, Any

from typing_extensions import Annotated


# ===================== VALIDATION =====================
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
    ConfigDict
)

# ===================== UTILITIES =====================
import random
from functools import lru_cache
from apscheduler.schedulers.background import BackgroundScheduler

# ===================== TESTING =====================
import pytest
from elso import MT5Wrapper

if TYPE_CHECKING:
    # Type stub for LIME Explanation when type checking
    class LimeExplanation:
        as_list: Callable[..., List[Tuple[str, float]]]
        show_in_notebook: Callable[..., None]
        local_exp: Dict[int, List[Tuple[int, float]]]
        # Add other methods you use
        
    LimeExplanationResult = LimeExplanation


def detect_clustering(trades: List[float]) -> bool:
    """Detects structural breaks in trade sequences using CUSUM of squares test."""
    diff_trades = np.diff(trades)
    _, p_values, _ = breaks_cusumolsresid(diff_trades)
    
    # Convert array of p-values to a single boolean (True if ANY break is significant)
    return bool(np.any(p_values < 0.01))  # Explicitly cast to bool

"""
If you need the test statistics for further analysis, modify the return statement:
python

test_stat, p_value, _ = breaks_cusumolsresid(diff_trades)
return test_stat, p_value < 0.01
"""


ObsType = npt.NDArray[np.float32]
ActType = int


def mt5_get(attr: str) -> Any:
    """Safely get MT5 attributes with type checking"""
    if hasattr(mt5, attr):
        return getattr(mt5, attr)
    raise AttributeError(f"MT5 has no attribute {attr}")

# Usage example:
account = mt5_get("account_info")()

PriceData = Dict[str, float]

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
            
            # 5. Result validation
            if hasattr(result, 'retcode') and result.retcode != success_retcode:
                delay = min(
                    BASE_DELAY * (2 ** attempt) * (1 + random.uniform(0, MAX_JITTER)),
                    MAX_SINGLE_DELAY
                )
                time.sleep(delay)
                attempt += 1
                continue
                
            return result
            
        except (ConnectionError, TimeoutError) as e:  # Removed MT5Wrapper.Error unless imported
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

# New, needs to be addded to the tree
class TradeSignal(BaseModel):
    symbol: Annotated[str, "GOLD"]
    entry: float
    stop_loss: float


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
    TIMEFRAME_M1 = 1      # 1 minute
    TIMEFRAME_M5 = 5      # 5 minutes
    TIMEFRAME_M15 = 15    # 15 minutes
    TIMEFRAME_M30 = 30    # 30 minutes
    TIMEFRAME_H1 = 60     # 1 hour
    TIMEFRAME_H4 = 240    # 4 hours
    TIMEFRAME_D1 = 1440   # 1 day
    TIMEFRAME_W1 = 10080  # 1 week
    TIMEFRAME_MN1 = 43200 # 1 month
    
    class Error(Exception):
        """Custom exception for MT5-related errors"""
        pass

    @staticmethod
    def initialize_with_validation(max_retries: int = 3, **kwargs) -> bool:
        for attempt in range(max_retries):
            try:
                if init_func := mt5_get("initialize"):
                    if bool(init_func(**kwargs)):
                        return True
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"MT5 init failed (attempt {attempt+1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise MT5Wrapper.Error(f"MT5 initialization failed after {max_retries} attempts")
        return False

        
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
                # If last_error is not available, provide a generic error message
                logger.error(f"Tick data failed for {symbol}.")
                raise MT5Wrapper.Error(f"MT5 error: Failed to get tick data for {symbol}")
                
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
    def order_send(request: dict) -> dict:
        """Safely execute MT5 order with complete validation."""
        # 1. Input validation
        required_keys = {
            'action': (int,),        # TRADE_ACTION_DEAL = 1
            'symbol': (str,),        # Trading symbol
            'volume': (float, int),  # Order size
            'type': (int,),          # ORDER_TYPE_BUY/SELL
            'price': (float, int),   # Execution price
            'deviation': (int,),     # 0-10
            'type_filling': (int,)   # ORDER_FILLING_FOK/IOC
        }

        for key, types in required_keys.items():
            if key not in request:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(request[key], types):
                raise TypeError(f"{key} must be {types}, got {type(request[key])}")

        # 2. Execute order
        try:
            # Proper MT5 attribute access
            result = mt5.order_send(request)  # type: ignore
            
            if result is None:
                # Correct last error access
                last_err = mt5.last_error()  # type: ignore
                raise MT5Wrapper.Error(f"MT5 returned None. Error: {last_err}")
                
            # Safe attribute extraction
            return {
                'retcode': int(result.retcode),
                'deal': int(result.deal),
                'order': int(result.order),
                'volume': float(result.volume),
                'price': float(result.price),
                'comment': str(result.comment)
            }
            
        except AttributeError as e:
            # Handle missing attributes in MT5 response
            last_err = mt5.last_error()  # type: ignore
            raise MT5Wrapper.Error(
                f"Invalid MT5 response structure: {str(e)}. Last error: {last_err}"
            )
        except Exception as e:
            # Catch-all for other errors
            last_err = mt5.last_error()  # type: ignore
            raise MT5Wrapper.Error(
                f"Order failed: {str(e)}. Last MT5 error: {last_err}"
            ) from e
    
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
    def account_info() -> Dict[str, Union[float, int]]:
        try:
            # Use mt5_get for safe access
            account_info_func = mt5_get("account_info")
            if not account_info_func:
                raise ValueError("MT5 'account_info' function not available")
                
            result = account_info_func()
            if not result:
                raise ValueError("Empty account info response")
                
            return {
                'balance': float(result.balance),
                'equity': float(result.equity),
                'margin': float(result.margin),
                'margin_free': float(result.margin_free),
                'margin_level': float(result.margin_level),
                'login': int(result.login)
            }
        except Exception as e:
            logger.error(f"Account info failed: {str(e)}")
            # Return default values instead of raising
            return {
                'balance': 0.0,
                'equity': 0.0,
                'margin': 0.0,
                'margin_free': 0.0,
                'margin_level': 0.0,
                'login': 0
            }
    
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
        try:
            total_func = mt5_get("symbols_total")
            result = total_func()
            if result is None:  # Explicitly handle None
                raise MT5Wrapper.Error("MT5 returned None for symbols_total")
            return int(result)
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

    @staticmethod
    def calculate_margin(trade_type: str, symbol: str, volume: float, price: float) -> Optional[float]:
        """Calculate required margin for a trade with comprehensive validation
        
        Args:
            trade_type: Must be either "buy" or "sell" (case-insensitive)
            symbol: Trading symbol (e.g., "EURUSD")
            volume: Trade volume (must be positive)
            price: Execution price (must be positive)
            
        Returns:
            float: Required margin if calculation succeeds
            None: If calculation fails or invalid inputs
            
        Raises:
            ValueError: For invalid input parameters
        """
        # Input validation
        if trade_type.lower() not in ("buy", "sell"):
            raise ValueError(f"Invalid trade_type: {trade_type}. Must be 'buy' or 'sell'")
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"Invalid symbol: {symbol}")
            return None
        if not isinstance(volume, (int, float)) or volume <= 0:
            logger.error(f"Invalid volume: {volume} (must be positive)")
            return None
        if not isinstance(price, (int, float)) or price <= 0:
            logger.error(f"Invalid price: {price} (must be positive)")
            return None

        try:
            # Safe function access
            calc_func = mt5_get("order_calc_margin")
            if calc_func is None:
                logger.error("MT5.order_calc_margin() not available - upgrade MetaTrader5 package")
                return None

            # Convert to MT5 constants with additional validation
            trade_type = trade_type.lower()
            if trade_type == "buy":
                action_type = MT5Wrapper.ORDER_TYPE_BUY
            elif trade_type == "sell":
                action_type = MT5Wrapper.ORDER_TYPE_SELL
            else:  # This should never happen due to earlier validation
                logger.critical("Programmer error: trade_type validation failed")
                return None

            # Execute calculation
            result = calc_func(
                action_type=action_type,
                symbol=symbol,
                volume=volume,
                price=price
            )

            # Validate and return result
            if result is None:
                last_err = mt5.last_error()  # type: ignore
                logger.error(f"Margin calculation returned None. MT5 error: {last_err}")
                return None
                
            return float(result)

        except Exception as e:
            logger.error(f"Margin calculation failed for {symbol} {trade_type}: {str(e)}", exc_info=True)
            return None


class Config(BaseModel):
    """Complete trading bot configuration with Pydantic V2 validation"""

    model_config = ConfigDict(
        strict=True,  # Rejects extra fields
        frozen=True,
        validate_assignment=True
    )
    
    # --- Core Validated Fields ---
    # Instrument and Account Settings
    SYMBOL: str = "GOLD"
    ACCOUNT_BALANCE: float = Field(..., gt=0, description="Current account balance in USD")
    RISK_PER_TRADE: float = Field(default=0.02, gt=0, le=0.05)
    INITIAL_STOP_LOSS: float = Field(default=100, gt=5, le=500)
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
    ATR_TRAILING_MULTIPLIER: float = Field(default=3.0, gt=0.5, le=5.0, 
                                        description="Multiplier for ATR-based trailing stops")
    TRAILING_STOP_ATR_MULTIPLIER: float = Field(
        default=3.0, 
        gt=1.0,  # Minimum 1x ATR
        le=5.0,  # Maximum 5x ATR
        description="Multiplier for ATR-based trailing stops (e.g. 3.0 = 3xATR)"
    )
    USE_ATR_SIZING: bool = True  # Dynamic SL enabled by default
    MIN_STOP_DISTANCE: float = 50  # Fallback if ATR too small (pips)
    ATR_STOP_LOSS_FACTOR: float = Field(default=1.5, gt=0.0, le=5.0)
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
    
    BASELINE_ATR = 10.0  # Symbol-specific typical ATR value
    DEFAULT_ATR = 5.0     # Fallback if live ATR unavailable
    PARTIAL_CLOSE_RATIO = 0.5  # Close 50% on adverse move

    MAX_CONSECUTIVE_LOSSES: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Number of consecutive losing trades before circuit breaker activates"
    )
    CIRCUIT_BREAKER_TIMEOUT: int = Field(
        default=3600,
        gt=300,
        le=86400,
        description="Cooling period in seconds after circuit breaker activates (1 hour default)"
    )
    LOSS_STREAK_LOOKBACK: int = Field(
        default=5,
        gt=1,
        le=20,
        description="Number of recent trades to analyze for loss streaks"
    )

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
    
    VOLATILITY_SETTINGS: Dict[str, Dict] = Field(
        default={
            'gold': {'baseline': 0.3, 'multiplier': 1.5},
            'xauusd': {'baseline': 0.3, 'multiplier': 1.5},
            'default': {'baseline': 0.2, 'multiplier': 1.0}
        },
        description="Volatility calibration parameters by symbol type"
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
    
    MIN_FILL_RATIO: float = Field(
        default=0.3,
        gt=0,
        le=1,
        description="Minimum acceptable fill ratio before warning"
    )
    
    MAX_FILL_RETRIES: int = Field(
        default=3,
        gt=0,
        description="Maximum attempts to get acceptable fill"
    )
    
    LIQUIDITY_PROFILES: Dict[str, Dict] = Field(
        default={
            'gold': {'base_fill': 0.85, 'size_sensitivity': 0.3},
            'forex': {'base_fill': 0.95, 'size_sensitivity': 0.1}
        }
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

    # Meta-learning configuration
    ENABLE_META_LEARNING: bool = True  # Set to False to disable meta-learning adaptation
    META_WINDOW: int = 100  # Number of candles for adaptation lookback
    META_SENSITIVITY: float = 0.3  # Aggressiveness of adaptation (0.1-0.5)
    
    # Debugging configuration
    DEBUG_MODE: bool = False  # Set to True for detailed debug logging
    DEBUG_LEVEL: str = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR
    LOG_SIGNALS: bool = True  # Log all trading signals

    META_LEARNING: Dict[str, Union[bool, int, float]] = Field(
        default={
            "ENABLED": True,
            "WINDOW": 200,       # candles
            "SENSITIVITY": 0.2
        },
        description="Meta-learning adaptation parameters"
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
    def validate_trailing_stop(self) -> 'Config':
        """Validate trailing stop parameters"""
        if not hasattr(self, 'TRAILING_STOP_ATR_MULTIPLIER'):
            return self
            
        if self.TRAILING_STOP_ATR_MULTIPLIER <= 0:
            raise ValueError('Trailing stop ATR multiplier must be positive')
        
        # Example: Warn if trailing stop is too tight
        if self.TRAILING_STOP_ATR_MULTIPLIER < 2.0:
            print(f"Warning: Trailing stop multiplier {self.TRAILING_STOP_ATR_MULTIPLIER} may be too tight")
        
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
    
    @field_validator('*')
    @classmethod
    def validate_all(cls, value, info: ValidationInfo):
        if info.field_name == 'SYMBOL' and not value.isupper():
            raise ValueError("Symbol must be uppercase")
        return value

    def __str__(self) -> str:
        # Build dynamic components
        risk_info = f"RISK={self.RISK_PER_TRADE*100}%"
        stop_info = f"SL={self.INITIAL_STOP_LOSS}pips"
        
        # Add ATR info if using ATR sizing
        atr_info = ""
        if self.USE_ATR_SIZING:
            atr_info = f", ATR={self.ATR_LOOKBACK}periods@{self.ATR_STOP_LOSS_FACTOR}x"
        
        # Add volatility settings if available
        vol_info = ""
        if hasattr(self, 'VOLATILITY_SETTINGS'):
            vol_info = f", VolProfile={self.VOLATILITY_SETTINGS.get(self.SYMBOL, {}).get('multiplier', 1.0)}x"
        
        # Simple timeframe mapping without calling MT5Wrapper
        timeframe_map = {
            1: "M1",
            5: "M5",
            15: "M15",
            30: "M30",
            60: "H1",
            240: "H4",
            1440: "D1",
            10080: "W1",
            43200: "MN1"
        }
        tf_name = timeframe_map.get(self.TIMEFRAME, f"Unknown({self.TIMEFRAME})")
        
        return (f"Config(SYMBOL={self.SYMBOL}, "
                f"{risk_info}, "
                f"{stop_info}"
                f"{atr_info}"
                f"{vol_info}, "
                f"TF={tf_name})")


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
        self._metrics_lock = threading.Lock()
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
        with self._metrics_lock:
            equity = np.array(self.metrics['equity_curve'])
        
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        current_dd = drawdown.min()  # Already negative
        
        with self._metrics_lock:
            # Compare absolute values since both are negative
            if abs(current_dd) > abs(self.metrics['max_dd']):
                self.metrics['max_dd'] = current_dd
    
    def _calc_annualized_return(self):
        """Calculate annualized return percentage"""
        with self._metrics_lock:
            if len(self.metrics['equity_curve']) < 2:
                return 0.0
            ec = self.metrics['equity_curve']
            days = len(self.metrics['daily_pnl']) or 1  # Prevent division by zero
        
        total_return = (ec[-1] / ec[0] - 1) * 100
        return ((1 + total_return/100) ** (252/days) - 1) * 100
    
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

    def prune_old_trades(self, max_trades: int = 10_000) -> None:
        """Limit memory usage by pruning old trades and maintaining consistency"""
        with self._metrics_lock:
            if len(self.metrics['trades']) <= max_trades:
                return
                
            # Prune trades and maintain related metrics
            self.metrics['trades'] = self.metrics['trades'][-max_trades:]
            
            # Rebuild derived metrics
            self.metrics['win_rate'] = [1 if t['pnl'] > 0 else 0 for t in self.metrics['trades']]
            
            # Rebuild equity curve from remaining trades
            self.metrics['equity_curve'] = [0.0]  # Reset starting equity
            for trade in self.metrics['trades']:
                self.metrics['equity_curve'].append(self.metrics['equity_curve'][-1] + trade['pnl'])
            
            # Force recalculation of other metrics
            self._update_sharpe()
            self._update_drawdown()

    def get_performance_report(self, model: Optional["GoldMLModel"] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report with warnings and enhanced metrics"""
        with self._metrics_lock:
            trades = list(self.metrics['trades'])
            metrics_snapshot = {
                'trades': list(self.metrics['trades']),
                'equity_curve': list(self.metrics['equity_curve']),
                # 'win_rate': list(self.metrics['win_rate']),
                'daily_pnl': list(self.metrics['daily_pnl']),
                'sharpe': self.metrics.get('sharpe'),
                'max_dd': self.metrics.get('max_dd', 0)
            }

        # Calculate core metrics
        profit_factor = IndicatorUtils.profit_factor(trades)
        total_trades = len(trades)
        sharpe = self.metrics.get('sharpe')
        drawdown = self.metrics.get('max_dd', 0)
        win_rate_value = self._calculate_win_rate(trades)  # Now accepts trades parameter
        win_rate_str = f"{win_rate_value * 100:.1f}%"
        daily_pnl = self.metrics.get('daily_pnl', [0])[-1]
        calmar_ratio = self._calc_calmar_ratio()
        sortino = self._calc_sortino_ratio()
        avg_trade_pnl = np.mean([t['pnl'] for t in self.metrics['trades']]) if total_trades > 0 else 0
        trade_stats = self.get_trade_statistics()

        # Calculate annualized return
        annualized_return = 0.0
        if len(metrics_snapshot['equity_curve']) >= 2:
            total_return = (metrics_snapshot['equity_curve'][-1] / metrics_snapshot['equity_curve'][0] - 1) * 100
            days_active = len(metrics_snapshot['daily_pnl']) or 1
            annualized_return = ((1 + total_return/100) ** (252/days_active) - 1) * 100

        # Calculate other metrics
        daily_pnl = metrics_snapshot['daily_pnl'][-1]['amount'] if metrics_snapshot['daily_pnl'] else 0
        calmar_ratio = self._calc_calmar_ratio()  # These methods will use the locked metrics
        sortino = self._calc_sortino_ratio()
        
        avg_trade_pnl = np.mean([t['pnl'] for t in metrics_snapshot['trades']]) if total_trades > 0 else 0
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

        # Build report dictionary
        report = {
            # Core metrics
            'annualized_return': f"{annualized_return:.2f}%",
            'sharpe_ratio': f"{metrics_snapshot['sharpe']:.2f}" if metrics_snapshot['sharpe'] is not None else "N/A",
            'max_drawdown': f"{metrics_snapshot['max_dd']:.2f}%",
            'daily_pnl': daily_pnl,
            'win_rate': win_rate_str,
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
            'last_retrain': getattr(self, 'last_retrain_time', "Never"),
        }

        # Add model-specific information if provided
        if model is not None:
            model_info = {
                '_version': getattr(model, '_version', 'N/A'),
                'model_metrics': model.calculate_metrics() if hasattr(model, 'calculate_metrics') else {}
            }
            report.update(model_info)

        return report

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

        self._validate_trade(trade_data)
        
        with self._metrics_lock:
            if not isinstance(trade_data, dict):
                raise TypeError("Trade data must be a dictionary")
            self.metrics['trades'].append(trade_data)
            self._update_trade_stats(trade_data)
            self.prune_old_trades()

        if not isinstance(trade_data, dict):
            raise TypeError("Trade data must be a dictionary")
        
        # Store trade data in single location
        self.metrics['trades'].append(trade_data)
        
        # Update model-specific tracking
        version = trade_data.get('model_version', 'unversioned')
        if version in self.model_trackers:
            self.model_trackers[version].update(trade_data)

    def _validate_trade(self, trade: Dict) -> None:
        if not isinstance(trade, dict):
            raise TypeError(f"Trade must be dict, got {type(trade)}")
        if 'pnl' not in trade or not isinstance(trade['pnl'], (int, float)):
            raise ValueError("Trade must contain numeric 'pnl'")
        if 'time' not in trade or not isinstance(trade['time'], datetime):
            raise ValueError("Trade must contain datetime 'time'")
        # Add other required fields as needed
    
    def _update_trade_stats(self, trade: Dict) -> None:
        """Update all metrics for a new trade"""
        with self._metrics_lock:
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

    def get_trades(self, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """Get trades with optional filtering.
        
        Args:
            filter_func: Optional function accepting a trade dict and returning bool.
                        Trade dicts are guaranteed to have 'pnl' and 'time'.
        """
        with self._metrics_lock:
            trades = list(self.metrics['trades'])  # Copy for thread safety
        
        return trades if filter_func is None else [t for t in trades if filter_func(t)]

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

    def check_clustering(self) -> bool:
        """Call this from within your monitoring class"""
        return detect_clustering(self.metrics['trades'])

    def record_reduction(self, ticket: int, volume: float) -> None:
        """Record position reduction in metrics"""
        if 'position_reductions' not in self.metrics:
            self.metrics['position_reductions'] = []
        
        self.metrics['position_reductions'].append({
            'ticket': ticket,
            'volume': volume,
            'time': datetime.now()
        })

    def get_recent_trades(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get most recent trades with optional count limit
        
        Args:
            count: Number of recent trades to return (default: 5)
            
        Returns:
            List of trade dictionaries sorted by most recent first
            Each trade dict contains:
            - pnl: float
            - time: datetime
            - type: str (optional)
            - model_version: str (optional)
            - other trade metadata
        """
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count must be positive integer")
        
        # Get sorted trades (most recent first)
        trades = sorted(
            self.metrics['trades'],
            key=lambda x: x.get('time', datetime.min),
            reverse=True
        )
        
        # Return limited number with copy to prevent modification
        return [dict(t) for t in trades[:count]]

    def get_trades_by_timeframe(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trades within a specific timeframe
        
        Args:
            hours: Lookback period in hours (default: 24)
            
        Returns:
            List of trades executed within the timeframe
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            t for t in self.metrics['trades']
            if t.get('time', datetime.min) >= cutoff
        ]

    def _calculate_win_rate(self, trades: Optional[List[Dict]] = None) -> float:
        """Calculate win rate from trades (uses self.metrics if trades=None)"""
        trades = trades if trades is not None else self.metrics['trades']
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return wins / len(trades)

    def get_meta_learning_stats(self, gold_model: 'GoldMLModel') -> Dict[str, Any]:
        """
        Get meta-learning adaptation statistics if meta-learner is active.
        
        Args:
            gold_model: The GoldMLModel instance to analyze
            
        Returns:
            Dictionary with adaptation metrics or empty dict if no meta-learner
        """
        if not hasattr(gold_model, 'meta_learner') or gold_model.meta_learner is None:
            return {}
            
        report = gold_model.meta_learner.get_adaptation_report()
        if report.empty:
            return {}
            
        return {
            'signal_changes': (report['raw_pred'] != report['adapted_pred']).mean(),
            'trend_alignment': report['trend_bias'].mean(),
            'adaptation_impact': self._calculate_adaptation_impact(report),
            'raw_report': report.to_dict(orient='records')[-10:]  # Last 10 adaptations
        }

    def _calculate_adaptation_impact(self, report: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance impact of adaptations"""
        if report.empty:
            return {}
            
        changed = report[report['raw_pred'] != report['adapted_pred']]
        if len(changed) == 0:
            return {'improvement': 0.0, 'deterioration': 0.0}
            
        # Compare direction of changes vs market
        market_direction = np.sign(report['close'].pct_change())
        raw_correct = (report['raw_pred'] == market_direction).mean()
        adapted_correct = (report['adapted_pred'] == market_direction).mean()
        
        return {
            'improvement': max(0, adapted_correct - raw_correct),
            'deterioration': max(0, raw_correct - adapted_correct),
            'net_impact': adapted_correct - raw_correct
        }

    def print_meta_learning_stats(self, gold_model: 'GoldMLModel') -> None:
        """Print formatted meta-learning statistics"""
        stats = self.get_meta_learning_stats(gold_model)
        if not stats:
            print("Meta-learning not active")
            return
            
        print("\nMeta-Learning Adaptation Stats:")
        print(f"Signal Changes: {stats['signal_changes']:.1%}")
        print(f"Trend Alignment: {stats['trend_alignment']:.2f}")
        
        impact = stats['adaptation_impact']
        if impact:
            print(f"Accuracy Impact: {impact['net_impact']:+.1%} "
                f"(↑{impact['improvement']:.1%} | ↓{impact['deterioration']:.1%})")


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
        self._cache_timestamps = {}  # Track last update times
        self._reset_connection_state()
        self.emergency_stop = EmergencyStop() # For connection/technical failures only
        self.data_fetcher = data_fetcher
        self._cache_lock = Lock()
        
        
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

    def _execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Execute a function with retry logic for MT5 operations.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            max_retries: Maximum retry attempts (default: self.max_retries)
            retry_delay: Delay between retries in seconds (default: self.retry_delay)
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function if successful
            
        Raises:
            Exception: If all retries fail
        """
        return _execute_with_retry_core(
            func=func,
            max_retries=max_retries or self.max_retries,
            retry_delay=retry_delay or self.retry_delay,
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
        cache_key = f"positions_{symbol}"
        
        # 1. Cache Read Operation (with timeout safety)
        def _read_cache():
            if not use_cache:
                return None  # Bypass cache
            
            if cache_key in self._position_cache:
                if time.time() - self._cache_timestamps[cache_key] < 5:  # 5-second validity
                    return self._position_cache[cache_key].copy()
            return None
        
        cached = self._safe_cache_access(_read_cache)
        if cached is not None:
            return cached

        # 2. Fetch Fresh Data
        def _get_positions():
            positions = MT5Wrapper.positions_get(symbol=symbol)
            return [] if positions is None else positions
            
        positions = self._execute_with_retry(_get_positions)
        result = [self._parse_position(pos) for pos in positions] if positions else []

        # 3. Cache Write Operation (with timeout safety)
        def _write_cache():
            if use_cache:
                self._position_cache[cache_key] = result.copy()
                self._cache_timestamps[cache_key] = time.time()
        
        self._safe_cache_access(_write_cache)
        return result

    def _safe_cache_access(self, operation: Callable, timeout: float = 1.0):
        """Wrapper for lock operations with timeout"""
        if threading.current_thread() == threading.main_thread():
            logger.warning("Cache access from main thread may cause deadlocks")
        
        if not self._cache_lock.acquire(timeout=timeout):
            raise RuntimeError(f"Cache lock timeout after {timeout}s")
        try:
            return operation()
        finally:
            self._cache_lock.release()

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
    
    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify stop loss and take profit of an existing position with retry logic
        
        Args:
            ticket: Position ticket ID
            sl: New stop loss price
            tp: New take profit price
        
        Returns:
            bool: True if modification succeeded, False otherwise
        """
        def _prepare_and_modify() -> Optional[Dict]:
            """Inner function preparing MT5 request with validation"""
            positions = MT5Wrapper.positions_get(ticket=ticket)
            if not positions:
                print(f"[ERROR] Position {ticket} not found")
                return None
                
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": tp,
                "magic": 123456,  # Your exact magic number
            }
            
            return MT5Wrapper.order_send(request)
        
        # Your exact retry logic with all original parameters
        result = self._execute_with_retry(
            func=_prepare_and_modify,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            success_retcode=MT5Wrapper.TRADE_RETCODE_DONE,
            ensure_connected=self.ensure_connected,
            on_success=lambda: setattr(self, 'last_activity_time', time.time()),
            on_exception=self._reset_connection_state
        )
        
        if result and hasattr(result, 'retcode'):
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[SUCCESS] Modified position {ticket} | SL: {sl:.5f} | TP: {tp:.5f}")
                return True
            print(f"[FAILED] Modify position {ticket} failed. Retcode: {result.retcode}")
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
        # 1. Try cache read with timeout protection
        def _read_cached_position():
            for positions in self._position_cache.values():
                for pos in positions:
                    if pos['ticket'] == ticket:
                        return pos.copy()  # Return a copy to prevent external mutation
            return None
        
        cached = self._safe_cache_access(_read_cached_position)
        if cached:
            return cached

        # 2. Fetch fresh if not in cache
        def _fetch_position():
            positions = MT5Wrapper.positions_get(ticket=ticket)
            return positions[0] if positions else None
        
        position = self._execute_with_retry(_fetch_position)
        if not position:
            return None

        parsed = self._parse_position(position)
        
        # 3. Update cache (if found)
        def _update_cache():
            symbol = parsed['symbol']
            cache_key = f"positions_{symbol}"
            if cache_key in self._position_cache:
                self._position_cache[cache_key] = [
                    p for p in self._position_cache[cache_key] 
                    if p['ticket'] != ticket
                ]
                self._position_cache[cache_key].append(parsed.copy())
        
        self._safe_cache_access(_update_cache)
        return parsed

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
        
        with self._connection_lock:  # Keep connection lock separate
            self._safe_cache_access(
                lambda: self._position_cache.clear(),
                timeout=0.5  # Shorter timeout for recovery ops
            )
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
    
    def account_info(self) -> Optional[Dict[str, float]]:
        """Get account balance/equity with retry logic"""
        def _get_account_info():
            return MT5Wrapper.account_info()
        return self._execute_with_retry(_get_account_info)

    def symbol_info(self, symbol: str) -> Optional[Dict[str, Union[float, int]]]:
        """Get symbol details with retry logic"""
        def _get_symbol_info():
            return MT5Wrapper.symbol_info(symbol)
        return self._execute_with_retry(_get_symbol_info)

    def calculate_margin(self, symbol: str, trade_type: str, volume: float, price: float) -> Optional[float]:
        """Calculate required margin using MT5Wrapper"""
        return MT5Wrapper.calculate_margin(
            trade_type=trade_type,
            symbol=symbol,
            volume=volume,
            price=price
        )


class DataFetcher:
    """Handles all data fetching operations using MT5 connection"""
    
    def __init__(self, mt5_connector: MT5Connector, config: Config, mt5_wrapper: Optional[MT5Wrapper] = None):
        self.mt5 = mt5_connector
        self.config = config
        self.mt5_wrapper = mt5_wrapper if mt5_wrapper is not None else MT5Wrapper()
        self._volatility_cache: Dict[str, float] = {} 
        self._cache_lock = Lock()
        self._last_update: Dict[str, datetime] = {}
        self._cache = {}
        self.candle_cache = {}
        self._price_cache: Dict[str, Dict] = {}
        self._symbol_info_cache: Dict[str, Dict] = {}
        self._atr_cache: Dict[str, Tuple[float, datetime]] = {}  # Symbol -> (ATR value, timestamp)
        self._cache_expiry = {
            'price': 1.0,       # 1 second for tick data
            'symbol_info': 120, # 2 minutes for symbol info
            'historical': 300,  # 5 minutes for historical data
            'volatility': 300   # 5 minutes for volatility
        }
        self.timeframe = MT5Wrapper.TIMEFRAME_H1

    
    def _parse_timeframe(self, tf: Union[int, str]) -> int:
        """Convert timeframe to MT5 integer constant with validation
        
        Args:
            tf: Either MT5 integer constant or string representation (e.g., 'M1', 'H1')
        
        Returns:
            int: MT5 timeframe constant
            
        Raises:
            ValueError: If timeframe is invalid
        """
        # Handle integer inputs
        if isinstance(tf, int):
            valid_integers = {
                MT5Wrapper.TIMEFRAME_M1,
                MT5Wrapper.TIMEFRAME_M5,
                MT5Wrapper.TIMEFRAME_M15,
                MT5Wrapper.TIMEFRAME_H1,
                MT5Wrapper.TIMEFRAME_H4,
                MT5Wrapper.TIMEFRAME_D1
            }
            if tf in valid_integers:
                return tf
            raise ValueError(f"Invalid MT5 timeframe constant: {tf}")

        # Handle string inputs
        tf_map = {
            'M1': MT5Wrapper.TIMEFRAME_M1,
            'M5': MT5Wrapper.TIMEFRAME_M5,
            'M15': MT5Wrapper.TIMEFRAME_M15,
            'H1': MT5Wrapper.TIMEFRAME_H1,
            'H4': MT5Wrapper.TIMEFRAME_H4,
            'D1': MT5Wrapper.TIMEFRAME_D1,
            # Alternative string formats
            'm1': MT5Wrapper.TIMEFRAME_M1,
            '5m': MT5Wrapper.TIMEFRAME_M5,
            '15m': MT5Wrapper.TIMEFRAME_M15,
            '1h': MT5Wrapper.TIMEFRAME_H1,
            '4h': MT5Wrapper.TIMEFRAME_H4,
            '1d': MT5Wrapper.TIMEFRAME_D1
        }
        
        tf_str = tf.upper()  # Normalize to uppercase
        if tf_str not in tf_map:
            raise ValueError(f"Unsupported timeframe string: {tf}")
        
        return tf_map[tf_str]
    
    def get_daily_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Get daily ATR with improved caching and validation"""
        # Input validation
        if not isinstance(symbol, str) or not symbol.isalpha():
            logger.error(f"Invalid symbol: {symbol}")
            return None
            
        period = max(5, min(period, 50))  # Clamp to 5-50 range

        with self._cache_lock:
            # Check cache first
            cache_key = f"atr_{symbol}_{period}"
            if cache_key in self._atr_cache:
                value, timestamp = self._atr_cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self._cache_expiry['historical']:
                    return value

            try:
                # Get rates with buffer
                rates = MT5Wrapper.copy_rates_from_pos(
                    symbol, 
                    MT5Wrapper.TIMEFRAME_D1, 
                    0, 
                    period + 5  # Extra buffer for calculation
                )
                if not rates or len(rates) < period:
                    logger.warning(f"Insufficient data for {period}-day ATR on {symbol}")
                    return None

                # Convert to numpy arrays
                high = np.array([r['high'] for r in rates], dtype=np.float32)
                low = np.array([r['low'] for r in rates], dtype=np.float32)
                close = np.array([r['close'] for r in rates], dtype=np.float32)

                # Calculate using IndicatorUtils
                atr_values = IndicatorUtils.atr(high, low, close, period)
                if len(atr_values) == 0 or np.isnan(atr_values[-1]):
                    return None

                result = float(atr_values[-1])
                
                # Update cache
                self._atr_cache[cache_key] = (result, datetime.now())
                return result

            except Exception as e:
                logger.error(f"ATR calculation failed for {symbol}: {str(e)}")
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
    
    def get_volatility(self, symbol: str, period: int = 14, lookback_periods: Optional[int] = None) -> float:
        """Get normalized volatility score (0-1) with thread-safe caching.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            period: ATR period (default: 14)
            lookback_periods: Optional override for historical volatility calculation
            
        Returns:
            Normalized volatility score (0-1) where 1 = maximum volatility
        """
        with self._cache_lock:
            # Determine cache keys based on parameters
            period_to_use = lookback_periods if lookback_periods is not None else period
            vol_cache_key = f"vol_{symbol}_{period_to_use}"
            atr_cache_key = f"atr_{symbol}_{period_to_use}"
            
            # Check cache first
            if vol_cache_key in self._volatility_cache:
                cached_atr = self._atr_cache.get(atr_cache_key)
                if cached_atr is not None:
                    atr_value, timestamp = cached_atr
                    if (datetime.now() - timestamp).total_seconds() < self._cache_expiry['volatility']:
                        cached_vol = self._volatility_cache.get(vol_cache_key)
                        if cached_vol is not None:
                            return cached_vol

            try:
                # Get fresh ATR value - use appropriate method
                atr_value = (
                    self.get_historical_atr(symbol, lookback_periods) 
                    if lookback_periods 
                    else self.get_daily_atr(symbol, period)
                )
                
                # Early return if ATR is None
                if atr_value is None:
                    logger.debug(f"No ATR value for {symbol}, using neutral fallback")
                    return 0.5

                # Get current price for normalization
                tick = self.get_current_price(symbol)
                if not tick or 'ask' not in tick:
                    logger.debug(f"No valid tick data for {symbol}, using neutral fallback")
                    return 0.5

                # Price validation
                try:
                    price = float(tick['ask'])
                    if price <= 0:
                        raise ValueError("Price must be positive")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Invalid price for {symbol}: {tick.get('ask')} - {str(e)}")
                    return 0.5

                # Normalize volatility (now safe from None values)
                normalized_vol = min(1.0, atr_value / (price * 0.01))  # Cap at 1% of price

                # Apply symbol-specific multipliers
                try:
                    symbol_type = 'gold' if 'gold' in symbol.lower() else 'default'
                    multiplier = self.config.VOLATILITY_SETTINGS.get(
                        symbol_type,
                        self.config.VOLATILITY_SETTINGS['default']
                    )['multiplier']
                    final_vol = min(1.0, normalized_vol * multiplier)
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Volatility config error: {str(e)}")
                    final_vol = normalized_vol

                # Update caches
                self._volatility_cache[vol_cache_key] = final_vol
                self._atr_cache[atr_cache_key] = (float(atr_value), datetime.now())

                return final_vol

            except Exception as e:
                logger.error(f"Volatility calculation failed: {str(e)}", exc_info=True)
                return 0.5
            
    def _fetch_historical_data(self, symbol: str, timeframe: int, num_candles: int) -> Optional[pd.DataFrame]:
        """Actual MT5 historical data fetching using MT5Wrapper instance"""
        try:
            # Call through the mt5_wrapper instance
            rates = self.mt5_wrapper.copy_rates_from_pos(
                symbol=symbol,
                timeframe=timeframe,
                start_pos=0,
                count=num_candles
            )
            
            if rates is None:
                return None
                
            # Convert to numpy array if we got a list of dicts
            if isinstance(rates, list):
                import numpy as np
                rates = np.array([list(d.values()) for d in rates])
                
            return self._process_rates_to_df(rates)
            
        except Exception as e:
            logger.error(f"Historical data fetch failed for {symbol} {timeframe}: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price data with caching, returning only numeric values"""
        if not self._validate_symbol(symbol):
            return None
            
        cache_key = f"price_{symbol}"
        
        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached['timestamp'] < self._cache_expiry['price']:
                return cached['data']
        
        # Fetch fresh data
        price_data = self._fetch_raw_price(symbol)
        if not price_data:
            return None
        
        # Cache and return
        self._cache[cache_key] = {
            'data': price_data,
            'timestamp': time.time()
        }
        return price_data

    def get_last_n_candles(self, symbol: str, timeframe: Union[int, str], n: int):
        """Get last N candles for a symbol/timeframe using MT5Wrapper
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Either MT5 integer constant or string representation
            n: Number of candles to fetch (default: 100)
        """
        # Input validation
        if not isinstance(symbol, str):
            raise TypeError(f"symbol must be str, got {type(symbol)}")
        
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be positive integer, got {n}")
        
        # Define cache_key OUTSIDE the if block
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        with self._cache_lock:
            if cache_key in self.candle_cache:
                cached = self.candle_cache[cache_key]
                if len(cached) >= n:
                    return cached.tail(n).copy()
        
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
                'last': float(tick['last'])
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

    def _process_rates_to_df(self, rates: Union[np.ndarray, List[Dict[str, Union[int, float]]]]) -> pd.DataFrame:
        """Convert MT5 rates data to pandas DataFrame
        
        Args:
            rates: Input data either as numpy array or list of dictionaries
            
        Returns:
            pd.DataFrame with properly formatted OHLC data
            
        Raises:
            ValueError: If rates data is empty
            TypeError: If input type is not supported
        """
        if rates is None or len(rates) == 0:
            raise ValueError("Empty rates data received")
        
        # Convert list of dicts to numpy array if needed
        if isinstance(rates, list):
            rates = np.array([list(d.values()) for d in rates])
        
        # Ensure we have a numpy array
        if not isinstance(rates, np.ndarray):
            raise TypeError(f"Expected numpy array or list of dicts, got {type(rates)}")
        
        # Create DataFrame with proper column names
        columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        df = pd.DataFrame(rates, columns=columns)
        
        # Convert and set time index
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_historical_atr(self, symbol: str, periods: int, timeframe: Optional[int] = None) -> Optional[float]:
        """Calculate ATR from historical data
        
        Args:
            symbol: Trading symbol
            periods: Number of periods for ATR calculation
            timeframe: MT5 timeframe constant (optional, uses instance default if None)
        """
        try:
            tf = timeframe if timeframe is not None else self.timeframe
            data = self.get_historical_data(
                symbol=symbol,
                timeframe=tf,
                num_candles=periods + 1  # Need one extra candle for calculation
            )
            if data is None or len(data) < periods:
                return None
            return self._calculate_atr(data, periods)
        except Exception as e:
            logger.error(f"Historical ATR calculation failed: {str(e)}")
            return None

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Core ATR calculation from DataFrame"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean().iloc[-1]
            return float(atr)
        except Exception as e:
            logger.error(f"ATR calculation error: {str(e)}")
            return 0.0

    def get_rsi(self, symbol: Optional[str] = None, timeframe: Optional[int] = None, period: int = 14) -> float:
        """
        Calculate current RSI value for a symbol.
        
        Args:
            symbol: Trading symbol (uses default if None)
            timeframe: MT5 timeframe (defaults to class timeframe)
            period: RSI calculation period (default: 14)
            
        Returns:
            float: Current RSI value (0-100) or 50.0 if calculation fails
        """
        from typing import Optional  # Add this at top of file if not already present
        
        try:
            # Handle None values with type-safe defaults
            symbol_str: str = symbol if symbol is not None else self.config.SYMBOL
            timeframe_int: int = timeframe if timeframe is not None else self.config.TIMEFRAME
            
            # Get fresh data (2x period to ensure accurate calculation)
            rates = self.get_historical_data(
                symbol=symbol_str,
                timeframe=timeframe_int,
                num_candles=period * 2
            )
            
            if rates is None or len(rates) < period:
                return 50.0  # Neutral value if insufficient data
                
            # Calculate price changes
            closes = rates['close'].astype(float)
            deltas = closes.diff()
            
            # Separate gains and losses
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            # Calculate average gains/losses
            avg_gain = gains.rolling(period).mean().iloc[-1]
            avg_loss = losses.rolling(period).mean().iloc[-1]
            
            # Handle edge case (zero loss)
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
                
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
        except Exception as e:
            symbol_for_error = symbol if symbol is not None else self.config.SYMBOL
            print(f"[ERROR] RSI calculation failed for {symbol_for_error}: {str(e)}")
            return 50.0  # Neutral fallback value
        
    def get_volatility_index(self, symbol: Optional[str] = None) -> float:
        """
        Calculate basic volatility index (0-100) using current ATR vs 14-day average.
        
        Args:
            symbol: Optional trading symbol (uses default symbol if None)
        
        Returns:
            float: 0-100 volatility index (30 = neutral baseline)
        """
        try:
            # Type-safe symbol handling
            symbol_str: str = symbol if symbol is not None else self.config.SYMBOL
            
            # 1. Get current ATR
            current_atr = self.get_current_atr(symbol_str, period=14)
            if current_atr is None:
                return 30.0  # Neutral default if data unavailable
            
            # 2. Get 14-day average ATR
            hist_data = self.get_historical_data(
                symbol=symbol_str,
                timeframe=self.config.TIMEFRAME,
                num_candles=14
            )
            if hist_data is None or len(hist_data) < 14:
                return 30.0
                
            avg_atr = hist_data['atr'].mean()
            
            # 3. Calculate simple ratio
            ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            # 4. Scale to 0-100 (30 = baseline)
            volatility_index = min(100, max(0, (ratio * 30)))
            
            return round(volatility_index, 1)
            
        except Exception:
            return 30.0  # Fail-safe neutral value
    
    def get_current_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Get current ATR value with thread-safe caching.
        
        Args:
            symbol: Trading symbol
            period: ATR period (default: 14)
            
        Returns:
            Current ATR value or None if calculation fails
        """
        # Input validation
        if not isinstance(symbol, str) or not symbol.isalpha():
            logger.error(f"Invalid symbol: {symbol}")
            return None
            
        period = max(5, min(period, 50))  # Keep within reasonable bounds

        with self._cache_lock:
            # Check cache first
            cache_key = f"current_atr_{symbol}_{period}"
            if cache_key in self._atr_cache:
                value, timestamp = self._atr_cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self._cache_expiry['volatility']:
                    return value

            try:
                # Get fresh data with buffer
                rates = self.get_historical_data(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    num_candles=period + 10  # Extra buffer for calculation
                )
                
                if rates is None or len(rates) < period:
                    logger.warning(f"Insufficient data for {period}-period ATR on {symbol}")
                    return None

                # Calculate ATR
                atr = self._calculate_atr(rates, period)
                if atr <= 0 or np.isnan(atr):
                    return None

                # Update cache
                self._atr_cache[cache_key] = (atr, datetime.now())
                return atr

            except Exception as e:
                logger.error(f"Current ATR calculation failed for {symbol}: {str(e)}")
                return None


class DataPreprocessor:
    """Handles data preprocessing and feature engineering with comprehensive validation"""
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.scaler: RobustScaler = RobustScaler()
        self.data_fetcher = data_fetcher
        self.features: List[str] = config.FEATURES
        self.window_size: int = 30  # Number of candles to look back
        self.training_data: Optional[pd.DataFrame] = None
        self._processed_features: Optional[np.ndarray] = None  # Change to protected/private variable

        self._processed_cache = deque(maxlen=10000)
        self._feature_names_cache = self._generate_feature_names()

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
        
    @lru_cache(maxsize=1000)
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
            if len(self._processed_cache) > 0:
                logger.warning("Cache not properly cleared during preprocessor reload")
            
            if not isinstance(loaded, RobustScaler):
                raise TypeError(f"Expected RobustScaler, got {type(loaded)}")
                
            if hasattr(loaded, 'n_features_in_') and loaded.n_features_in_ != len(self.features):
                raise ValueError(
                    f"Feature mismatch. Expected {len(self.features)}, "
                    f"got {loaded.n_features_in_}"
                )
                
            self.clear_cache()

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
            self._processed_features = X
            
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
            
            # Create trade mask (filter out "no trade" samples where y_seq == -1)
            trade_mask = y_seq != -1
            
            # After sequence creation
            self.update_cache(X_seq[trade_mask])  # Cache only trade-worthy samples
            return X_seq[trade_mask], y_seq[trade_mask]
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    @property 
    def feature_names_with_time(self) -> List[str]:
        """Get time-aware feature names for explanations"""
        return self._feature_names_cache

    @feature_names_with_time.setter
    def feature_names_with_time(self, features: List[str]):
        """Update feature names and regenerate time-aware names"""
        self.features = features
        self._feature_names_cache = self._generate_feature_names()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe technical indicators with no lookahead bias and type guarantees."""
        df = df.copy()
        
        # 1. Convert all numeric columns and enforce types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume'] 
        df[numeric_cols] = df[numeric_cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        ).astype('float64')

        # 2. Lagged price data (MUST for no lookahead)
        close = df['close'].shift(1)  # Critical: use previous candle's close
        high = df['high'].shift(1)
        low = df['low'].shift(1)

        # 3. RSI (type-safe implementation)
        delta = close.diff()
        gain = delta.where(delta.gt(0), 0.0)      # .gt() avoids dtype issues
        loss = delta.where(delta.lt(0), 0.0).abs() # .lt() for safety
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            df['rsi'] = (100 - (100 / (1 + rs))).clip(0, 100).fillna(50)  # Bound RSI

        # 4. MACD (original logic with type safety)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26  # Preserve original calculation
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # Unchanged

        # 5. Bollinger Bands (lagged close)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        df['upper_band'] = sma20 + 2 * std20
        df['lower_band'] = sma20 - 2 * std20

        # 6. Volume indicators (if available)
        if 'volume' in df.columns:
            volume = df['volume'].shift(1)  # Lagged volume
            df['vwap'] = (volume * close).cumsum() / volume.cumsum()
            df['volume_ma'] = volume.rolling(20).mean()
            df['volume_roc'] = volume.pct_change(5)

        # 7. ADX (using lagged high/low)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['adx'] = self._calculate_adx(
                tuple(zip(df.index, high)),
                tuple(zip(df.index, low)), 
                tuple(zip(df.index, close)),
                window=self.config.ADX_WINDOW
            )

        # 8. Cleanup
        return df.dropna().astype('float32')  # Reduce memory usage
    
    def clear_cache(self) -> None:
        """Clear the indicator cache"""
        self._indicator_cache.clear()
        self._processed_cache.clear()
        self._calculate_adx.cache_clear()
        self._calculate_volume_indicators.cache_clear()

    def validate_for_explainability(self) -> bool:
        """Check if preprocessor is explainability-ready"""
        checks = [
            hasattr(self, '_processed_cache'),
            len(self._processed_cache) >= self.window_size,
            len(self._feature_names_cache) == len(self.features) * self.window_size,
            isinstance(self.scaler, RobustScaler)
        ]
        return all(checks)

    def _create_sequences(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Optimized sequence creation with memory checks"""
        if len(data) < window_size:
            raise ValueError(f"Need {window_size} samples, got {len(data)}")
            
        # Pre-allocate array
        n_sequences = len(data) - window_size + 1
        seq_array = np.empty((n_sequences, window_size, data.shape[1]), dtype=np.float32)
        
        for i in range(n_sequences):
            seq_array[i] = data[i:i+window_size]
            
        return seq_array

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

    def _generate_feature_names(self) -> List[str]:
        """Generate time-step aware feature names (e.g., 'close_t-5')"""
        return [
            f"{feat}_t-{i}" 
            for i in range(self.window_size, 0, -1) 
            for feat in self.features
        ]

    
    @property 
    def processed_features(self) -> np.ndarray:
        """Get cached processed features with validation"""
        if self._processed_features is None:
            raise AttributeError("Processed features not initialized")
        return self._processed_features

    @processed_features.setter
    def processed_features(self, value: np.ndarray):
        """Set processed features with validation"""
        if not isinstance(value, np.ndarray):
            raise TypeError("Processed features must be numpy array")
        self._processed_features = value

    def update_cache(self, new_data: np.ndarray) -> None:
        """Update rolling cache with new processed data"""
        if not isinstance(new_data, np.ndarray):
            raise TypeError("Input must be numpy array")
        self._processed_cache.extend(new_data)

    def get_recent_samples(self, n_samples: int) -> np.ndarray:
        """Get most recent samples for explainability"""
        return np.array(list(self._processed_cache)[-n_samples:])


class AgnosticMetaLearning:
    def __init__(self, base_model: 'GoldMLModel', adaptation_window: int = 100, 
                 sensitivity: float = 0.3):
        """
        A PyTorch-free meta-learning adapter for trading models.
        
        Args:
            base_model: Your trained GoldMLModel instance
            adaptation_window: Number of recent samples for adaptation (default: 100 candles)
            sensitivity: How aggressively to adapt (0.1=conservative, 0.5=aggressive)
        """
        self.base_model = base_model
        self.window = adaptation_window
        self.sensitivity = np.clip(sensitivity, 0.05, 0.5)
        self.buffer = pd.DataFrame()

        self.max_sensitivity = 0.5  # Maximum allowed sensitivity
        self.min_sensitivity = 0.1  # Minimum allowed sensitivity
        
        # Dynamic adaptation parameters
        self.adaptation_params = {
            'feature_weights': None,
            'volatility_adjustment': 1.0,
            'trend_bias': 0.0
        }
        
        # Performance tracking
        self.adaptation_history = []


    def update_market_state(self, new_data: pd.DataFrame):
        """Update the adaptation buffer with new market data"""
        self.buffer = pd.concat([self.buffer, new_data]).tail(self.window)
        
        if len(self.buffer) >= 10:  # Minimum samples to adapt
            self._calculate_adaptation_rules()

    def _calculate_adaptation_rules(self):
        """Core pandas/numpy adaptation logic"""
        # 1. Calculate market regime statistics
        returns = self.buffer['close'].pct_change().dropna()
        self.adaptation_params['volatility_adjustment'] = 1 / (1 + returns.std())
        self.adaptation_params['trend_bias'] = np.sign(returns.tail(5).mean())
        
        # 2. Dynamic feature reweighting
        corr_matrix = self.buffer.corr()
        feature_importance = corr_matrix['close'].abs().drop('close')
        self.adaptation_params['feature_weights'] = (
            feature_importance / feature_importance.sum()
        )

    def adapt_prediction(self, raw_prediction: int, features: pd.Series) -> int:
        """
        Apply meta-learning adjustments to raw model predictions.
        
        Args:
            raw_prediction: Base model's prediction (-1, 0, 1)
            features: Current feature values as pandas Series
            
        Returns:
            Adapted prediction signal
        """
        if len(self.buffer) < 10 or raw_prediction == 0:
            return raw_prediction
            
        # Convert to numpy arrays with explicit type casting
        feature_values = np.asarray(features.values, dtype=np.float64)
        weights = np.asarray(
            self.adaptation_params['feature_weights'].values,
            dtype=np.float64
        )
        
        # Calculate meta-features with proper array types
        trend_alignment = float(self.adaptation_params['trend_bias']) * np.dot(
            feature_values,
            weights
        )
        
        # Apply adaptation rules
        confidence_threshold = 0.5 * self.sensitivity * self.adaptation_params['volatility_adjustment']
        
        if abs(trend_alignment) > confidence_threshold:
            return np.sign(trend_alignment)
        return raw_prediction

    def predict(self, current_features: pd.DataFrame) -> int:
        """
        Enhanced prediction with meta-learning adaptation.
        
        Args:
            current_features: DataFrame with current market features (shape: 1 x n_features)
            
        Returns:
            Adapted trading signal (-1, 0, 1)
        """
        # Get base model prediction
        raw_pred = self.base_model.predict_signal(current_features.values)
        
        # Apply meta-adaptation
        adapted_pred = self.adapt_prediction(raw_pred, current_features.iloc[0])
        
        # Record adaptation impact
        self.adaptation_history.append({
            'timestamp': pd.Timestamp.now(),
            'raw_pred': raw_pred,
            'adapted_pred': adapted_pred,
            'volatility': self.adaptation_params['volatility_adjustment'],
            'trend_bias': self.adaptation_params['trend_bias']
        })
        
        return adapted_pred

    def get_adaptation_report(self) -> pd.DataFrame:
        """Return DataFrame with adaptation history"""
        return pd.DataFrame(self.adaptation_history)

    def record_trade_outcome(self, trade_type: str, entry_price: float, 
                        exit_price: float, pnl: float) -> None:
        """
        Records trade outcomes for meta-learning adaptation.
        
        Args:
            trade_type: 'buy' or 'sell'
            entry_price: Price when trade was opened
            exit_price: Price when trade was closed
            pnl: Profit/loss in account currency
        """
        if len(self.buffer) == 0:
            return
            
        # Calculate trade metrics
        trade_duration = len(self.buffer)  # Number of candles trade was open
        success = pnl > 0
        
        # Base adaptation based on trade outcome
        if success:
            self.sensitivity *= 1.05
            self.adaptation_params['volatility_adjustment'] *= 0.95
        else:
            self.sensitivity *= 0.95
            self.adaptation_params['volatility_adjustment'] *= 1.05
        
        # Duration-based adjustment - trades open longer get more weight
        duration_weight = min(1.0, trade_duration / 50)  # Normalize to 50 candles
        self.sensitivity *= (1 + 0.1 * duration_weight) if success else (1 - 0.1 * duration_weight)
        
        # Volatility-based adjustment
        if self.adaptation_params['volatility_adjustment'] < 0.5:
            self.sensitivity *= 0.9  # Be conservative in high volatility
        
        # Apply final bounds
        self.sensitivity = np.clip(self.sensitivity, self.min_sensitivity, self.max_sensitivity)
            
        # Record metrics
        self.adaptation_history.append({
            'timestamp': pd.Timestamp.now(),
            'type': 'trade_outcome',
            'trade_type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'duration_candles': trade_duration,  # Now being recorded
            'success': success,
            'final_sensitivity': self.sensitivity,
            'volatility_adjustment': self.adaptation_params['volatility_adjustment']
        })


class GoldMLModel:
    """Machine Learning model for trading predictions with Stable Baselines3 (PPO) for RL """
    
    def __init__(self, config: Config, monitor: PerformanceMonitor, 
                data_fetcher: Optional[DataFetcher] = None,
                enable_meta_learning: bool = False):
        if not isinstance(config, Config):  # <-- This was misaligned
            raise TypeError("config must be an instance of Config")
        self.config = config
        self.data_fetcher = data_fetcher or DataFetcher(self.mt5, self.config)
        self.model: Optional[PPO] = None 
        self._init_model()
        self.performance = ModelPerformanceTracker(self)
        self.monitor = monitor
        self.monitor.add_model(self)
        self.preprocessor = DataPreprocessor(config, self.data_fetcher) 
        if not self.preprocessor.validate_for_explainability():
            logger.warning("Preprocessor not fully configured for explainability")
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
        
        if not self.config.PREPROCESSOR_PATH.exists():
            self._train_initial_model()

        # Initialize RiskManager
        self.risk_manager = RiskManager(
            mt5_connector=self.mt5,
            config=self.config,
            performance_monitor=monitor,
            data_fetcher=self.data_fetcher
        )

        self.last_model_update = datetime.now()

        self.last_base_signal = None  # Track last base signal 

        # Meta-learning integration
        self.meta_learner = None
        if enable_meta_learning:
            self._init_meta_learning()

    def _init_meta_learning(self):
        """Initialize meta-learning adapter with safe attribute access"""
        try:
            self.meta_learner = AgnosticMetaLearning(
                base_model=self,
                adaptation_window=self.config.META_WINDOW if hasattr(self.config, 'META_WINDOW') else 100,
                sensitivity=self.config.META_SENSITIVITY if hasattr(self.config, 'META_SENSITIVITY') else 0.3
            )
        except Exception as e:
            logger.error(f"Meta-learning initialization failed: {str(e)}")
            self.meta_learner = None

    def update_meta_state(self, new_data: pd.DataFrame):
        """Update meta-learning with latest market data"""
        if self.meta_learner:
            self.meta_learner.update_market_state(new_data)

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
        """Train PPO model with cross-validation and out-of-sample testing.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            bool: True if training succeeds and validation passes, False otherwise.
        """
        if X is None or y is None:
            raise ValueError("Training data cannot be None")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        try:
            logger.info("Starting model training with validation...")

            # 1. Train-Validation Split (Time-Series Aware)
            test_size = int(0.2 * len(X))  # Last 20% for out-of-sample testing
            X_train, X_val = X[:-test_size], X[-test_size:]
            y_train, y_val = y[:-test_size], y[-test_size:]

            # 2. Cross-Validation (TimeSeriesSplit for walk-forward validation)
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                # Split into training/validation folds
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # Train on this fold
                env = TradingEnvironment(
                    data=X_fold_train,
                    features=self.preprocessor.features,
                    window_size=self.preprocessor.window_size,
                    symbol=self.config.SYMBOL,
                    config=self.config,
                    risk_manager=self.risk_manager,
                    initial_balance=self.config.ACCOUNT_BALANCE
                )
                
                self.model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    device="auto",
                    **self.config.RL_PARAMS
                )
                
                self.model.learn(
                    total_timesteps=self.config.RL_PARAMS['total_timesteps'],
                    callback=self._create_callbacks(),
                    progress_bar=True
                )
                
                # Evaluate on validation fold
                fold_score = self._evaluate_model(X_fold_val, y_fold_val)
                val_scores.append(fold_score)
                logger.info(f"Fold accuracy: {fold_score:.2%}")

            # 3. Final Training on Full Training Set (X_train)
            env = TradingEnvironment(
                data=X_train,
                features=self.preprocessor.features,
                window_size=self.preprocessor.window_size,
                symbol=self.config.SYMBOL,
                config=self.config,
                risk_manager=self.risk_manager,
                initial_balance=self.config.ACCOUNT_BALANCE
            )
            
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="auto",
                **self.config.RL_PARAMS
            )
            
            self.model.learn(
                total_timesteps=self.config.RL_PARAMS['total_timesteps'],
                callback=self._create_callbacks(),
                progress_bar=True
            )

            # 4. Out-of-Sample Test (X_val)
            test_score = self._evaluate_model(X_val, y_val)
            logger.info(f"✅ Training complete. Validation accuracy: {np.mean(val_scores):.2%}, Test accuracy: {test_score:.2%}")

            # 5. Save model only if test performance is acceptable
            if test_score >= np.mean(val_scores) * 0.9:  # Allow 10% degradation
                self.save_model()
                return True
            else:
                logger.error("Test performance degraded significantly. Model not saved.")
                return False

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
        
    def should_retrain_based_on_performance(self, threshold: float = 50.0) -> bool:
        """Determine if retraining is needed based on performance metrics.
        
        Args:
            threshold: Win rate percentage below which we retrain (default: 50%)
            
        Returns:
            bool: True if retraining is recommended
        """
        metrics = self.calculate_metrics()
        if metrics is None:
            return False
            
        return (metrics['win_rate'] < threshold or 
                metrics['sharpe_ratio'] < 1.0 or
                metrics['max_drawdown'] > metrics['total_return'] * 0.5)

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

    def explain_prediction_shap(self, X_sample: np.ndarray) -> shap.Explanation:
        """Explain prediction using SHAP"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Flatten input for model compatibility
        X_flat = X_sample.reshape(1, -1)
        
        # Create explainer
        explainer = shap.DeepExplainer(
            model=self.model.policy,
            data=self._get_background_data()
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_flat)
        return shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_flat,
            feature_names=self._get_feature_names()
        )

    def explain_prediction_lime(self, X_sample: np.ndarray) -> 'LimeExplanationResult':
        """Explain prediction using LIME"""
        explainer = LimeTabularExplainer(
            training_data=self._get_background_data().reshape(-1, self.preprocessor.window_size, len(self.preprocessor.features)),
            feature_names=self._get_feature_names(),
            mode="classification"
        )
        
        return explainer.explain_instance(
            data_row=X_sample.flatten(),
            predict_fn=self._predict_proba,
            top_labels=3
        )

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
        expected_features = self._get_feature_count()
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
        """Enhanced version with preprocessor validation"""
        try:
            # 1. Validate model path exists
            model_path = self.config.RL_MODEL_PATH.with_suffix('.zip')
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            # 2. Attempt to load model
            logger.info(f"Loading model from {model_path}...")
            self.model = PPO.load(model_path, print_system_info=True)
            
            # 3. Validate model loaded successfully
            if self.model is None:
                raise ValueError("Model failed to load (returned None)")
                
            # 4. Core model validation
            core_success, core_error = self._validate_model_core()
            if not core_success:
                raise core_error if core_error else Exception("Core validation failed")

            # 5. Architecture verification
            if not self._verify_model_architecture():
                raise ValueError("Model architecture verification failed")

            # 6. Preprocessor validation
            if not hasattr(self.preprocessor, 'window_size'):
                raise AttributeError("Preprocessor missing window_size")
                
            if not hasattr(self.preprocessor, 'features'):
                raise AttributeError("Preprocessor missing features")
                
            # 7. Feature compatibility check
            n_features = len(self.preprocessor.features)
            if n_features <= 0:
                raise ValueError(f"Invalid feature count: {n_features}")
                
            # 8. Input shape validation
            expected_features = n_features * self.preprocessor.window_size
            dummy_input = np.zeros((1, expected_features), dtype=np.float32)
            
            if not hasattr(self.model, 'predict'):
                raise AttributeError("Model has no predict method")
                
            action, _ = self.model.predict(dummy_input, deterministic=True)
            if not isinstance(action, (np.ndarray, int, np.integer)):
                raise ValueError(f"Invalid action type: {type(action)}")

            # 9. Preprocessor-model compatibility
            if not self._validate_preprocessor_compatibility():
                raise ValueError("Preprocessor-model compatibility check failed")

            # 10. Update sync state if everything passed
            self.update_sync_state()
            
            logger.info(f"✅ Model loaded successfully. Features: {n_features}, "
                    f"Input shape: {self.model.policy.observation_space.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}", exc_info=True)
            self.model = None
            return False
    
    def evaluate_new_model(self, test_data: pd.DataFrame) -> bool:
        """Evaluate new model before putting into production."""
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
            
            # Add SHAP explanation for the first test sample (new code)
            if is_better and len(X_test_scaled) > 0:
                try:
                    explanation = self.explain_prediction_shap(X_test_scaled[:1])
                    logger.info(f"SHAP explanation for first test sample:\n{explanation}")
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {str(e)}")
            
            logger.info(f"Model evaluation - Old: {old_acc:.2%}, New: {new_acc:.2%}, Approved: {is_better}")
            return is_better
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return False
    
    def predict_signal(self, X: np.ndarray) -> int:
        """Simplified prediction with meta-learning adaptation
        
        Args:
            X: Input features with shape (window_size, n_features)
            
        Returns:
            int: Trading signal (-1, 0, 1)
        """
        try:
            # Basic model validation
            if not self._ensure_valid_model() or self.model is None:
                return -1
                
            # Core prediction
            X_flat = X.reshape(1, -1).astype(np.float32)
            action, _ = self.model.predict(X_flat, deterministic=True)
            base_signal = self._parse_prediction(action)
            self.last_base_signal = base_signal
            
            # Apply meta-learning if enabled
            if hasattr(self, 'meta_learner') and self.meta_learner:
                feature_df = pd.DataFrame(X, columns=self.preprocessor.features)
                final_signal = self.meta_learner.adapt_prediction(
                    raw_prediction=base_signal,
                    features=feature_df.iloc[0]
                )
                
                if self.config.DEBUG_MODE and final_signal != base_signal:
                    logger.debug(f"Meta-adaptation: {base_signal}→{final_signal}")
            else:
                final_signal = base_signal
                
            return final_signal
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return -1

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
        """Get model health status including version, prediction stats, feature weights, and architecture."""
        return {
            'version': self.get_model_version(),
            'last_prediction': self.prediction_stats,
            'feature_weights': self._get_current_feature_weights(),
            'architecture': self._get_model_architecture(),
        }
    
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

    def _get_feature_names(self) -> List[str]:
        """Get feature names with time steps"""
        return [
            f"{feature}_t-{i}" 
            for i in range(self.preprocessor.window_size, 0, -1)
            for feature in self.preprocessor.features
        ]

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

    def visualize_feature_importance(self, sample_size: int = 100) -> Figure:
        """Generate global feature importance plot using SHAP values.
        
        Returns:
            Figure: Feature importance plot
        """
        if not self.model:
            raise ValueError("Model not loaded")

        # Get background data and sample to explain
        background = self._get_background_data(n_samples=sample_size)
        sample = background[:10]  # Small subset for visualization

        # SHAP explainer
        explainer = shap.DeepExplainer(
            model=self.model.policy,
            data=background
        )
        shap_values = explainer.shap_values(sample)

        # Create figure explicitly
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Plot
        shap.summary_plot(
            shap_values,
            features=sample,
            feature_names=self._get_feature_names(),
            plot_type="bar",
            show=False,
            ax=ax
        )
        fig.suptitle("Global Feature Importance (SHAP)")
        fig.tight_layout()
        return fig

    def _get_background_data(self, n_samples: int = 100) -> np.ndarray:
        """Get representative background data for explainability.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            np.ndarray: Shape (n_samples, window_size * n_features)
        """
        if not hasattr(self.preprocessor, 'processed_features'):
            raise AttributeError("Preprocessor has no processed_features")

        # Get last n_samples from preprocessed data
        data = self.preprocessor.processed_features[-n_samples * 2:]  # Buffer
        
        # Convert to sequences
        sequences = []
        for i in range(len(data) - self.preprocessor.window_size):
            seq = data[i:i + self.preprocessor.window_size]
            sequences.append(seq.flatten())  # Flatten to (window_size * n_features,)

        return np.array(sequences[:n_samples])  # Return requested sample size

    def _predict_proba(self, instances: np.ndarray) -> np.ndarray:
        """Predict action probabilities for LIME explanations using pandas/numpy.
        
        Args:
            instances: Shape (n_samples, window_size * n_features)
            
        Returns:
            np.ndarray: Probability scores shape (n_samples, n_actions)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Convert to pandas DataFrame for easier manipulation
        n_features = len(self.preprocessor.features)
        window_size = self.preprocessor.window_size
        
        # Reshape to (n_samples, window_size, n_features)
        reshaped = instances.reshape(-1, window_size, n_features)
        
        # Create a DataFrame for each sample
        probs = []
        for sample in reshaped:
            # Create DataFrame with feature names
            df = pd.DataFrame(
                sample,
                columns=self.preprocessor.features
            )
            
            # Add time-related features if needed
            df['time_index'] = range(window_size)
            
            # Calculate simple probability estimates based on feature statistics
            # This is a simplified approach - you'll need to customize based on your actual model logic
            
            # Example: Calculate probability based on moving averages
            long_prob = 0.5  # Base probability
            short_prob = 0.5  # Base probability
            
            # Adjust probabilities based on feature values
            if 'close' in df.columns:
                ma_short = df['close'].rolling(5).mean().iloc[-1]
                ma_long = df['close'].rolling(20).mean().iloc[-1]
                
                if ma_short > ma_long:
                    long_prob += 0.2
                    short_prob -= 0.2
                else:
                    long_prob -= 0.2
                    short_prob += 0.2
            
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi < 30:
                    long_prob += 0.1
                elif rsi > 70:
                    short_prob += 0.1
            
            # Normalize probabilities to sum to 1
            total = long_prob + short_prob
            long_prob /= total
            short_prob /= total
            
            probs.append([short_prob, long_prob])  # Assuming binary classification
        
        return np.array(probs)

    def get_state_version(self) -> str:
        """Get combined version hash of model and preprocessor states"""
        try:
            # Safely get model parameters if available
            if self.model is None:
                model_hash = "no_model"
            else:
                # Use model's observation space shape as version identifier
                obs_shape = getattr(self.model.policy.observation_space, 'shape', None)
                model_hash = hashlib.md5(str(obs_shape).encode()).hexdigest()[:8]
            
            # Get preprocessor configuration hash
            preprocessor_config = {
                'features': self.preprocessor.features,
                'window_size': self.preprocessor.window_size,
                'scaler_params': self.preprocessor.scaler.get_params() if hasattr(self.preprocessor.scaler, 'get_params') else None
            }
            preprocessor_hash = hashlib.md5(str(preprocessor_config).encode()).hexdigest()[:8]
            
            version = f"{model_hash}-{preprocessor_hash}"
            return version
            
        except Exception as e:
            logger.error(f"Version generation failed: {str(e)}")
            return "error_version"

    def check_state_sync(self) -> bool:
        """Verify model and preprocessor are synchronized"""
        current_version = self.get_state_version()
        if not hasattr(self, '_last_sync_version'):
            self._last_sync_version = ""
        
        if current_version != self._last_sync_version:
            logger.warning(f"State version changed from {self._last_sync_version} to {current_version}")
            return False
        return True

    def update_sync_state(self) -> None:
        """Update the synchronization state after changes"""
        self._last_sync_version = self.get_state_version()

    def _validate_preprocessor_compatibility(self) -> bool:
        """Validate that preprocessor outputs match model expectations"""
        try:
            # 1. First validate we have a working model and preprocessor
            if self.model is None:
                raise ValueError("Cannot validate - model is None")
                
            if not hasattr(self.model, 'policy'):
                raise ValueError("Model has no policy attribute")
                
            if self.model.policy is None:
                raise ValueError("Model policy is None")
                
            if not hasattr(self.model.policy, 'observation_space'):
                raise ValueError("Policy has no observation_space")
                
            if self.preprocessor is None:
                raise ValueError("Preprocessor is None")
                
            # 2. Check feature count
            expected_features = self._get_feature_count()
            actual_features = len(getattr(self.preprocessor, 'features', []))
            
            if expected_features != actual_features:
                raise ValueError(
                    f"Feature count mismatch. Model expects {expected_features}, "
                    f"preprocessor has {actual_features}"
                )
                
            # 3. Check window size with multiple fallbacks
            obs_space = self.model.policy.observation_space
            expected_window = None
            
            # Try different ways to get expected window size
            if hasattr(obs_space, 'shape'):
                shapes = getattr(obs_space, 'shape', [None])
                if shapes and len(shapes) > 0:
                    expected_window = shapes[0]
            
            # If still None, try alternative approach
            if expected_window is None:
                if hasattr(self.model, 'observation_space'):
                    shapes = getattr(self.model.observation_space, 'shape', [None])
                    if shapes and len(shapes) > 0:
                        expected_window = shapes[0]
            
            # Final fallback to preprocessor's window size
            if expected_window is None:
                expected_window = getattr(self.preprocessor, 'window_size', None)
                logger.warning("Using preprocessor window size as fallback")
                
            actual_window = getattr(self.preprocessor, 'window_size', None)
            
            if expected_window is None or actual_window is None:
                raise ValueError("Could not determine window sizes for comparison")
                
            if expected_window != actual_window:
                raise ValueError(
                    f"Window size mismatch. Model expects {expected_window}, "
                    f"preprocessor uses {actual_window}"
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Preprocessor validation failed: {str(e)}")
            return False

    def refresh_model(self) -> bool:
        """Reload model and verify integrity"""
        if not self.config.RL_MODEL_PATH.exists():
            logger.error("No model file to refresh")
            return False
            
        try:
            self.model = PPO.load(self.config.RL_MODEL_PATH)
            return self.validate_model()
        except Exception as e:
            logger.error(f"Refresh failed: {str(e)}")
            return False

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from the model's trade history.
        
        Returns:
            dict: Metrics including Sharpe ratio, max drawdown, win rate, total return.
                Returns empty dict if no trades exist.
        """
        if not hasattr(self, 'trade_history') or not self.trade_history:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_trade_duration_hours': 0.0,
                'n_trades': 0
            }

        returns = np.array([trade['profit'] for trade in self.trade_history])
        cumulative_returns = np.cumsum(returns)
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Max Drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = peak - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win Rate
        win_rate = np.sum(returns > 0) / len(returns) * 100
        
        # Total Return
        total_return = np.sum(returns)
        
        # Additional metrics useful for trading bots
        avg_trade_duration = np.mean([
            (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600 
            for trade in self.trade_history 
            if 'exit_time' in trade and 'entry_time' in trade
        ]) if len(self.trade_history) > 0 else 0
        
        return {
            'sharpe_ratio': float(round(sharpe_ratio, 2)),
            'max_drawdown': float(round(max_drawdown, 2)),
            'win_rate': float(round(win_rate, 1)),
            'total_return': float(round(total_return, 2)),
            'avg_trade_duration_hours': float(round(avg_trade_duration, 1)),
            'n_trades': int(len(self.trade_history))
        }
    

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
    """Manages risk and position sizing"""
    
    def __init__(self, mt5_connector: MT5Connector, config: Config, 
                 performance_monitor: PerformanceMonitor, data_fetcher: DataFetcher):
        self.mt5 = mt5_connector
        self.config = config  # Store the passed config
        self.performance_monitor = performance_monitor  # Store the performance monitor
        self.today_trades = 0
        self.data_fetcher = data_fetcher
        self.emergency_stop = EmergencyStop() # For trading risk violations only
        self.max_trades = config.MAX_TRADES_PER_DAY  # Use from config instead of Config class
        self._symbol_info_cache: Dict[str, Dict] = {} 
        self._retry_attempts = 3  # Number of retry attempts
        self._retry_delay = 1.0  # Delay between retries in seconds
        self.current_volatility_regime = "medium"
        self.volatility_multipliers = {
            "high": 1.3,
            "medium": 1.0,
            "low": 0.7
        }
        self._atr_history_cache: Dict[str, List[float]] = {}

        # Circuit breaker state
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.last_loss_time = None

    def get_atr_history(self, symbol: str, period: int = 20) -> List[float]:
        """Get recent ATR values using DataFetcher's daily ATR calculations
        
        Args:
            symbol: Trading symbol (e.g. "XAUUSD")
            period: Number of daily ATR values to return
            
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
        try:
            # 1. Get base daily regime
            base_regime = self._get_base_regime(symbol)  # Your existing logic
            
            # 2. Check intraday spikes
            if self._check_intraday_spikes(symbol):
                return "high"  # Override if spike detected
                
            return base_regime
            
        except Exception as e:
            logger.error(f"Volatility detection failed: {str(e)}")
            return "medium"

    def _check_intraday_spikes(self, symbol: str) -> bool:
        """Detects abnormal volatility in smaller timeframes"""
        # Use your existing MT5Wrapper constants directly
        timeframes = [
            MT5Wrapper.TIMEFRAME_M15,  # 15 minutes
            MT5Wrapper.TIMEFRAME_H1    # 1 hour
        ]
        
        for timeframe in timeframes:
            data = self.data_fetcher.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,  # Passing the integer constant directly
                num_candles=24
            )
            if data is None:
                continue
                
            current_range = data['high'][-1] - data['low'][-1]
            avg_range = (data['high'] - data['low']).rolling(12).mean()[-1]
            
            if current_range > 2.5 * avg_range:
                return True
        return False

    def _get_base_regime(self, symbol: str) -> str:
        """Your existing daily ATR logic extracted"""
        atr = self.data_fetcher.get_daily_atr(symbol)
        if atr is None:
            return "medium"
            
        atr_history = self.get_atr_history(symbol)
        sma_atr = np.mean(atr_history[-20:]) if len(atr_history) >= 20 else atr
        
        if atr > sma_atr * 1.2: return "high"
        if atr < sma_atr * 0.8: return "low"
        return "medium"

    def check_margin_requirements(self, symbol: str, volume: float, price: float, trade_type: str) -> bool:
        if not self.mt5.ensure_connected():
            logger.error("Margin check failed - MT5 not connected")
            return False

        try:
            # 1. Get account info
            account = self.mt5.account_info()
            if not account or account['balance'] <= 0:
                logger.error(f"Invalid account balance: {account.get('balance') if account else 'None'}")
                return False

            # 2. Calculate required margin (now using only trade_type)
            margin_required = self.mt5.calculate_margin(
                symbol=symbol,
                trade_type=trade_type,  # Only this parameter needed
                volume=volume,
                price=price
            )
            
            if margin_required is None:
                logger.error("Margin calculation failed")
                return False

            # 3. Apply safety buffer
            available_margin = account['balance'] * 0.8  # 80% buffer
            
            if margin_required > available_margin:
                logger.warning(f"Insufficient margin: Required={margin_required:.2f}, Available={available_margin:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Margin check failed: {str(e)}")
            return False
    
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
            symbol_info = self.data_fetcher.get_symbol_info(symbol)
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
        """Calculate position size with bulletproof type handling"""
        try:
            # Convert inputs to Decimal (using your existing decimal import)
            stop_loss_dec = Decimal(str(stop_loss_pips))
            if stop_loss_dec <= Decimal('0'):
                logger.error(f"Invalid stop loss: {stop_loss_pips}")
                return 0.0

            # Get price data (using your PriceData type)
            tick = self.data_fetcher.get_current_price(symbol)
            if not tick:
                return 0.0

            # Extract first valid price (handles datetime/float union)
            ask_price = None
            for key in ['ask', 'last', 'bid']:
                if key in tick and isinstance(tick[key], (int, float)):
                    try:
                        ask_price = Decimal(str(tick[key]))
                        if ask_price > Decimal('0'):
                            break
                    except:
                        continue

            if not ask_price:
                return 0.0

            # Rest of calculation using your imports
            symbol_info = self.data_fetcher.get_symbol_info(symbol)
            if not symbol_info:
                return 0.0

            # Using your decimal context (getcontext().prec = 8)
            point_value = Decimal(str(symbol_info.get('point', 0)))
            balance = Decimal(str(self.data_fetcher.get_account_balance() or 0))
            risk_percent = Decimal(str(self.config.RISK_PER_TRADE))
            contract_size = Decimal(str(symbol_info.get('trade_contract_size', 1.0)))

            risk_amount = balance * (risk_percent / Decimal('100'))
            sl_amount = stop_loss_dec * point_value * contract_size
            
            if sl_amount == Decimal('0'):
                return 0.0

            position_size = risk_amount / sl_amount
            min_position = Decimal('0.01')
            max_position = (balance * Decimal('0.5')) / (ask_price * contract_size)

            # ROUND_DOWN ensures conservative position sizing
            from decimal import ROUND_DOWN  # Using your existing decimal import
            final_size = max(min_position, min(position_size, max_position))
            return float(final_size.quantize(Decimal('0.00001'), rounding=ROUND_DOWN))

        except Exception as e:
            logger.error(f"Position calc error: {str(e)}", exc_info=True)
            return 0.0
    
    def can_trade_today(self) -> bool:
        """Check if we can place more trades today"""
        return self.today_trades < self.max_trades
    
    def increment_trade_count(self):
        """Increment daily trade count"""
        self.today_trades += 1
        
    def check_emergency_conditions(self) -> bool:
        """Check various risk thresholds including consecutive losses and recent performance"""
        try:
            # 1. Daily loss check (existing)
            daily_pnl = self.performance_monitor.metrics.get('daily_pnl', [0])[-1]
            if daily_pnl < -self.config.MAX_DAILY_LOSS:
                self.emergency_stop.activate(f"Daily loss limit breached: {daily_pnl:.2f}")
                return True

            # 2. Enhanced loss streak detection (using both methods)
            recent_trades = self.performance_monitor.get_recent_trades(
                count=self.config.LOSS_STREAK_LOOKBACK
            )
            
            # 3. New: Timeframe-based loss concentration check
            hourly_trades = self.performance_monitor.get_trades_by_timeframe(hours=1)
            
            # Combined analysis
            loss_conditions = {
                'streak': sum(1 for t in recent_trades if t['profit'] < 0) if recent_trades else 0,
                'hourly_rate': sum(1 for t in hourly_trades if t['profit'] < 0)/len(hourly_trades) if hourly_trades else 0
            }
            
            # Trigger circuit breaker if either condition is met
            if (loss_conditions['streak'] >= self.config.MAX_CONSECUTIVE_LOSSES or 
                loss_conditions['hourly_rate'] > 0.8):  # 80% loss rate
                
                if not self.circuit_breaker_active:
                    reason = (
                        f"Loss conditions triggered - "
                        f"streak: {loss_conditions['streak']}/{self.config.MAX_CONSECUTIVE_LOSSES}, "
                        f"hourly loss rate: {loss_conditions['hourly_rate']:.0%}"
                    )
                    self.activate_circuit_breaker(reason)
                return True

            # 4. Drawdown check (existing)
            max_dd = self.performance_monitor.metrics.get('max_dd', 0)
            if max_dd < -self.config.MAX_DRAWDOWN_PCT:
                self.emergency_stop.activate(f"Max drawdown exceeded: {max_dd:.2f}%")
                return True

            # 5. Circuit breaker timeout check (fixed type handling)
            current_time = time.time()
            if (self.circuit_breaker_active and 
                self.last_loss_time is not None and
                (current_time - self.last_loss_time) > self.config.CIRCUIT_BREAKER_TIMEOUT):
                self.deactivate_circuit_breaker()
                
            return self.circuit_breaker_active

        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}", exc_info=True)
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

    def activate_circuit_breaker(self, reason: str):
        """Activate trading halt due to excessive losses"""
        self.circuit_breaker_active = True
        self.last_loss_time = time.time()
        logger.critical(
            f"ACTIVATING CIRCUIT BREAKER: {reason}. "
            f"Timeout: {self.config.CIRCUIT_BREAKER_TIMEOUT//3600}h"
        )
        self.emergency_stop.activate(f"Circuit Breaker: {reason}")

    def deactivate_circuit_breaker(self):
        """Resume normal trading after cooling period"""
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        logger.info("Circuit breaker deactivated - resuming normal trading")
        self.emergency_stop.deactivate()
    
    def reset_daily_counts(self):
        """Reset daily counters with logging"""
        if self.today_trades > 0:
            logger.info(f"Resetting daily counts (was {self.today_trades} trades)")
        self.today_trades = 0

    def validate_trade_params(self, symbol: str, price: float, sl: float, tp: float) -> Tuple[bool, str]:
        """Bulletproof trade validation with proper None handling"""
        # Get market data with explicit checks
        try:
            spread = self.data_fetcher.get_current_spread(symbol)
            atr = self.data_fetcher.get_daily_atr(symbol)
            current_price = self.data_fetcher.get_current_price(symbol)
            
            # Explicit None checks
            if spread is None:
                return False, "Could not determine current spread"
            if atr is None:
                return False, "Could not calculate ATR"
            if current_price is None or 'bid' not in current_price or 'ask' not in current_price:
                return False, "Invalid current price data"

            # Convert to float with validation
            try:
                bid = float(current_price['bid'])
                ask = float(current_price['ask'])
                spread = float(spread)
                atr = float(atr)
            except (TypeError, ValueError) as e:
                return False, f"Invalid numeric data: {str(e)}"

            # Core validation rules (now safe from None errors)
            checks = [
                (price > 0, "Price must be positive"),
                ((price > sl and sl > 0) or (price < sl), "Invalid stop loss position"),
                ((tp > price > sl) or (tp < price < sl), "Invalid take profit position"),
                (abs(price - sl) > spread * 3, f"SL too close (needs > {spread*3:.2f} spread)"),
                (abs(tp - price) > spread * 3, f"TP too close (needs > {spread*3:.2f} spread)"),
                (abs(price - sl) <= atr * 2, f"SL too wide (max {atr*2:.2f} ATR)"),
                (abs(tp - price) >= abs(price - sl) * 1.5, "Risk/reward ratio < 1.5"),
                (bid * 0.99 <= price <= ask * 1.01, "Price deviated >1% from current market")
            ]
            
            for condition, message in checks:
                if not condition:
                    return False, message
                    
            return True, "Valid parameters"
            
        except Exception as e:
            logger.error(f"Trade validation crashed: {str(e)}")
            return False, "Validation error"

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
    
    
    def clear_position_cache(self, ticket: Optional[int] = None) -> None:
        """Clears position cache entries with thread-safe access.
        
        Uses MT5Connector's _safe_cache_access to ensure atomic operations.
        
        Args:
            ticket: If provided, only clears positions matching this ticket ID.
                    If None, clears the entire position cache.
                    
        Behavior:
            - For None ticket: Fully clears all cached positions
            - For specific ticket: Removes only matching positions while preserving others
            - All operations are thread-safe with 1.0s timeout
            
        Raises:
            RuntimeError: If cache lock cannot be acquired within timeout
            AttributeError: If MT5 connector is not available
            
        Example:
            >>> clear_position_cache()  # Clears all cached positions
            >>> clear_position_cache(ticket=12345)  # Only removes position with ticket 12345
            
        Notes:
            - Uses a defensive copy of cache keys during iteration
            - Maintains cache consistency even during concurrent access
            - Connected to MT5Connector's _position_cache through composition
        """
        def _clear_all() -> None:
            """Clears entire position cache atomically"""
            self.mt5._position_cache.clear()
            if hasattr(self.mt5, '_cache_timestamps'):
                self.mt5._cache_timestamps.clear()
        
        def _clear_single() -> None:
            """Removes specific position by ticket ID across all symbols"""
            for symbol in list(self.mt5._position_cache.keys()):  # Defensive copy
                # Preserve non-matching positions
                self.mt5._position_cache[symbol] = [
                    pos for pos in self.mt5._position_cache[symbol] 
                    if pos['ticket'] != ticket
                ]
                # Update timestamp if tracking
                if hasattr(self.mt5, '_cache_timestamps'):
                    self.mt5._cache_timestamps[symbol] = time.time()
        
        self.mt5._safe_cache_access(
            _clear_all if ticket is None else _clear_single,
            timeout=1.0  # Conservative timeout for cache operations
        )


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


class TradingBot:
    """
    Trading Logic:

    Consider adding order book analysis for better fill simulation

    Add trade clustering detection
"""
    """Main trading bot class"""
    def __init__(self, config: Config, mt5_connector: MT5Connector, data_fetcher: DataFetcher, preprocessor: DataPreprocessor, mt5_wrapper: Optional[MT5Wrapper] = None):
        self.mt5 = mt5_connector
        self.config = config
        
        # Initialize core components in proper order
        self.performance_monitor = PerformanceMonitor(config)
        self.preprocessor = preprocessor  # Use the injected preprocessor instead of creating new
        self.mt5_wrapper = mt5_wrapper 
        self.data_fetcher = data_fetcher

        self._meta_learner = None
        if config.ENABLE_META_LEARNING:
            self._init_meta_learning()  # Now manages ALL meta-learning
        
        # Initialize ML model with meta-learning
        self.ml_model = GoldMLModel(
            config=config,
            monitor=self.performance_monitor,
            data_fetcher=self.data_fetcher,
            enable_meta_learning=config.ENABLE_META_LEARNING
        )

        # Track signals for debugging
        self.base_signal = None
        self.final_signal = None

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

    def _init_meta_learning(self):
        """Initialize meta-learning adapter with direct config access"""
        if not getattr(self.config, 'ENABLE_META_LEARNING', False):
            self._meta_learner = None  # Use protected attribute
            return
            
        self._meta_learner = AgnosticMetaLearning(
            base_model=self.ml_model,  # Pass the ML model, not TradingBot
            adaptation_window=getattr(self.config, 'META_WINDOW', 100),
            sensitivity=getattr(self.config, 'META_SENSITIVITY', 0.3)
        )
        
        if getattr(self.config, 'DEBUG_MODE', False):
            print(f"Meta-learning initialized with window={self.config.META_WINDOW} sensitivity={self.config.META_SENSITIVITY}")

    @property
    def meta_learner(self):
        """Read-only access to meta-learner"""
        return self._meta_learner

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

                # Trade clustering check (MUST be after position management)
                all_trades = self.performance_monitor.metrics.get('all_trades', [])
                if len(all_trades) >= 10:
                    recent_trades = all_trades[-10:]
                    
                    # Convert to returns for clustering detection
                    returns = [t['pnl']/t['entry_price'] for t in recent_trades 
                            if isinstance(t, dict) and 'pnl' in t and 'entry_price' in t and t['entry_price'] > 0]
                    
                    if len(returns) >= 5:
                        if detect_clustering(returns):
                            logger.warning("Trade clustering detected - reducing exposure")
                            self.reduce_position_sizes(0.5)  # Call TradingBot's method directly
                            # Reset to prevent repeated triggers
                            self.performance_monitor.metrics['all_trades'].clear()
             
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
        
        # 1. Get market context for meta-learning
        if hasattr(self, 'meta_learner') and self.meta_learner:
            market_data = self.data_fetcher.get_last_n_candles(
                symbol=Config.SYMBOL,
                timeframe=MT5Wrapper.TIMEFRAME_H1,  # Add timeframe from config
                n=self.meta_learner.window        # Pass window size as 'n'
            )
            if market_data is not None:
                self.meta_learner.update_market_state(market_data)

        # 2. Validate signal
        if signal not in [-1, 0, 1]:
            return False
            
        # 3. Trade type determination (modified)
        trade_type = "buy" if signal == 1 else ("sell" if signal == -1 else None)
        if trade_type is None:
            return False

        print(f"DEBUG: Executing {trade_type.upper()} trade (Signal origin: {'meta' if signal != self.base_signal else 'base'})")
        
        
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
                tp_price = 0
            else:  # sell
                sl_price = requested_price + stop_loss_pips
                tp_price = 0

            # ===== CRITICAL VALIDATION ADDITION =====
            if not self.risk_manager._validate_price_levels(Config.SYMBOL, requested_price, sl_price, tp_price):
                logger.error(f"Invalid SL/TP levels. SL: {sl_price:.2f}, TP: {tp_price:.2f}")
                return False
            # ========================================

            # 5. Calculate position size
            aggressive_size = self.risk_manager.calculate_position_size(
                Config.SYMBOL, stop_loss_pips
            )

            # ===== ADD VALIDATION HERE =====
            is_valid, reason = self.risk_manager.validate_trade_params(
                symbol=Config.SYMBOL,
                price=requested_price,
                sl=sl_price,
                tp=tp_price
            )
            if not is_valid:
                logger.warning(
                    f"🚨 Trade rejected: {reason}\n"
                    f"Requested: {requested_price:.5f}\n"
                    f"SL: {sl_price:.5f} | TP: {tp_price:.5f}\n"
                    f"Spread: {self.data_fetcher.get_current_spread(Config.SYMBOL):.2f}pips"
                )
                self.performance_monitor.update({
                    'type': 'rejection',
                    'symbol': Config.SYMBOL,
                    'price': requested_price,
                    'reason': reason,
                    'signal': signal,
                    'time': datetime.now()
                })
                return False
            # ===== END VALIDATION BLOCK =====

            # Calculate impact but don't use it for execution
            order_book = OrderBookSimulator()
            impact = order_book.get_impact(aggressive_size)
            adjusted_price = current_price_value * (1 + impact) if trade_type == 'buy' else current_price_value * (1 - impact)
            
            # Log the impact for analysis
            self.performance_monitor.update({
                'order_book_impact': impact,
                'adjusted_price': adjusted_price,
                'raw_price': current_price_value,
            })

            if not self.risk_manager.check_margin_requirements(
                symbol=Config.SYMBOL,
                volume=aggressive_size,
                price=current_price_value,
                trade_type=trade_type
            ):
                logger.error(f"Insufficient margin for {aggressive_size} lots")
                return False

            

            # 6. Trade Execution
            result = self.mt5.send_order(
                symbol=Config.SYMBOL,
                order_type=trade_type,
                volume=aggressive_size,
                sl=sl_price,
                tp=tp_price,
                comment=f"AGGR-{trade_type.upper()}-ATR{stop_loss_pips:.0f}"
            )

            # After successful trade execution 
            if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                ticket = int(result.get('order', 0)) or int(result.get('ticket', 0))
                
                # 1. Immediate trailing stop setup
                position = self.mt5.get_position(ticket)
                if position:
                    # Set initial ATR-based stop
                    self._update_trailing_stop(position)
                    
                    # 2. Special handling for aggressive trades
                    if signal in [-2, 2]:  # Your extra-aggressive signals
                        atr = self.data_fetcher.get_daily_atr(Config.SYMBOL)
                        if atr:
                            # Tighter stop for aggressive trades (2.5x ATR instead of 3x)
                            tighter_sl = (position['price_open'] - (2.5 * atr)) if position['type'] == 'buy' else (position['price_open'] + (2.5 * atr))
                            self.mt5.modify_position(
                                ticket=position['ticket'],
                                sl=tighter_sl,
                                tp=position['tp']
                            )
                
                # 3. Add to aggressive trades monitoring
                self._activate_aggressive_trades({
                    'ticket': ticket,
                    'symbol': Config.SYMBOL,
                    'entry_time': datetime.now(),
                    'initial_sl': position['sl'] if position else None
                })
            
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
                    'price_prediction_error': abs(fill_price - adjusted_price),
                    'actual_execution_price': fill_price,
                    'pnl': pnl,
                    'slippage': slippage,
                    'stop_loss_pips': stop_loss_pips,
                    'time': datetime.now()
                })

                # 13. Meta-learning outcome tracking (only if trade was successful)
                if hasattr(self, 'meta_learner') and self.meta_learner and result:
                    self.meta_learner.record_trade_outcome(
                        trade_type=trade_type,
                        entry_price=current_price_value,
                        exit_price=fill_price,
                        pnl=pnl
                    )

            except (KeyError, TypeError, ValueError) as e:
                print(f"Trade performance tracking failed: {str(e)}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Unexpected error in trade execution: {str(e)}")
            return False
            
    
    def _update_data_buffer(self, new_data: pd.DataFrame) -> None:
        """Update data buffer with robust validation and monitoring.
        
        Args:
            new_data: DataFrame with datetime index and market data
            
        Raises:
            TypeError: If input isn't a DataFrame
            ValueError: If empty or malformed data
        """
        # Validate input
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(new_data)}")
        if new_data.empty:
            raise ValueError("Empty data received")
        if not isinstance(new_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index")

        try:
            # Get existing indices efficiently
            existing_indices = (
                pd.concat(self.data_buffer).index 
                if self.data_buffer 
                else pd.DatetimeIndex([])
            )
            
            # Vectorized filtering
            new_samples = new_data[~new_data.index.isin(existing_indices)]
            
            if not new_samples.empty:
                # Validate new samples
                required_cols = {'open', 'high', 'low', 'close'}
                if not required_cols.issubset(new_samples.columns):
                    raise ValueError(f"Missing required columns: {required_cols - set(new_samples.columns)}")
                
                # Update buffer
                self.data_buffer.append(new_samples)
                
                # Enforce size limit
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer = self.data_buffer[-self.max_buffer_size:]
                
                # Monitor update
                self.performance_monitor.update({
                    'buffer_size': len(self.data_buffer),
                    'buffer_capacity': self.max_buffer_size,
                    'new_samples': len(new_samples),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Buffer update failed: {str(e)}", exc_info=True)
            self.performance_monitor.update({
                'buffer_error': str(e),
                'buffer_size': len(self.data_buffer),
                'status': 'error'
            })
            raise  # Re-raise for calling code to handle
    
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
            if self._update_trailing_stop(full_position):
                position_modified = True
    
        # Clear cache if positions were modified
        if position_modified:
            logger.debug("Position changes detected - clearing cache")
            self.risk_manager.clear_position_cache()

    def _activate_aggressive_trades(self, position: Dict):
        """Special handling for high-conviction trades"""
        atr = self.data_fetcher.get_daily_atr(position['symbol'])
        if not atr:
            return
            
        # Tighter stops for aggressive trades
        multiplier = 2.5  # vs normal 3x ATR
        if position['type'] == 'buy':
            new_sl = position['price_open'] - (multiplier * atr)
        else:
            new_sl = position['price_open'] + (multiplier * atr)
        
        # Move TP closer as well (optional)
        tp_adjustment = 1.5  # 1.5x ATR profit target
        if position['type'] == 'buy':
            new_tp = position['price_open'] + (tp_adjustment * atr)
        else:
            new_tp = position['price_open'] - (tp_adjustment * atr)
        
        self.mt5.modify_position(
            ticket=position['ticket'],
            sl=new_sl,
            tp=new_tp
        )

    def _handle_losing_position(self, position: Dict) -> bool:
        """Handle losing positions with volatility checks and partial closing"""
        # 1. First check volatility if available
        try:
            if hasattr(self.data_fetcher, 'get_volatility_index'):
                vol_idx = self.data_fetcher.get_volatility_index(position['symbol'])
                if vol_idx > 30:  # Threshold from config would be better
                    return self._close_position_fully(position)  # Renamed method
        except Exception as e:
            print(f"[WARNING] Volatility check failed: {str(e)}")

        # 2. Normal position management
        try:
            current_price = float(position['price_current'])
            entry_price = float(position['price_open'])
            buffer_pct = self.config.LOSS_BUFFER_PCT
            close_ratio = self.config.PARTIAL_CLOSE_RATIO

            # Breakeven logic
            breakeven_price = entry_price * (1 + buffer_pct) if position['type'] == 'buy' else entry_price * (1 - buffer_pct)
            
            if ((position['type'] == 'buy' and current_price >= breakeven_price) or
                (position['type'] == 'sell' and current_price <= breakeven_price)):
                return self.mt5.modify_position(
                    position['ticket'],
                    sl=breakeven_price,
                    tp=float(position['tp'])
                )
            
            # Partial close if no breakeven adjustment
            close_volume = round(float(position['volume']) * close_ratio, 2)
            if close_volume >= 0.01:  # Minimum lot size
                return self.mt5.close_position(position['ticket'], close_volume)
                
            return False

        except Exception as e:
            print(f"[ERROR] Position handling failed: {str(e)}")
            return False

    def _close_position_fully(self, position: Dict) -> bool:
        """Close entire position (helper method)"""
        try:
            return self.mt5.close_position(position['ticket'], position['volume'])
        except Exception as e:
            print(f"[ERROR] Full close failed: {str(e)}")
            return False
    
    def _update_trailing_stop(self, position: Dict) -> bool:
        """ATR-based trailing stop with internal data fetching"""
        symbol = position['symbol']
        
        # 1. Get required market data
        current_price = self.data_fetcher.get_current_price(symbol)
        if not current_price:
            print(f"[WARN] No price data for {symbol}")
            return False
            
        atr = self._get_current_atr(symbol)  # Use your existing cache-aware method
        if atr is None:
            print(f"[WARN] No ATR data for {symbol}")
            return False
        
        # 2. Calculate new stop
        new_sl: float
        modified = False
        
        if position['type'] == 'buy':
            new_sl = current_price['bid'] - (3 * atr)
            if new_sl > position['sl']:  # Only move stop up
                modified = self.mt5.modify_position(
                    ticket=position['ticket'],
                    sl=new_sl,
                    tp=position['tp']
                )
        else:
            new_sl = current_price['ask'] + (3 * atr)
            if new_sl < position['sl'] or position['sl'] == 0:  # Only move stop down
                modified = self.mt5.modify_position(
                    ticket=position['ticket'],
                    sl=new_sl,
                    tp=position['tp']
                )
        
        if modified:
            print(f"[INFO] Updated trailing stop for {symbol} to {new_sl:.5f}")
        return modified

    def _update_all_trailing_stops(self):
        """Update all active positions with trailing stops"""
        try:
            # Pass the symbol from config when calling get_open_positions
            open_positions = self.mt5.get_open_positions(symbol=self.config.SYMBOL)
            if open_positions is None:
                return False
                
            for position in open_positions:
                if position['symbol'] == self.config.SYMBOL:
                    self._update_trailing_stop(position)
            return True
        except Exception as e:
            print(f"Error updating trailing stops: {str(e)}")
            return False
        
    def show_performance(self, periodic: bool = False, include_metrics: bool = False) -> None:
        """Display comprehensive performance metrics with optional advanced metrics
        
        Args:
            periodic: If True, indicates this is a periodic update (may reduce output)
            include_metrics: If True, shows advanced risk/return metrics
        """
        if not hasattr(self, 'performance_monitor'):
            print("Performance monitoring not initialized")
            return
            
        # Get the full performance report
        report = self.performance_monitor.get_performance_report()
        
        # Core display logic
        print("\n=== PERFORMANCE REPORT ===" if not periodic else "\n=== Periodic Update ===")
        
        # Always show these core metrics
        core_metrics = [
            ('Equity', f"${report['equity_curve'][-1]:,.2f}"),
            ('Total Trades', report['total_trades']),
            ('Win Rate', report['win_rate']),
            ('Profit Factor', report['profit_factor']),
            ('Annualized Return', f"{self.performance_monitor._calc_annualized_return():.2f}%")
        ]
        
        for metric, value in core_metrics:
            print(f"{metric}: {value}")

        # Conditional advanced metrics
        if include_metrics:
            print("\n=== RISK METRICS ===")
            advanced_metrics = [
                ('Sharpe Ratio', report['sharpe_ratio']),
                ('Sortino Ratio', report['sortino_ratio']),
                ('Calmar Ratio', report['calmar_ratio']),
                ('Max Drawdown', report['max_drawdown']),
                ('Avg Winning Trade', report['advanced_stats']['avg_win']),
                ('Avg Losing Trade', report['advanced_stats']['avg_loss']),
                ('Win/Loss Ratio', report['advanced_stats']['win_loss_ratio'])
            ]
            
            for metric, value in advanced_metrics:
                print(f"{metric}: {value}")

        # Environment metrics (if available)
        if hasattr(self, 'trading_env') and self.trading_env is not None:
            try:
                print("\n=== ENVIRONMENT METRICS ===")
                env_metrics = [
                    ('Profit Factor', self.trading_env.get_profit_factor()),
                    ('Total Reward', self.trading_env.get_total_reward())
                ]
                
                for metric, value in env_metrics:
                    print(f"{metric}: {value:.2f}")
                    
                if env_metrics[0][1] < 1.5:  # Profit factor check
                    logger.warning("Environment profit factor below 1.5 - reconsider strategy")
                    
            except Exception as e:
                logger.error(f"Failed to get environment metrics: {str(e)}")

        # Warnings section (always shown)
        if report.get('warnings'):
            print("\n=== WARNINGS ===")
            for warning in report['warnings']:
                print(f"! {warning}")

        # Plotting logic (conditionally based on periodic flag)
        should_plot = not periodic or len(self.performance_monitor.metrics['all_trades']) % 100 == 0
        if should_plot:
            try:
                import matplotlib.pyplot as plt
                plt = self.performance_monitor.plot_equity_curve()
                
                # Enhanced plotting for periodic updates
                if periodic:
                    plt.title(f"Equity Curve (Last {len(self.performance_monitor.metrics['all_trades'])} Trades)")
                else:
                    plt.title("Full Equity Curve")
                    
                plt.show()
            except ImportError:
                print("Matplotlib not available - cannot show equity curve")
            except Exception as e:
                print(f"Failed to plot equity curve: {str(e)}")

    def reduce_position_sizes(self, reduction_factor: float) -> None:
        """Reduce all open positions by specified factor (0.5 = 50%)"""
        if not 0 < reduction_factor <= 1:
            raise ValueError("Reduction factor must be between 0 and 1")

        positions = self.mt5.get_open_positions(self.config.SYMBOL)
        for position in positions:
            try:
                # Calculate volume to close
                close_volume = float(position['volume']) * (1 - reduction_factor)
                close_volume = max(0.01, min(close_volume, float(position['volume'])))
                
                # Partially close the position
                success = self.mt5.close_position(
                    ticket=int(position['ticket']),
                    volume=close_volume
                )

                if success:
                    logger.info(f"Reduced position {position['ticket']} by {close_volume:.2f} lots")
                    self.performance_monitor.record_reduction(
                        position['ticket'],
                        close_volume
                    )
                else:
                    logger.error(f"Failed to reduce position {position['ticket']}")

            except Exception as e:
                logger.error(f"Error reducing position {position['ticket']}: {str(e)}")

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
            if self.mt5.close_position(pos['ticket'], pos['volume']):
                self.performance_monitor.record_reduction(
                    pos['ticket'],
                    pos['volume']  # Records full closure as 100% reduction
                )

    def trigger_emergency_stop(self):
        """
        Emergency stop with volatility protection.
        Triggers when:
        1. Manually called
        2. ATR exceeds 3x baseline (if baseline exists)
        """
        try:
            # 1. Check volatility condition if baseline exists
            baseline_atr = getattr(self.config, 'BASELINE_ATR', None)
            current_atr = None
            
            if baseline_atr is not None:
                current_atr = self.data_fetcher.get_daily_atr(self.config.SYMBOL)
                if current_atr and current_atr > (3 * baseline_atr):
                    print(f"[EMERGENCY] Volatility spike detected (ATR: {current_atr:.2f} > {3*baseline_atr:.2f})")

            # 2. Execute emergency procedures
            print("[EMERGENCY] Activating full shutdown sequence")
            self.risk_manager.emergency_stop.activate()
            self._close_all_positions()
            
            # 3. Print final status
            print(f"[EMERGENCY] All positions closed | Volatility: {current_atr or 'N/A'}")

        except Exception as e:
            print(f"[EMERGENCY ERROR] {str(e)}")
            # Final attempt to close positions
            try:
                self._close_all_positions()
            except Exception as final_error:
                print(f"[CRITICAL] Failed to close positions: {str(final_error)}")

    def _get_current_atr(self, symbol: str) -> Optional[float]:
        """Get cached ATR value with fallback to fresh calculation
        
        Args:
            symbol: Trading symbol (e.g., 'GOLD')
            
        Returns:
            Current ATR value or None if unavailable
        """
        # Try cache first
        cache_key = f"atr_{symbol}_14"
        if cache_key in self.data_fetcher._atr_cache:
            value, timestamp = self.data_fetcher._atr_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < 300:  # 5 min cache
                return value
        
        # Fallback to fresh calculation
        return self.data_fetcher.get_daily_atr(symbol, period=14)

    def reload_model(self) -> bool:
        """Public interface for model refresh"""
        return self.ml_model.refresh_model()
    
    def _add_to_position(self, position: Dict, current_price: Dict) -> bool:
        """Add to an existing position when conditions are favorable"""
        try:
            # 1. Check if we can add to position
            open_positions = self.mt5.get_open_positions(symbol=position['symbol'])
            if open_positions is None or len(open_positions) >= self.config.MAX_OPEN_POSITIONS:
                return False
                
            # 2. Calculate new position size (max 50% of original)
            original_size = float(position['volume'])
            
            # 3. Calculate dynamic stop loss
            atr = self.data_fetcher.get_daily_atr(position['symbol'])
            if atr is None:
                return False
                
            stop_distance = atr * self.config.ATR_STOP_LOSS_FACTOR
            
            # 4. Determine direction and price levels
            if position['type'] == 'buy':
                add_type = 'buy'
                price = current_price['ask']
                sl = price - stop_distance
            else:  # sell
                add_type = 'sell'
                price = current_price['bid']
                sl = price + stop_distance
                
            # 5. Calculate size to add (50% of original, with min size check)
            add_size = round(original_size * 0.5, 2)
            if add_size < 0.01:  # Minimum lot size
                return False
                
            # 6. Verify margin requirements
            if not self.risk_manager.check_margin_requirements(
                symbol=position['symbol'],
                volume=add_size,
                price=price,
                trade_type=add_type
            ):
                return False
                
            # 7. Execute additional position
            result = self.mt5.send_order(
                symbol=position['symbol'],
                order_type=add_type,
                volume=add_size,
                sl=sl,
                tp=0,  # No take profit - trailing stop will handle
                comment=f"ADD-{position['ticket']}"
            )
            
            # 8. Explicit boolean return
            return bool(result and result.get('retcode') == mt5.TRADE_RETCODE_DONE)
            
        except Exception as e:
            logger.error(f"Failed to add to position {position['ticket']}: {str(e)}")
            return False


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
            bot = self._init_bot_with_data(train_data, enable_meta_learning=False)
            
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
            data_fetcher=self.data_fetcher,
            config=bot.config
        )
        
        # 2. Convert data to numpy upfront
        features = data[bot.preprocessor.features].values.astype(np.float32)
        close_prices = data['close'].values.astype(np.float32)
        
        equity_curve: List[float] = []

        bot = self._init_bot_with_data(data, enable_meta_learning=False)
        
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
            bot = self._init_bot_with_data(data, enable_meta_learning=False)
            mc_results.append(self._run_backtest(bot, shuffled_data))
        
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
            name: self._run_backtest(
                self._init_bot_with_data(self._apply_scenario(resolved_data.copy(), params), enable_meta_learning=False),
                resolved_data
            )
            for name, params in resolved_scenarios.items()
        }

    def _apply_scenario(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply market scenario (flash crash, low vol, etc.)"""
        if params.get('type') == 'vol_spike':
            data['close'] = data['close'] * (1 + np.random.normal(0, params['size'], len(data)))
        elif params.get('type') == 'trend':
            data['close'] = data['close'] * (1 + np.linspace(0, params['slope'], len(data)))
        return data

    def _init_bot_with_data(self, data: pd.DataFrame, enable_meta_learning: bool = False) -> TradingBot:
        """Initialize bot with preloaded data
        
        Args:
            data: DataFrame containing historical data
            enable_meta_learning: Whether to enable meta-learning adaptation (False for backtests)
        """
        bot = TradingBot(
            config=self.config,
            mt5_connector=self.mt5_connector,
            data_fetcher=self.data_fetcher,
            preprocessor=DataPreprocessor(  # Updated instantiation
                config=self.config,
                data_fetcher=self.data_fetcher
            )
        )
        
        # Initialize ML model with meta-learning disabled for backtests
        bot.ml_model = GoldMLModel(
            config=self.config,
            monitor=PerformanceMonitor(self.config),
            data_fetcher=self.data_fetcher,
            enable_meta_learning=enable_meta_learning  # False for backtests
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
    """Get volatility for slippage simulation (uses DataFetcher's volatility calculations)
    
    Note: Maintains separate cache from RiskManager as simulation needs may differ
    from actual trading conditions.
    """
    
    def __init__(self, initial_balance: float, config: Config, data_fetcher: DataFetcher):  # CHANGED: Added config parameter
        self.balance = initial_balance
        self.positions = []
        self.data_fetcher = data_fetcher
        self._volatility_cache = {}
        self.trade_history = []
        self.equity = [initial_balance]
        self.current_price = None
        self.config = config  # CHANGED: Store config

        self.fill_stats = {
            'total_requests': 0,
            'partial_fills': 0,
            'avg_fill_ratio': 0,
            'symbol_stats': defaultdict(lambda: {'count': 0, 'total_filled': 0})
        }
        
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

    def log_fill(self, symbol: str, requested: float, filled: float) -> None:
        """Update fill statistics with comprehensive tracking
        
        Args:
            symbol: Trading symbol (e.g. "GOLD")
            requested: Original order size
            filled: Actually filled amount
            
        Returns:
            None (updates internal stats)
        """
        if requested <= 0 or filled < 0:
            logger.error(f"Invalid fill values: requested={requested}, filled={filled}")
            return

        fill_ratio = filled / requested
        
        # Update global stats
        self.fill_stats['total_requests'] += 1
        self.fill_stats['partial_fills'] += 1 if fill_ratio < 0.99 else 0
        
        # Calculate new rolling average (fixed parenthesis)
        prev_avg = self.fill_stats['avg_fill_ratio']
        total_reqs = self.fill_stats['total_requests']
        self.fill_stats['avg_fill_ratio'] = (
            (prev_avg * (total_reqs - 1) + fill_ratio) 
            / total_reqs
        )
        
        # Update symbol-specific stats
        sym_stats = self.fill_stats['symbol_stats'][symbol]
        sym_stats['count'] += 1
        sym_stats['total_filled'] += fill_ratio
        sym_stats['avg_fill'] = sym_stats['total_filled'] / sym_stats['count']
        
        # Track fill ratio distribution
        ratio_bucket = round(fill_ratio * 10) / 10  # 0.0-1.0 in 0.1 increments
        if 'distribution' not in self.fill_stats:
            self.fill_stats['distribution'] = defaultdict(int)
        self.fill_stats['distribution'][ratio_bucket] += 1

    def _simulate_fill(self, symbol: str, size: float, is_buy: bool) -> float:
        """Simulate partial fills based on market conditions"""
        # Base fill probability (90% in normal markets)
        base_fill_prob = 0.9
        
        # Adjust for volatility
        vol_factor = 1.0 - (self._get_volatility(symbol) * 0.5)
        
        # Time-of-day factor (Asian session has lower liquidity)
        hour = datetime.now().hour
        liquidity_factor = 0.7 if 0 <= hour <= 5 else 1.0
        
        # Size penalty (larger orders harder to fill)
        size_penalty = min(1.0, 1.0 / (size ** 0.2))
        
        fill_prob = base_fill_prob * vol_factor * liquidity_factor * size_penalty
        fill_ratio = np.random.beta(fill_prob * 10, (1-fill_prob) * 10)
        
        return max(0.01, round(size * fill_ratio, 2))  # Ensure minimum 0.01 lot size

    def _get_volatility(self, symbol: str) -> float:
        """Get cached volatility reading"""
        if symbol not in self._volatility_cache:
            self._volatility_cache[symbol] = self.data_fetcher.get_volatility(symbol)
        return self._volatility_cache[symbol]
    

class OrderBookSimulator:
    def get_impact(self, volume: float) -> float:
        """Calculate price impact using VWAP"""
        return volume * 0.0001  # Example: 1bp per standard lot
    

class EmergencyStop:
    def __init__(self):
        self._active = False
        self.activation_time: Optional[datetime] = None  # Explicit type hint
        self.reason: Optional[str] = None
        self._lock = threading.Lock()  # Thread safety
        self._subscribers: List[Callable[[dict], None]] = []  # Callback functions
        self._timeout: Optional[float] = None  # Optional auto-reset timeout
        
    def activate(self, reason: str = "Manual activation", timeout: Optional[float] = None) -> None:
        """Activate emergency stop with optional auto-reset timeout.
        
        Args:
            reason: Detailed explanation of activation
            timeout: Seconds until auto-reset (None for manual only)
        """
        with self._lock:
            if not self._active:  # Only log first activation
                logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
                
            self._active = True
            self.activation_time = datetime.now(pytz.UTC)  # Using your pytz import
            self.reason = reason
            self._timeout = timeout
            
            # Notify subscribers
            for callback in self._subscribers:
                try:
                    callback(self.status())
                except Exception as e:
                    logger.error(f"Emergency stop callback failed: {str(e)}")

    def deactivate(self) -> None:
        """Manual deactivation"""
        with self._lock:
            if self._active:
                logger.warning(f"EMERGENCY STOP DEACTIVATED. Was active since {self.activation_time}")
                self._active = False
                self._timeout = None
                
                for callback in self._subscribers:
                    try:
                        callback(self.status())
                    except Exception as e:
                        logger.error(f"Deactivation callback failed: {str(e)}")

    def check(self) -> bool:
        """Check status with optional auto-reset logic"""
        with self._lock:
            if self._active and self._timeout and self.activation_time is not None:
                elapsed = (datetime.now(pytz.UTC) - self.activation_time).total_seconds()
                if elapsed >= self._timeout:
                    self.deactivate()
            return self._active

    def status(self) -> dict:
        """Detailed status with timeout info"""
        with self._lock:
            status = {
                "active": self._active,
                "since": self.activation_time,
                "reason": self.reason,
                "timeout_remaining": None
            }
            
            if self._active and self._timeout and self.activation_time is not None:
                status["timeout_remaining"] = max(
                    0, 
                    self._timeout - (datetime.now(pytz.UTC) - self.activation_time).total_seconds()
                )
                
            return status

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Register callback for status changes"""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]) -> None:
        """Remove callback"""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def __str__(self) -> str:
        status = self.status()
        if status["active"]:
            timeout_info = ""
            if status["timeout_remaining"] is not None:
                timeout_info = f", auto-reset in {status['timeout_remaining']:.0f}s"
            return (f"EmergencyStop(ACTIVE since {status['since']}, "
                    f"reason: {status['reason']}{timeout_info})")
        return "EmergencyStop(INACTIVE)"




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

    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}", exc_info=True)
        sys.exit(1)  # Exit with error code

    # Add command-line interface for backtesting
    parser = argparse.ArgumentParser(description='Trading Bot with Backtesting')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--walkforward', action='store_true', help='Run walkforward validation')
    parser.add_argument('--montecarlo', action='store_true', help='Run Monte Carlo simulations')
    parser.add_argument('--periods', type=int, default=500, help='Number of periods to backtest')
    args = parser.parse_args()

    try: 
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
                last_trailing_update = time.time() 
                
                # Main trading loop
                while True:
                    try:
                        # Run one iteration of the trading logic
                        bot.run()

                        # Update trailing stops every 60 seconds
                        if time.time() - last_trailing_update >= 60:
                            if bot._update_all_trailing_stops():
                                last_trailing_update = time.time()  # Only update on success
                        
                        # Check for retraining every N trades (e.g., every 50 trades)
                        if trade_count % 50 == 0:  
                            # First check if we have enough new data
                            new_data = data_fetcher.get_historical_data(
                                symbol=config.SYMBOL,
                                timeframe=config.TIMEFRAME,
                                num_candles=config.MIN_RETRAIN_SAMPLES * 2  # Get extra data buffer
                            )
                            
                            if new_data is not None and len(new_data) >= config.MIN_RETRAIN_SAMPLES:
                                # Calculate current metrics for logging (regardless of retraining decision)
                                current_metrics = bot.ml_model.calculate_metrics()
                                logger.debug(f"Current performance metrics: {current_metrics}")
                                
                                # Check both conditions: data freshness AND performance metrics
                                data_condition = bot.ml_model.should_retrain(len(new_data))
                                performance_condition = bot.ml_model.should_retrain_based_on_performance()
                                
                                if data_condition and performance_condition:
                                    logger.info("Performance-based retraining triggered...")
                                    logger.info(f"Triggering metrics: {current_metrics}")
                                    
                                    try:
                                        # Data preprocessing
                                        new_data = new_data.drop_duplicates().sort_index()
                                        new_data = new_data[~new_data.index.duplicated(keep='last')]
                                        
                                        # Execute retraining
                                        if bot.ml_model.retrain_model(new_data):
                                            logger.info("Retraining completed successfully")
                                            
                                            # Model reload and validation
                                            if bot.reload_model():
                                                new_metrics = bot.ml_model.calculate_metrics()
                                                logger.info(f"Post-retraining improvement: "
                                                        f"Sharpe: {current_metrics['sharpe_ratio']:.2f}→{new_metrics['sharpe_ratio']:.2f} "
                                                        f"Win%: {current_metrics['win_rate']:.1f}→{new_metrics['win_rate']:.1f}")
                                            else:
                                                logger.warning("Model reload failed after retraining")
                                        else:
                                            logger.warning("Retraining failed - keeping previous model")
                                            
                                    except Exception as e:
                                        logger.error(f"Retraining error: {str(e)}", exc_info=True)
                                        # Attempt to continue trading with previous model
                                else:
                                    logger.debug(f"Retraining conditions not met - "
                                            f"Data: {'✓' if data_condition else '✗'} "
                                            f"Performance: {'✓' if performance_condition else '✗'}")
                            else:
                                logger.debug(f"Insufficient new data for retraining - "
                                        f"Has: {len(new_data) if new_data is not None else 0}/"
                                        f"Needed: {config.MIN_RETRAIN_SAMPLES}")
                        
                        # Increment trade count
                        trade_count += 1
                        
                        # Periodic reporting
                        if trade_count % 100 == 0:
                            print(f"\nCompleted {trade_count} trades")
                            bot.show_performance(include_metrics=True)
                            
                        # Adaptive sleep based on market volatility
                        sleep_duration = max(0.1, min(1.0, 1.0 - data_fetcher.get_volatility(
                            symbol=config.SYMBOL,
                            lookback_periods=20
                        ) * 0.1))
                        
                    except Exception as e:
                        logger.error(f"Error in trading loop iteration: {str(e)}", exc_info=True)
                        if not mt5_connector.ensure_connected():
                            logger.error("Failed to reconnect to MT5 - retrying in 10 seconds")
                            time.sleep(10)
                        
            except KeyboardInterrupt:
                print("\nShutting down...")
                print("\n=== FINAL PERFORMANCE REPORT ===")
                bot.show_performance(include_metrics=True)
            except Exception as e:
                logger.critical(f"Fatal error in trading bot: {str(e)}", exc_info=True)
            finally:
                bot.mt5.disconnect()
                logger.info("MT5 connection closed")

    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}", exc_info=True)
        sys.exit(1)


"""J  
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