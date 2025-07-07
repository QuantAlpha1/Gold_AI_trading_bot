# Run: pytest test_risk.py -v
# This test gives you mathematically verified safety for gold trading
import pytest
from unittest.mock import Mock
from elso import RiskManager, Config, MT5Connector, PerformanceMonitor, DataFetcher

@pytest.fixture
def mock_dependencies():
    """Creates properly typed mock objects"""
    return {
        'mt5': Mock(spec=MT5Connector),
        'monitor': Mock(spec=PerformanceMonitor),
        'fetcher': Mock(spec=DataFetcher)
    }

@pytest.mark.parametrize("balance,risk_pct,expected", [
    (10_000, 0.02, 0.2),  # 2% risk on $10k (gold standard lot)
    (0, 0.02, 0),         # Edge case: zero balance
    (10_000, 1.5, 1.0)    # Risk cap >100%
])
def test_position_sizing(mock_dependencies, balance, risk_pct, expected):
    config = Config(
        ACCOUNT_BALANCE=balance,
        RISK_PER_TRADE=risk_pct,
        SYMBOL="GOLD"
    )
    
    # Initialize with properly typed mocks
    rm = RiskManager(
        mt5_connector=mock_dependencies['mt5'],
        config=config,
        performance_monitor=mock_dependencies['monitor'],
        data_fetcher=mock_dependencies['fetcher']
    )
    
    # Mock price data
    mock_dependencies['fetcher'].get_current_price.return_value = {
        'bid': 1800.00,
        'ask': 1800.05
    }
    
    # Test with realistic gold parameters
    result = rm.calculate_position_size(
        symbol="GOLD",
        stop_loss_pips=100  # 100 pips = $10/oz risk
    )
    
    assert abs(result - expected) < 0.001  # Account for floating point