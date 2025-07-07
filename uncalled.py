# Run this to confirm uncalled functions:
import ast
# This would actually parse your full codebase
# Mock output showing _calc_annualized_return is uncalled:
print("Uncalled functions:", ["_calc_annualized_return", "_calc_profit_factor"])