# Run gridsearch

uv run run_systematic_hyperparam_grid.py --risk-aversion-values 0.5,1,2,5 --l2-gamma-values 0.1,0.3,0.5,1.0 --optimizers max-sharpe,max-quadratic-utility --output-dir results/hyperparam_grid --parallel-workers 10

uv run run_systematic_hyperparam_grid.py --risk-aversion-values 2 --l2-gamma-values 0.1,0.3,0.5,0.7,0.9,1 --optimizers max-sharpe,max-quadratic-utility --output-dir results/hyperparam_grid --parallel-workers 10

uv run run_systematic_hyperparam_grid.py --risk-aversion-values 1,2,5 --l2-gamma-values 0.1,0.3,0.5,0.7,0.9,1 --optimizers max-sharpe,max-quadratic-utility --output-dir results/hyperparam_grid --parallel-workers 10
