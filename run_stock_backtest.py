"""
Single stock backtesting script for SLURM parallel execution
Usage: python run_stock_backtest.py <STOCK_SYMBOL> <START_DATE> <END_DATE>
Example: python run_stock_backtest.py AAPL 2020-01-01 2024-12-31
"""

import sys
import os
from datetime import datetime, timedelta
import polars as pl
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python run_stock_backtest.py <STOCK_SYMBOL> <START_DATE> <END_DATE>"
        )
        print("Example: python run_stock_backtest.py AAPL 2020-01-01 2024-12-31")
        sys.exit(1)

    stock = sys.argv[1]
    start_date_str = sys.argv[2]
    end_date_str = sys.argv[3]

    # Parse dates
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    print(f"Starting backtest for {stock} from {start_date} to {end_date}")

    # Load environment variables
    load_dotenv()

    # Create custom config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openrouter"
    config["backend_url"] = "https://openrouter.ai/api/v1"
    config["embedding_backend_url"] = "https://api.openai.com/v1"
    config["deep_think_llm"] = "qwen/qwen3-235b-a22b:free"
    config["quick_think_llm"] = "z-ai/glm-4.5-air:free"
    config["max_debate_rounds"] = 1
    config["online_tools"] = True

    # Initialize trading graph
    print(f"Initializing TradingAgentsGraph for {stock}...")
    ta = TradingAgentsGraph(
        selected_analysts=["news", "market"], debug=True, config=config
    )

    # Generate date range (weekdays only)
    test_range = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            test_range.append(current_date.isoformat())
        current_date += timedelta(days=1)

    print(f"Testing {len(test_range)} trading days for {stock}")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Initialize dataframe
    df = pl.DataFrame()

    # Run backtests
    successful_runs = 0
    failed_runs = 0

    try:
        for i, test_date in enumerate(test_range):
            try:
                print(
                    f"[{i+1}/{len(test_range)}] Propagating {stock} for date: {test_date}"
                )
                _, decision = ta.propagate(stock, test_date)

                # Save decision to dataframe
                df = pl.concat(
                    [
                        df,
                        pl.DataFrame(
                            {
                                "stock": [stock],
                                "test_date": [test_date],
                                "decision": [decision],
                            }
                        ),
                    ]
                )

                successful_runs += 1

                # Save checkpoint every 10 days
                if (i + 1) % 10 == 0:
                    checkpoint_file = f"results/{stock}_decisions_{start_date_str}_{end_date_str}_checkpoint.csv"
                    df.write_csv(checkpoint_file)
                    print(f"Checkpoint saved: {checkpoint_file}")

            except Exception as e:
                print(f"Error on {test_date} for {stock}: {e}")
                failed_runs += 1
                # Save partial results on error
                if len(df) > 0:
                    error_file = f"results/{stock}_decisions_{start_date_str}_{end_date_str}_partial.csv"
                    df.write_csv(error_file)
                    print(f"Partial results saved: {error_file}")
                continue

    except KeyboardInterrupt:
        print(f"\nBacktest interrupted for {stock}")
    finally:
        # Save final results
        if len(df) > 0:
            output_file = (
                f"results/{stock}_decisions_{start_date_str}_{end_date_str}.csv"
            )
            df.write_csv(output_file)
            print(f"\n{'='*60}")
            print(f"Final results saved: {output_file}")
            print(f"Stock: {stock}")
            print(f"Successful runs: {successful_runs}")
            print(f"Failed runs: {failed_runs}")
            print(f"Total trading days: {len(test_range)}")
            print(f"{'='*60}")
        else:
            print(f"No results to save for {stock}")
            sys.exit(1)


if __name__ == "__main__":
    main()
