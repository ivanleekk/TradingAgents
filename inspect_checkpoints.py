import sqlite3
import json
import os
from typing import Dict

def inspect_db(db_path: str = "data_dir/checkpoints/checkpoints.db"):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"--- Inspecting Checkpointer: {db_path} ---")
    
    with sqlite3.connect(db_path) as conn:
        # 1. Total Checkpoints
        count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        print(f"Total checkpoints stored: {count}")

        # 2. Unique Threads (Tasks)
        threads = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        print(f"Unique tasks (thread_ids) in DB: {len(threads)}")

        # 3. Sample of threads
        if threads:
            print("\nLatest 10 tasks in DB:")
            for t in threads[-10:]:
                print(f" - {t[0]}")

        # 4. Progress by Variation and Step
        print("\n--- Detailed Progress Breakdown ---")
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        serde = JsonPlusSerializer()

        for var in ["A", "B", "C", "D"]:
            # Get the latest checkpoint for each thread in this variation
            rows = conn.execute("""
                SELECT thread_id, checkpoint 
                FROM checkpoints 
                WHERE thread_id LIKE ? 
                GROUP BY thread_id 
                HAVING checkpoint_id = MAX(checkpoint_id)
            """, (f"{var}_%",)).fetchall()
            
            if not rows:
                continue

            steps = {}
            for tid, cp_blob in rows:
                cp = serde.loads(cp_blob)
                # 'next' tells us which node is about to run
                next_nodes = cp.get("next", [])
                step_name = next_nodes[0] if next_nodes else "Finalized/Completed"
                steps[step_name] = steps.get(step_name, 0) + 1
            
            print(f"\nVariation {var} ({len(rows)} tasks):")
            for step, count in sorted(steps.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {step}: {count} tasks")

def main():
    db_path = "data_dir/checkpoints/checkpoints.db"
    inspect_db(db_path)

if __name__ == "__main__":
    main()
