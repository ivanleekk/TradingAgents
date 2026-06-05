import sqlite3
import os
import sys

def reset_checkpoint(thread_id: str, db_path: str = "data_dir/checkpoints/checkpoints.db"):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Connecting to database: {db_path}")
    with sqlite3.connect(db_path) as conn:
        # Check if thread_id exists
        cursor = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,))
        count = cursor.fetchone()[0]
        if count == 0:
            # Let's search with LIKE to help the user
            like_cursor = conn.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?", (f"%{thread_id}%",))
            matches = [row[0] for row in like_cursor.fetchall()]
            if matches:
                print(f"No exact match for '{thread_id}'. Did you mean one of these?")
                for m in matches:
                    print(f"  - {m}")
            else:
                print(f"No checkpoints found matching '{thread_id}'.")
            return

        print(f"Found {count} checkpoint(s) for thread '{thread_id}'.")
        
        # Deleting from checkpoints
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        # Deleting from writes
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        
        conn.commit()
        print(f"Successfully deleted all checkpoints and writes for '{thread_id}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reset_checkpoint.py <thread_id>")
        print("Example: python reset_checkpoint.py C_SPY_2024-03-25")
        sys.exit(1)
        
    thread_id = sys.argv[1]
    reset_checkpoint(thread_id)
