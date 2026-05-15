
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def check_pending_batches():
    if not os.path.exists("pending_batches.json"):
        print("No pending_batches.json found.")
        return

    with open("pending_batches.json", "r") as f:
        pending = json.load(f)

    print(f"Total batches in pending_batches.json: {len(pending)}")
    
    status_counts = {}
    node_counts = {}
    
    for batch_id, meta in pending.items():
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            node = meta.get("node_name", "unknown")
            if status != "completed":
                node_counts[node] = node_counts.get(node, 0) + 1
                print(f"Batch {batch_id} for {meta['variation_id']} - {node} is {status}")
        except Exception as e:
            print(f"Error retrieving {batch_id}: {e}")

    print("\nStatus Summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
        
    print("\nNon-completed Nodes Summary:")
    for node, count in node_counts.items():
        print(f"  {node}: {count}")

if __name__ == "__main__":
    check_pending_batches()
