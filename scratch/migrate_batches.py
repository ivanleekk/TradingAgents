import json
import os

def migrate_pending_batches():
    combined_file = "pending_batches.json"
    if not os.path.exists(combined_file):
        print("No combined pending_batches.json found. Nothing to migrate.")
        return

    with open(combined_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Combined file is empty or invalid JSON.")
            return

    # Group by variation_id
    variations = {}
    for bid, meta in data.items():
        var_id = meta.get("variation_id", "UNKNOWN")
        if var_id not in variations:
            variations[var_id] = {}
        variations[var_id][bid] = meta

    # Write separate files
    for var_id, var_data in variations.items():
        var_file = f"pending_batches_{var_id}.json"
        
        # If the target file already exists, merge them
        if os.path.exists(var_file):
            with open(var_file, "r") as f:
                try:
                    existing_data = json.load(f)
                    var_data.update(existing_data)
                except:
                    pass
                    
        with open(var_file, "w") as f:
            json.dump(var_data, f, indent=4)
        print(f"Migrated {len(var_data)} batches to {var_file}")

    # Backup the old file
    os.rename(combined_file, f"{combined_file}.bak")
    print(f"Original {combined_file} backed up to {combined_file}.bak")

if __name__ == "__main__":
    migrate_pending_batches()
