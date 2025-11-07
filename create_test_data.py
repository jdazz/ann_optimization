import json
import random

# Load your full dataset
with open("earthquakes.txt", "r") as f:
    data = json.load(f)

# The entries are in the 'features' list
entries = data["features"]


# Select 13 random entries for testing
test_entries = random.sample(entries, 2)

# The rest go into training
train_entries = [e for e in entries if e not in test_entries]

# Save training dataset
with open("training.json", "w") as f:
    json.dump({"type": data["type"], "features": train_entries}, f, indent=2)

# Save testing dataset
with open("testing.json", "w") as f:
    json.dump({"type": data["type"], "features": test_entries}, f, indent=2)

print("Training and testing datasets saved successfully!")