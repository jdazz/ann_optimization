import json

# Load your dataset
with open("training.json", "r") as f:
    data = json.load(f)

# Get the list of all keys from the first entry (assuming all entries have same keys)
all_keys = list(data[0][0].keys())

# Define output variable
Output_Variable = ['value']  # Change this if your target key is different

# Input variables are all keys except the output
Input_Variable = [key for key in all_keys if key not in Output_Variable]

print("Input Variables:", Input_Variable)
print("Output Variable:", Output_Variable)