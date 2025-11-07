import pandas as pd
import json
import random

# Load your Excel file
excel_file = "measurements2.xlsx"  # replace with your file path
sheet_name = 0  # or the name of your sheet, e.g., "Sheet1"
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Select only the columns you want
input_columns = ['distance', 'speed', 'temp_inside', 'temp_outside', 'AC', 'rain']
output_columns = ['consume']

df_inputs = df[input_columns]
df_outputs = df[output_columns]

# Combine input and output into a single list of dictionaries for each row
full_data = [
    {**df_inputs.iloc[i].to_dict(), **df_outputs.iloc[i].to_dict()}
    for i in range(len(df))
]

# Shuffle and split
random.seed(42)  # for reproducibility
test_size = int(len(full_data) * 0.2)  # 20% for testing
test_entries = random.sample(full_data, test_size)
train_entries = [row for row in full_data if row not in test_entries]

# Save training dataset
with open("training.json", "w") as f:
    json.dump([train_entries], f, indent=2)

# Save testing dataset
with open("testing.json", "w") as f:
    json.dump([test_entries], f, indent=2)

print("Training and testing datasets saved successfully!")