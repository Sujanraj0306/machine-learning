import random
import csv

# Number of samples
num_samples = 1000

# Define categorical values
categories = ["A", "B", "C"]

# Define CSV headers
headers = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'category', 'target']

data = []
for i in range(num_samples):
    f1 = round(random.uniform(1, 300), 2)
    f2 = random.randint(29, 45)
    f3 = round(random.uniform(1, 300), 2)
    f4 = random.randint(29, 45)
    f5 = round(random.uniform(1, 300), 2)
    f6 = round(random.uniform(1, 300), 2)
    f7 = random.randint(29, 45)
    f8 = random.randint(29, 45)
    f9 = round(random.uniform(1, 300), 2)
    f10 = random.randint(29, 45)
    
    # 90% chance of target = 1, 10% chance of target = 0
    target = 1 if random.random() < 0.9 else 0
    #  target = random.choice([1,0])
    
    # Random category
    category = random.choice(categories)
    
    data.append({
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'f5': f5,
        'f6': f6,
        'f7': f7,
        'f8': f8,
        'f9': f9,
        'f10': f10,
        'category': category,
        'target': target
    })

# Save to CSV
with open('generated_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)

print("Dataset created successfully: generated_data.csv")

# data = pd.read_csv("generated_data.csv")
# X = data[['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']]
# y = data['target']