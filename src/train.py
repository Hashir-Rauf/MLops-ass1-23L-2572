from train_23L2572 import load_data
path = input("Enter the path of the CSV file: ")
data = load_data(path)
if data is not None:
    print("Data Loaded Successfully!")
    print(data.head())
else:
    print("Failed to load data.")