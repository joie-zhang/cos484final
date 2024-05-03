from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor

# Function to load a dataset and save it as a CSV file
def save_dataset(dataset_name, file_name, batch_size=10000):
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)

        # Get the first split (assuming 'train')
        # Adjust the split if necessary based on what the dataset contains
        if 'train' in dataset:
            split = 'train'
        else:
            # If no 'train' split, take the first available split
            split = next(iter(dataset.keys()))

        # Get the total number of examples in the dataset
        total_examples = len(dataset[split])

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # Open the CSV file in write mode
        with open(file_name, 'w') as file:
            # Write the header
            file.write(','.join(dataset[split].features.keys()) + '\n')

            # Iterate over the dataset in batches
            for i in tqdm(range(0, total_examples, batch_size)):
                # Load a batch of examples
                batch = dataset[split].select(range(i, min(i + batch_size, total_examples)))

                # Convert the batch to a DataFrame
                df = pd.DataFrame(batch)

                # Append the batch to the CSV file
                df.to_csv(file, header=False, index=False)

        print(f"Dataset {dataset_name} has been saved as {file_name}")

    except Exception as e:
        print(f"An error occurred while processing dataset {dataset_name}: {str(e)}")

# List of datasets to save
datasets = [
    ("Skylion007/openwebtext", "data/openwebtext.csv")
]

# Create a thread pool with a specified number of worker threads
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit the save_dataset function for each dataset to the thread pool
    futures = [executor.submit(save_dataset, dataset_name, file_name) for dataset_name, file_name in datasets]

    # Wait for all the futures to complete
    for future in tqdm(futures, total=len(datasets), desc="Datasets"):
        future.result()
