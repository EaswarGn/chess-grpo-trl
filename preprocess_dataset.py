from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from prompts import system_prompt, build_user_prompt, build_completion


def process_split(dataset, split_name, n=None, color="#00ff00"):
    """
    Processes a dataset split by appending prompt/completion columns to the existing data.
    """
    processed_data = []
    
    if split_name == "validation" or split_name == "test":
        return
        n = 50
        
    if split_name == "train":
        print(f"Shuffling {split_name} split...")
        dataset = dataset.shuffle(seed=42)  # seed ensures reproducibility
    
    # 1. Limit the dataset to n items if specified
    if n is not None:
        # We use .select() for HF datasets or simple slicing for lists
        dataset = dataset.select(range(min(n, len(dataset))))
    
    for example in tqdm(dataset, desc=f"Processing {split_name} split", colour=color):
        # Construct the new columns
        prompt = [
            {"content": system_prompt, "role": "system"},
            {"content": build_user_prompt(example["fen"]), "role": "user"}
        ]
        
        example["prompt"] = prompt
        
        processed_data.append(example)
    
    return Dataset.from_list(processed_data)


def process_dataset(dataset_id: str = "codingmonster1234/chess-puzzles-rlvr", n: int = None) -> DatasetDict:
    # 1. Load the raw datasets
    print("Loading source datasets...")
    raw_dataset = load_dataset(dataset_id)
    
    processed_splits = {}
    
    for split in raw_dataset:
        # 2. Pass n down to the split processor
        processed_split = process_split(
            raw_dataset[split], 
            split_name=split, 
            n=n, 
            color="#2ecc71"
        )
        
        processed_splits[split] = processed_split

    # 3. Combine into a DatasetDict
    dataset_dict = DatasetDict(processed_splits)
    return dataset_dict