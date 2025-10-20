import torch
from datasets import load_dataset
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy

class TextP2PFLDataset(P2PFLDataset):
    """Custom P2PFLDataset for text data (SST-2 sentiment analysis)"""
    
    def __init__(self, dataset_name="stanfordnlp/sst2", split="train", **kwargs):
        # Load the dataset
        ds = load_dataset(dataset_name)[split]
        
        # Convert to the format expected by P2PFLDataset
        # We need to create a list of dictionaries with the data
        data_list = []
        for i in range(len(ds)):
            data_list.append({
                "text": ds[i]["sentence"],
                "label": ds[i]["label"]
            })
        
        # Initialize the parent class with the data
        super().__init__(data_list, **kwargs)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = self.data[idx]
        return {
            "text": item["text"],
            "label": torch.tensor(item["label"], dtype=torch.long)
        }
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data)
