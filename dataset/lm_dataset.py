import json
from torch.utils.data import Dataset
import torch
import os
import random
from dataset import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning from huggingface tokenizers

class PreTrainDataset(Dataset):
    # init -> __len__ -> __getitem__
    def __init__(self,data_path, tokenizer,max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length # max_length for GPU
        self.samples = load_dataset("json",data_files=data_path,split="train")# lazy load dataset
    def __len__(self):
        return len(self.samples)
    
    # 1. got row of json
    # 2.text ->(tokenizer) input_ids
    # 3.BOS EOS PAD
    # 4.labels(prevent PAD from contributing to loss)
    # 5.attention_mask(tell PAD locations)
    # 6.return inpout_ids, attention_mask, labels
    def __getitem__(self, idx):
        
        #1.
        samples = self.samples[idx]

        #2.
        tokens= self.tokenizer(
            str(samples["text"]),
            add_special_tokens=True,
            max_length=self.max_length -2,
            truncation=True, # cut off if exceed max_length
        ).input_ids

        #3.
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens)) # pad to max_length
        input_ids = torch.tensor(input_ids,dtype=torch.long)

        #4.
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100 # ignore pad token in loss

        #5.
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long() # 1 for real tokens, 0 for pad

        #6.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
