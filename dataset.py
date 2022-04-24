import torch
from torch.utils.data import Dataset
from data_utils import *

class TrainDataset(Dataset):
    def __init__(self, tokenizer, max_len, merged_dataset):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_text = merged_dataset["feature_text"].values
        self.patient_notes = merged_dataset["pn_history"].values
        self.annotation_lengths = merged_dataset["annotation_length"].values
        self.locations = merged_dataset["location"].values
    
    def __len__(self):
        return len(self.feature_text)

    def __getitem__(self, item):
        inputs = self.tokenizer(self.patient_notes[item], self.feature_text[item], add_special_tokens=True, max_length=self.max_len, padding="max_length", return_offsets_mapping=False)
        #inputs = self.tokenizer(self.patient_notes[item], self.feature_text[item], add_special_tokens=True, max_length=self.max_len, padding="max_length")
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        encoded = self.tokenizer(self.patient_notes[item], add_special_tokens=True, max_length=self.max_len, padding="max_length", return_offsets_mapping=True)
        offset_mapping = encoded['offset_mapping']
        ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
        label = np.zeros(len(offset_mapping))
        label[ignore_idxes] = -1
        if self.annotation_lengths[item] != 0:
            for location in self.locations[item]:
                for i in [s.split() for s in location.split(';')]:
                    start_idx = -1
                    end_idx = -1
                    start, end = int(i[0]), int(i[1])
                    for idx in range(len(offset_mapping)):
                        if (start_idx == -1) & (start < offset_mapping[idx][0]):
                            start_idx = idx - 1
                        if (end_idx == -1) & (end < offset_mapping[idx][1]):
                            end_idx = idx + 1
                    if start_idx == -1:
                        start_idx = end_idx
                    if (start_idx != -1) & (end_idx != -1):
                        label[start_idx:end_idx] = 1

        return inputs, label

class TestDataset(Dataset):
    def __init__(self, tokenizer, max_len, merged_dataset):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_text = merged_dataset["feature_text"].values
        self.patient_notes = merged_dataset["pn_history"].values
    
    def __len__(self):
        return len(self.feature_text)

    def __getitem__(self, item):
        inputs = self.tokenizer(self.patient_notes[item], self.feature_text[item], add_special_tokens=True, max_length=self.max_len, padding="max_length", return_offsets_mapping=False)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        
        return inputs