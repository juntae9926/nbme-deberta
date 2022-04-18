import torch
from torch.utils.data import Dataset
from data_utils import *

class TrainDataset(Dataset):
    def __init__(self, Config, merged_dataset):
        self.config = Config
        self.feature_text = merged_dataset["feature_text"].values
        self.patient_notes = merged_dataset["pn_history"].values
        self.annotation_lengths = merged_dataset["annotation_length"].values
        self.locations = merged_dataset["location"].values
    
    def __len__(self):
        return len(self.feature_text)

    def __getitem__(self, item):
        inputs = self.config.tokenizer(self.patient_notes[item], self.feature_text[item], add_special_tokens=True, max_length=self.config.max_len, padding="max_length", return_offsets_mapping=False)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        encoded = self.config.tokenizer(self.patient_notes[item], add_special_tokens=True, max_length=self.config.max_len, padding="max_length", return_offsets_mapping=True)
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

# class Dataset(Dataset):
#     def __init__(self, testset, max_len, tokenizer):
#         self.testset = testset
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.patient_notess = testset["clean_text"].values # pn_history에서 space 제거한 문장들
#         self.feature_text = testset["feature_text"].values # feature_text에서 space 제거한 문장들
#         self.char_targets = testset["target"].values.tolilst()
    
#     def __len__(self):
#         return len(self.patient_notess)

#     def __getitem__(self, idx):
#         patient_notes = self.patient_notess[idx]
#         feature_text = self.feature_text[idx]
#         char_target = self.char_targets[idx]

#         encoding = encodings_from_precomputed(feature_text, patient_notes, self.tokenizer.precomputed, self.tokenizer, max_len=self.max_len)
        
#         return {
#             "ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
#             "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
#             "target": torch.tensor([0], dtype=torch.float),
#             "offsets": np.array(encoding["offset_mapping"]),
#             "text": patient_notes
#         }
