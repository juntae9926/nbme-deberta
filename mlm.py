import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AdamW

from tqdm import tqdm
import os
import wandb

class CFG:
    seed = 42
    model_name = "microsoft/deberta-v3-large"
    epochs = 3
    batch_size = 32
    lr = 1e-6
    weight_decay = 1e-6
    max_len = 512
    mask_prob = 0.15
    n_accumulate = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLMDataset:
    def __init__(self, data, tokenizer, special_tokens):
        self.data = data
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]

        tokenized_data = self.tokenizer.encode_plus(text, max_length = CFG.max_len, truncation=True,
                                                    padding='max_length', add_special_tokens=True,
                                                    return_tensors='pt')
        
        input_ids = torch.flatten(tokenized_data.input_ids)
        attention_mask = torch.flatten(tokenized_data.attention_mask)
        labels = get_masked_labels(input_ids, self.special_tokens)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
def get_masked_labels(input_ids, special_tokens):
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < CFG.mask_prob)

    for special_token in special_tokens:
        token = special_token.item()
        mask_arr *= (input_ids != token)
    selection = torch.flatten(mask_arr[0].nonzero()).tolist()
    input_ids[selection] = 128000 # tokenizer에 따라 [MASK] token을 다르게 설정

    return input_ids

def set_seed(seed=CFG.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONASHSEED'] = str(seed)

def train_loop(model, dataloader, optimizer, epoch, device):
    model.train()
    batch_losses = []
    loop = tqdm(dataloader, leave=True)
    for batch_num, batch in enumerate(loop):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        batch_loss = loss / CFG.n_accumulate
        batch_losses.append(batch_loss.item())

        loop.set_description(f"EPOCH {epoch + 1}")
        loop.set_postfix(loss=batch_loss.item())
        batch_loss.backward()

        if batch_num % CFG.n_accumulate == 0 or batch_num == len(dataloader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            model.zero_grad()
    
    return np.mean(batch_losses)

def main():

    device = CFG.device

    df = pd.read_csv("/root/pretrain_deberta/data/patient_notes.csv")
    pn_history = df["pn_history"].unique() # unique 함수로 array로 변환

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelWithLMHead.from_pretrained(CFG.model_name)

    special_tokens = tokenizer.encode_plus('[CLS] [SEP] [MASK] [PAD]', add_special_tokens = False, return_tensors='pt')
    special_tokens = torch.flatten(special_tokens["input_ids"])

    dataset = MLMDataset(pn_history, tokenizer, special_tokens)
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    history = []
    best_loss = np.inf
    prev_loss = np.inf
    model.gradient_checkpointing_enable()
    model.to(device)
    print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

    for epoch in range(CFG.epochs):
        loss = train_loop(model, dataloader, optimizer, epoch, device)
        history.append(loss)
        print(f"Loss: {loss}")

        if loss < best_loss:
            print("New Best Loss {:.4f} -> {:.4f}, Saving Model".format(prev_loss, loss))
            model.save_pretrained("./")

            best_loss = loss
        prev_loss = loss

if __name__ == "__main__":
    # project_name = CFG.model_name.split("/")[-1]
    # wandb.init(project=project_name, entity="juntae9926")
    # wandb.config.update(args)
    # print(f"Start with wandb with {project_name}")
    
    main()

