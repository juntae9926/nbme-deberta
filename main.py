import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast

import torch
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
import torch.nn as nn

from sklearn.model_selection import GroupKFold

from data_utils import *
from model import *
from dataset import *
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Config:
    apex=True
    print_freq=100
    num_workers=4
    #model = "roberta-large"
    model="microsoft/deberta-base"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=8
    fc_dropout=0.2
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=5
    train_fold=[0, 1, 2, 3, 4]
    train=True
    val=True


def main():
    
    # Load dataset
    train = pd.read_csv("data/train.csv") # [id, case_num, pn_num, feature_num, annotation, location]
    train["annotation"] = train["annotation"].apply(ast.literal_eval)
    train["annotation_length"] = train["annotation"].apply(len)
    train["location"] = train["location"].apply(ast.literal_eval)
    features = features = pd.read_csv("data/features.csv") # [feature_num, case_num, feature_text]
    patient_notes = pd.read_csv("data/patient_notes.csv") # [pn_num, case_num, pn_history]

    # Merge dataset
    train = train.merge(features, how="left", on=["feature_num", "case_num"])
    train = train.merge(patient_notes, how="left", on=["pn_num", "case_num"]) 
    # [id, case_num, pn_num, feature_num, annotation, location, feature_text, pn_history]

    train["pn_history"] = train["pn_history"].apply(lambda x: x.strip()) # 반복 제거
    train["feature_text"] = train["feature_text"].apply(process_feature_text)
    train["feature_text"] = train["feature_text"].apply(clean_spaces)
    train["clean_text"] = train["pn_history"].apply(clean_spaces)
    train["target"] = ""

    # Split dataset
    Fold = GroupKFold(n_splits=Config.n_fold)
    groups = train["pn_num"].values
    for n, (train_idx, val_idx) in enumerate(Fold.split(train, train["location"], groups)):
        train.loc[val_idx, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model)
    tokenizer.save_pretrained("tokenizer")
    Config.tokenizer = tokenizer

    ## Define max length
    # for i in ["pn_history"]:
    #     pn_history_lengths = []
    #     tk0 = tqdm(patient_notes[i].fillna("").values, total=len(patient_notes))
    #     for text in tk0:
    #         length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    #         pn_history_lengths.append(length)
    # for j in ["feature_text"]:
    #     feature_lengths = []
    #     tk0 = tqdm(features[j].fillna("").values, total=len(features))
    #     for text in tk0:
    #         length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    #         feature_lengths.append(length)
    # Config.max_len = max(pn_history_lengths) + max(feature_lengths) + 3 # <cls>, <sep>, <sep>
    Config.max_len = 466
    print("Tokenizer max length: ", Config.max_len)

    if Config.train == True:

        for fold in range(Config.n_fold):
            if fold in Config.train_fold:

                train_set = train[train["fold"] != fold].reset_index(drop=True)
                val_set = train[train["fold"] == fold].reset_index(drop=True)

                val_pn_history = val_set["pn_history"].values
                val_labels = create_labels_for_scoring(val_set)

                train_dataset = TrainDataset(Config, train_set)
                val_dataset = TrainDataset(Config, val_set)

                train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=1, pin_memory=True)

                model = Network(Config, config_file=None, pretrained=True)
                torch.save(model.config, "model_config.pth")
                model.to(device)

                def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
                    param_optimizer = list(model.named_parameters())
                    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                    optimizer_parameters = [
                        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        'lr': encoder_lr, 'weight_decay': weight_decay},
                        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                        'lr': encoder_lr, 'weight_decay': 0.0},
                        {'params': [p for n, p in model.named_parameters() if "model" not in n],
                        'lr': decoder_lr, 'weight_decay': 0.0}
                    ]
                    return optimizer_parameters
                
                optimizer_parameters = get_optimizer_params(model, encoder_lr=Config.encoder_lr, decoder_lr=Config.decoder_lr, weight_decay=Config.weight_decay)
                optimizer = AdamW(optimizer_parameters, lr=Config.encoder_lr, eps=Config.eps, betas=Config.betas)

                def get_scheduler(cfg, optimizer, num_train_steps):
                    if cfg.scheduler=='linear':
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
                        )
                    elif cfg.scheduler=='cosine':
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
                        )
                    return scheduler
                
                num_train_steps = int(len(train_set) / Config.batch_size * Config.epochs)
                scheduler = get_scheduler(Config, optimizer, num_train_steps)

                criterion = nn.BCEWithLogitsLoss(reduction="none")

                best_score = 0

                for epoch in range(Config.epochs):
                    
                    # Train
                    model.train()
                    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")
                    scaler = torch.cuda.amp.GradScaler(enabled=Config.apex)
                    train_losses = AverageMeter()
                    global_step = 0
                    for idx, (inputs, labels) in enumerate(train_loader):
                        for k, v in inputs.items():
                            inputs[k] = v.to(device)
                        labels = labels.to(device)
                        
                        with torch.cuda.amp.autocast(enabled=Config.apex):
                            y_preds = model(inputs)

                        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
                        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()

                        if Config.gradient_accumulation_steps > 1:
                            loss = loss/Config.gradient_accumulation_steps

                        train_losses.update(loss.item(), Config.batch_size)
                        scaler.scale(loss).backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

                        batch_bar.set_postfix(loss="{:.04f}".format(train_losses.avg), grad_norm="{:.04f}".format(grad_norm), lr="{:.08f}".format(scheduler.get_lr()[0]))

                        if (idx + 1) % Config.gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            global_step += 1
                            if Config.batch_scheduler:
                                scheduler.step()
                        batch_bar.update()
                    batch_bar.close()

                    # if idx % Config.print_freq == 0 or idx == (len(train_loader) - 1):
                    print('Epoch: [{0}][{1}/{2}] ' 'Loss: {loss.val:.4f}({loss.avg:.4f}) ' 'Grad: {grad_norm:.4f}  ' 'LR: {lr:.8f}  '  \
                        .format(epoch+1, idx, len(train_loader), loss=train_losses, grad_norm=grad_norm, lr=scheduler.get_lr()[0]))

                    # Validation
                    model.eval()
                    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc="Val")
                    val_losses = AverageMeter()
                    preds = []
                    for idx, (inputs, labels) in enumerate(val_loader):
                        for k, v in inputs.items():
                            inputs[k] = v.to(device)
                        labels = labels.to(device)

                        with torch.no_grad():
                            y_preds = model(inputs)

                        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
                        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()

                        if Config.gradient_accumulation_steps > 1:
                            loss = loss/Config.gradient_accumulation_steps

                        val_losses.update(loss.item(), Config.batch_size)
                        preds.append(y_preds.sigmoid().to("cpu").numpy())

                        batch_bar.set_postfix(loss="{:.04f}".format(val_losses.avg))
                        batch_bar.update()
                    batch_bar.close()
                        # if idx % Config.print_freq == 0 or idx == (len(val_loader) - 1):
                    print('EVAL: [{0}/{1}] ' 'Loss: {loss.val:.4f}({loss.avg:.4f}) '.format(idx, len(val_loader), loss=val_losses))
                        
                    predictions = np.concatenate(preds).reshape((len(val_set), Config.max_len))

                    # Scoring
                    char_probs = get_char_probs(val_pn_history, predictions, Config.tokenizer)
                    results = get_results(char_probs, th=0.5)
                    preds = get_predictions(results)
                    score = get_score(val_labels, preds)

                    if best_score < score:
                        best_score = score
                        torch.save({"model": model.state_dict(),
                                    "predictions": predictions},
                                    "model/{}_fold{}_best.pth".format(Config.model.replace("/", "-"), fold))
                        print("----- best model saved -----")

if __name__ == "__main__":
    main()