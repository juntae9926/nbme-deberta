import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast
import argparse

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
import torch.nn as nn

from sklearn.model_selection import GroupKFold

from data_utils import *
from model import *
from dataset import *
from train_utils import *
import wandb
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def inference(tokenizer, max_len, device):
    # Load test dataset
    test = pd.read_csv("data/test.csv")
    submission = pd.read_csv("data/sample_submission.csv")
    features = pd.read_csv("data/features.csv")
    patient_notes = pd.read_csv("data/patient_notes.csv")

    test = test.merge(features, how="left", on=["feature_num", "case_num"])
    test = test.merge(patient_notes, how="left", on=["pn_num", "case_num"])    

    test_dataset = TestDataset(tokenizer, max_len, test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # Load best model
    model_path = os.listdir("model")
    model_path = model_path.sort(reverse = True)
    model = Network(args.model)
    model.load_state_dict(torch.load("model/" + model_path)["model"])

    preds = []
    predictions = []
    model.eval()
    model.to(device)
    batch_bar = tqdm(test_loader, total=len(test_loader))
    for inputs in batch_bar:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    prediction = np.concatenate(preds)
    prediction = prediction.reshape((len(test), max_len))
    char_probs = get_char_probs(test["pn_history"].values, prediction, tokenizer)
    predictions.append(char_probs)
    predictions = np.mean(predictions, axis=0)

    # Submission
    results = get_results(predictions)
    submission["location"] = results
    submission[["id", "location"]].to_csv("submission_{}.csv".format(model_path), index=False)


def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
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

    train = hard_processing(train)
    train.to_csv("dataset.csv")

    # Split dataset
    Fold = GroupKFold(n_splits=args.n_fold)
    groups = train["pn_num"].values
    for n, (train_idx, val_idx) in enumerate(Fold.split(train, train["location"], groups)):
        train.loc[val_idx, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained("tokenizer")
    #max_len = get_max_length(tokenizer, patient_notes, features)
    max_len = 466
    if args.train == True:
        best_score = 0

        model = Network(args.model)
        model.to(device)

        optimizer_parameters = get_optimizer_params(model, encoder_lr=args.lr, decoder_lr=args.lr, weight_decay=0.01)
        optimizer = AdamW(optimizer_parameters, lr=args.lr, eps=1e-6, betas=(0.9, 0.999))
        #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=args.lr,  T_up=150, gamma=args.gamma)
        if args.scheduler == "expon":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif args.scheduler == "cosin":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(715 * args.epochs), eta_min=1e-7)
        elif args.scheduler == "cycle":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr/2, max_lr=args.lr, step_size_up=50, step_size_down=None, mode='exp_range', gamma=0.995)
        elif args.scheduler == "lambda":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        criterion = nn.BCEWithLogitsLoss(reduction="none")  

        for fold in range(args.n_fold):
            print(f" {fold}_fold -> Validation")

            train_set = train[train["fold"] != fold].reset_index(drop=True)
            val_set = train[train["fold"] == fold].reset_index(drop=True)

            val_pn_history = val_set["pn_history"].values
            val_labels = create_labels_for_scoring(val_set)

            train_dataset = TrainDataset(tokenizer, max_len, train_set)
            val_dataset = TrainDataset(tokenizer, max_len, val_set)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)            

            for epoch in range(args.epochs):
                
                # Train
                model.train()
                batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")
                scaler = torch.cuda.amp.GradScaler(enabled=True)
                train_losses = AverageMeter()
                for idx, (inputs, labels) in enumerate(train_loader):
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)
                    labels = labels.to(device)
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        y_preds = model(inputs)
                    
                    loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
                    loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss/args.gradient_accumulation_steps

                    train_losses.update(loss.item(), args.batch_size)
                    if args.wandb == True:
                        wandb.log({"train loss": train_losses.avg})
                        wandb.log({"learning rate": scheduler.get_lr()[0]})
                    scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

                    batch_bar.set_postfix(loss="{:.06f}".format(train_losses.val), grad_norm="{:.04f}".format(grad_norm), lr="{:.07f}".format(scheduler.get_lr()[0]))

                    if (idx + 1) % args.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                    batch_bar.update()
                batch_bar.close()
            
                print("Epoch: [{0}][{1}/{2}] Loss: {loss.val:.6f}({loss.avg:.6f}) Grad: {grad_norm:.4f} LR: {lr:.7f}".format(   \
                        fold, epoch+1, args.epochs, loss=train_losses, grad_norm=grad_norm, lr=scheduler.get_lr()[0]))

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

                    if args.gradient_accumulation_steps > 1:
                        loss = loss/args.gradient_accumulation_steps

                    val_losses.update(loss.item(), args.batch_size)
                    if args.wandb == True:
                        wandb.log({"val loss": val_losses.avg})
                    preds.append(y_preds.sigmoid().to("cpu").numpy())

                    batch_bar.set_postfix(loss="{:.06f}".format(val_losses.val))
                    batch_bar.update()
                batch_bar.close()
                print("EVAL: [{0}][{1}/{2}] Loss: {loss.val:.6f}({loss.avg:.6f})".format(fold, epoch+1, args.epochs, loss=val_losses))
                    
                predictions = np.concatenate(preds).reshape((len(val_set), max_len))

                # Scoring
                char_probs = get_char_probs(val_pn_history, predictions, tokenizer)
                results = get_results(char_probs, th=0.5)
                preds = get_predictions(results)
                score = get_score(val_labels, preds)
                if args.wandb == True:
                    wandb.log({"micro f1 score": score})
                print(f"micro f1 score is {score}")

                if best_score < score:
                    best_score = score
                    if not os.path.isdir("model"):
                        os.mkdir("model")
                    torch.save({"model": model.state_dict(),
                                "predictions": predictions},
                                "model/deberta_{:0.2f}.pth".format(score))
                    print("----- best model saved -----")

    if args.test == True:  
        inference(tokenizer, max_len, args.device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--model", default="microsoft/deberta-base", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--scheduler", default="cosin", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--n-fold", default=5, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--test", default=True, type=bool)
    parser.add_argument("--wandb", default=True, type=bool)
    args = parser.parse_args()

    if args.wandb == True:    
        wandb.init(project="nbme", entity="juntae9926")
        wandb.config.update(args)
        print("Start with wandb")

    main()