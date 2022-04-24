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
from model import Network
from megatron import BioMegatron
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
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    patient_notes = pd.read_csv("data/patient_notes.csv")

    test = test.merge(features, how="left", on=["feature_num", "case_num"])
    test = test.merge(patient_notes, how="left", on=["pn_num", "case_num"])   
    test["pn_history"] = test["pn_history"].apply(lambda x: x.strip()) # 반복 제거
    test["feature_text"] = test["feature_text"].apply(process_feature_text)
    test["feature_text"] = test["feature_text"].apply(clean_spaces)
    test["clean_text"] = test["pn_history"].apply(clean_spaces)

    test_dataset = TestDataset(tokenizer, max_len, test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # Load best model
    model = Network(args.model) if not "biomegatron" in args.model else BioMegatron()
    
    if "deberta" in args.model:
        submission_name = "deberta-large"
        model_path = os.listdir(f"model/{submission_name}")
        model_path.sort(reverse = True)   
        model.load_state_dict(torch.load(f"model/{submission_name}/" + model_path[0])["model"])
        
    elif "biobert" in args.model:
        submission_name = "biobert"
        model_path = os.listdir(f"model/{submission_name}")
        model_path.sort(reverse = True)   
        model.load_state_dict(torch.load("model/biobert/" + model_path[0])["model"])
        
    elif "biomegatron" in args.model:
        submission_name = "biomegatron"
        model_path = os.listdir(f"model/{submission_name}")
        model_path.sort(reverse=True)
        model.load_state_dict(torch.load("model/biomegatron/" + model_path[0])["model"])
        
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
    if not os.path.isdir("submission"):
        os.mkdir("submission")
    submission[["id", "location"]].to_csv("submission/{}_{}.csv".format(submission_name, model_path[0]), index=False)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    train = pd.read_csv("data/train.csv") # [id, case_num, pn_num, feature_num, annotation, location]
    train["annotation"] = train["annotation"].apply(ast.literal_eval)
    train["annotation_length"] = train["annotation"].apply(len)
    train["location"] = train["location"].apply(ast.literal_eval)
    features = pd.read_csv("data/features.csv") # [feature_num, case_num, feature_text]
    patient_notes = pd.read_csv("data/patient_notes.csv") # [pn_num, case_num, pn_history]

    # Merge dataset
    train = train.merge(features, how="left", on=["feature_num", "case_num"])
    train = train.merge(patient_notes, how="left", on=["pn_num", "case_num"]) 

    train["pn_history"] = train["pn_history"].apply(lambda x: x.strip()) # 반복 제거
    #train["pn_history"] = train["pn_history"].apply(process_feature_text) ## 내가붙인거
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
    if "biomegatron" in args.model:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    #tokenizer.save_pretrained("biobert/tokenizer")
    #max_len = get_max_length(tokenizer, patient_notes, features)
    max_len = 466
    if args.train == True:
        best_score = 0

        fold = 0
        print(f" {fold}_fold -> Validation")

        if "biomegatron" in args.model:
            model = BioMegatron(tokenizer)
        else:
            model = Network(args.model)
        model.to(device)

        optimizer_parameters = get_optimizer_params(model, encoder_lr=args.lr, decoder_lr=args.lr, weight_decay=0.01)
        optimizer = AdamW(optimizer_parameters, lr=args.lr, eps=1e-6, betas=(0.9, 0.999))
        if args.scheduler == "expon":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(train)*(1-1/args.batch_size) / args.batch_size)*args.epochs, num_cycles=0.5)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=((len(train)/args.batch_size) * (1 - 1/args.batch_size) * args.epochs/2), eta_min=1e-7)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=((len(train)/args.batch_size) * (1 - 1/args.batch_size)), eta_min=1e-7)
        elif args.scheduler == "cycle":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr/2, max_lr=args.lr, step_size_up=50, step_size_down=None, mode='exp_range', gamma=args.gamma)
        elif args.scheduler == "lambda":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: args.gamma ** epoch)
        elif args.scheduler == "megatron":
            scheduler = AnnealingLR(optimizer, start_lr=args.lr, warmup_iter=10, total_iters=10, decay_style="cosine", last_iter=0, min_lr=1e-6)

        criterion = nn.BCEWithLogitsLoss(reduction="none")  

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
                
                try: 
                    curr_lr = scheduler.get_lr()[0]
                except:
                    curr_lr = scheduler.get_lr()

                train_losses.update(loss.item(), args.batch_size)
                if args.wandb == True:
                    wandb.log({"train loss": train_losses.avg})
                    wandb.log({"learning rate": curr_lr})
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

                batch_bar.set_postfix(loss="{:.06f}".format(train_losses.val), grad_norm="{:.04f}".format(grad_norm), lr="{:.07f}".format(curr_lr))

                if (idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                batch_bar.update()
            batch_bar.close()
        
            print("Epoch: [{0}/{1}] Loss: {loss.val:.6f}({loss.avg:.6f}) Grad: {grad_norm:.4f} LR: {lr:.7f}".format(   \
                    epoch+1, args.epochs, loss=train_losses, grad_norm=grad_norm, lr=curr_lr))

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
            print("EVAL: [{0}/{1}] Loss: {loss.val:.6f}({loss.avg:.6f})".format(epoch+1, args.epochs, loss=val_losses))
                
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
                if "deberta" in args.model:
                    if not os.path.isdir("model/deberta-large"):
                        os.mkdir("model/deberta-large")
                    torch.save({"model": model.state_dict(),
                                "predictions": predictions},
                                "model/deberta-large/{:0.2f}.pth".format(score))
                elif "biobert" in args.model:
                    if not os.path.isdir("model/biobert"):
                        os.mkdir("model/biobert")
                    torch.save({"model": model.state_dict(),
                                "predictions": predictions},
                                "model/biobert/{:0.2f}.pth".format(score))
                elif "biomegatron" in args.model:
                    if not os.path.isdir("model/biomegatron"):
                        os.mkdir("model/biomegatron")
                    torch.save({"model": model.state_dict(),
                                "predictions": predictions},
                                "model/biomegatron/{:0.2f}.pth".format(score))
                print("----- best model saved -----")

    if args.test == True:  
        inference(tokenizer, max_len, args.device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--model", default="microsoft/deberta-base", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--scheduler", default="cosine", type=str, help="[expon, cosine, lambda, cycle, megatron]")
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--n-fold", default=10, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--gamma", default=0.999, type=float)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.wandb == True:    
        project_name = args.model.split("/")[-1]
        wandb.init(project=project_name, entity="juntae9926")
        wandb.config.update(args)
        print(f"Start with wandb with {project_name}")

    main(args)