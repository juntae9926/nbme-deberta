import re
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
import ast
import itertools
from sklearn.metrics import f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)

def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score

def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths



def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text, 
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions

def process_feature_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text

def clean_spaces(text):
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\r', ' ', text)
    return text

def token_pred_to_char_pred(token_pred, offsets):
    char_pred = np.zeros((np.max(offsets), token_pred.shape[1]))
    for i in range(len(token_pred)):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        char_pred[s:e] = token_pred[i]
        if token_pred.shape[1] == 3:
            s += 1
            char_pred[s: e, 1], char_pred[s: e, 2] = (np.max(char_pred[s: e, 1:], 1), np.min(char_pred[s: e, 1:], 1),)
    return char_pred

def labels_to_sub(labels):
    all_spans = []
    for label in labels:
        indices = np.where(label > 0)[0]
        indices_grouped = [list(g) for _, g in itertools.groupby(indices, key=lambda n, c=itertools.count(): n - next(c))]
        spans = [f"{min(r)} {max(r) + 1}" for r in indices_grouped]
        all_spans.append(";".join(spans))
    return all_spans

def char_target_to_span(char_target):
    spans = []
    start, end = 0, 0
    for i in range(len(char_target)):
        if char_target[i] == 1 and char_target[i - 1] == 0:
            if end:
                spans.append([start, end])
            start = i
            end = i + 1
        elif char_target[i] == 1:
            end = i + 1
        else:
            if end:
                spans.append([start, end])
            start, end = 0, 0
    return spans

def post_process_spaces(target, text):
    target = np.copy(target)

    if len(text) > len(target):
        padding = np.zeros(len(text) - len(target))
        target = np.concatenate([target, padding])
    else:
        target = target[:len(text)]

    if text[0] == " ":
        target[0] = 0
    if text[-1] == " ":
        target[-1] = 0

    for i in range(1, len(text) - 1):
        if text[i] == " ":
            if target[i] and not target[i - 1]:
                target[i] = 0

            if target[i] and not target[i + 1]:
                target[i] = 0

            if target[i - 1] and target[i + 1]:
                target[i] = 1
    return target

## DATA TOKENIZATION

def get_tokenizer(name, precompute=False, df=None, folder=None):
    if folder is None:
        tokenizer = AutoTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(folder)

    tokenizer.name = name

    tokenizer.special_tokens = {
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
        "pad": tokenizer.pad_token_id,
    }

    if precompute:
        tokenizer.precomputed = precompute_tokens(df, tokenizer)
    else:
        tokenizer.precomputed=None
        
    return tokenizer

def precompute_tokens(df, tokenizer):
    feature_texts = df["feature_text"].unique()
    ids = {}
    offsets = {}

    for feature_text in feature_texts:
        encoding = tokenizer(
            feature_text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        ids[feature_text] = encoding["input_ids"]
        offsets[feature_text] = encoding["offset_mapping"]

    texts = df["clean_text"].unique()

    for text in texts:
        encoding = tokenizer(
            text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        ids[text] = encoding["input_ids"]
        offsets[text] = encoding["offset_mapping"]
        
    return {"ids": ids, "offsets": offsets}

def encodings_from_precomputed(feature_text, text, precomputed, tokenizer, max_len=300):
    tokens = tokenizer.special_tokens

    if "roberta" in tokenizer.name:
        qa_sep = [tokens["sep"], tokens["sep"]]
    else:
        qa_sep = [tokens["sep"]]

    input_ids = [tokens["cls"]] + precomputed["ids"][feature_text] + qa_sep
    n_question_tokens = len(input_ids)

    input_ids += precomputed["ids"][text]
    input_ids = input_ids[: max_len - 1] + [tokens["sep"]]

    if "roberta" not in tokenizer.name:
        token_type_ids = np.ones(len(input_ids))
        token_type_ids[:n_question_tokens] = 0
        token_type_ids = token_type_ids.tolist()
    else:
        token_type_ids = [0] * len(input_ids)

    offsets = [(0, 0)] * n_question_tokens + precomputed["offsets"][text]
    offsets = offsets[: max_len - 1] + [(0, 0)]

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)

    encoding = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "offset_mapping": offsets,
    }

    return encoding