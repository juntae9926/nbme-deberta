from distutils import text_file
import re
import numpy as np
from transformers import AutoTokenizer
import ast
import itertools
from sklearn.metrics import f1_score
from tqdm import tqdm


def hard_processing(train):
    
    train.loc[338, 'annotation'] = ast.literal_eval('[["father heart attack"]]')
    train.loc[338, 'location'] = ast.literal_eval('[["764 783"]]')

    train.loc[621, 'annotation'] = ast.literal_eval('[["for the last 2-3 months"]]')
    train.loc[621, 'location'] = ast.literal_eval('[["77 100"]]')

    train.loc[655, 'annotation'] = ast.literal_eval('[["no heat intolerance"], ["no cold intolerance"]]')
    train.loc[655, 'location'] = ast.literal_eval('[["285 292;301 312"], ["285 287;296 312"]]')

    train.loc[1262, 'annotation'] = ast.literal_eval('[["mother thyroid problem"]]')
    train.loc[1262, 'location'] = ast.literal_eval('[["551 557;565 580"]]')

    train.loc[1265, 'annotation'] = ast.literal_eval('[[\'felt like he was going to "pass out"\']]')
    train.loc[1265, 'location'] = ast.literal_eval('[["131 135;181 212"]]')

    train.loc[1396, 'annotation'] = ast.literal_eval('[["stool , with no blood"]]')
    train.loc[1396, 'location'] = ast.literal_eval('[["259 280"]]')

    train.loc[1591, 'annotation'] = ast.literal_eval('[["diarrhoe non blooody"]]')
    train.loc[1591, 'location'] = ast.literal_eval('[["176 184;201 212"]]')

    train.loc[1615, 'annotation'] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
    train.loc[1615, 'location'] = ast.literal_eval('[["249 257;271 288"]]')

    train.loc[1664, 'annotation'] = ast.literal_eval('[["no vaginal discharge"]]')
    train.loc[1664, 'location'] = ast.literal_eval('[["822 824;907 924"]]')

    train.loc[1714, 'annotation'] = ast.literal_eval('[["started about 8-10 hours ago"]]')
    train.loc[1714, 'location'] = ast.literal_eval('[["101 129"]]')

    train.loc[1929, 'annotation'] = ast.literal_eval('[["no blood in the stool"]]')
    train.loc[1929, 'location'] = ast.literal_eval('[["531 539;549 561"]]')

    train.loc[2134, 'annotation'] = ast.literal_eval('[["last sexually active 9 months ago"]]')
    train.loc[2134, 'location'] = ast.literal_eval('[["540 560;581 593"]]')

    train.loc[2191, 'annotation'] = ast.literal_eval('[["right lower quadrant pain"]]')
    train.loc[2191, 'location'] = ast.literal_eval('[["32 57"]]')

    train.loc[2553, 'annotation'] = ast.literal_eval('[["diarrhoea no blood"]]')
    train.loc[2553, 'location'] = ast.literal_eval('[["308 317;376 384"]]')

    train.loc[3124, 'annotation'] = ast.literal_eval('[["sweating"]]')
    train.loc[3124, 'location'] = ast.literal_eval('[["549 557"]]')

    train.loc[3858, 'annotation'] = ast.literal_eval('[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]')
    train.loc[3858, 'location'] = ast.literal_eval('[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]')

    train.loc[4373, 'annotation'] = ast.literal_eval('[["for 2 months"]]')
    train.loc[4373, 'location'] = ast.literal_eval('[["33 45"]]')

    train.loc[4763, 'annotation'] = ast.literal_eval('[["35 year old"]]')
    train.loc[4763, 'location'] = ast.literal_eval('[["5 16"]]')

    train.loc[4782, 'annotation'] = ast.literal_eval('[["darker brown stools"]]')
    train.loc[4782, 'location'] = ast.literal_eval('[["175 194"]]')

    train.loc[4908, 'annotation'] = ast.literal_eval('[["uncle with peptic ulcer"]]')
    train.loc[4908, 'location'] = ast.literal_eval('[["700 723"]]')

    train.loc[6016, 'annotation'] = ast.literal_eval('[["difficulty falling asleep"]]')
    train.loc[6016, 'location'] = ast.literal_eval('[["225 250"]]')

    train.loc[6192, 'annotation'] = ast.literal_eval('[["helps to take care of aging mother and in-laws"]]')
    train.loc[6192, 'location'] = ast.literal_eval('[["197 218;236 260"]]')

    train.loc[6380, 'annotation'] = ast.literal_eval('[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]')
    train.loc[6380, 'location'] = ast.literal_eval('[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]')

    train.loc[6562, 'annotation'] = ast.literal_eval('[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]')
    train.loc[6562, 'location'] = ast.literal_eval('[["290 320;327 337"], ["290 320;342 358"]]')

    train.loc[6862, 'annotation'] = ast.literal_eval('[["stressor taking care of many sick family members"]]')
    train.loc[6862, 'location'] = ast.literal_eval('[["288 296;324 363"]]')

    train.loc[7022, 'annotation'] = ast.literal_eval('[["heart started racing and felt numbness for the 1st time in her finger tips"]]')
    train.loc[7022, 'location'] = ast.literal_eval('[["108 182"]]')

    train.loc[7422, 'annotation'] = ast.literal_eval('[["first started 5 yrs"]]')
    train.loc[7422, 'location'] = ast.literal_eval('[["102 121"]]')

    train.loc[8876, 'annotation'] = ast.literal_eval('[["No shortness of breath"]]')
    train.loc[8876, 'location'] = ast.literal_eval('[["481 483;533 552"]]')

    train.loc[9027, 'annotation'] = ast.literal_eval('[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]')
    train.loc[9027, 'location'] = ast.literal_eval('[["92 102"], ["123 164"]]')

    train.loc[9938, 'annotation'] = ast.literal_eval('[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]')
    train.loc[9938, 'location'] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

    train.loc[9973, 'annotation'] = ast.literal_eval('[["gaining 10-15 lbs"]]')
    train.loc[9973, 'location'] = ast.literal_eval('[["344 361"]]')

    train.loc[10513, 'annotation'] = ast.literal_eval('[["weight gain"], ["gain of 10-16lbs"]]')
    train.loc[10513, 'location'] = ast.literal_eval('[["600 611"], ["607 623"]]')

    train.loc[11551, 'annotation'] = ast.literal_eval('[["seeing her son knows are not real"]]')
    train.loc[11551, 'location'] = ast.literal_eval('[["386 400;443 461"]]')

    train.loc[11677, 'annotation'] = ast.literal_eval('[["saw him once in the kitchen after he died"]]')
    train.loc[11677, 'location'] = ast.literal_eval('[["160 201"]]')

    train.loc[12124, 'annotation'] = ast.literal_eval('[["tried Ambien but it didnt work"]]')
    train.loc[12124, 'location'] = ast.literal_eval('[["325 337;349 366"]]')

    train.loc[12279, 'annotation'] = ast.literal_eval('[["heard what she described as a party later than evening these things did not actually happen"]]')
    train.loc[12279, 'location'] = ast.literal_eval('[["405 459;488 524"]]')

    train.loc[12289, 'annotation'] = ast.literal_eval('[["experienced seeing her son at the kitchen table these things did not actually happen"]]')
    train.loc[12289, 'location'] = ast.literal_eval('[["353 400;488 524"]]')

    train.loc[13238, 'annotation'] = ast.literal_eval('[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
    train.loc[13238, 'location'] = ast.literal_eval('[["293 307"], ["321 331"]]')

    train.loc[13297, 'annotation'] = ast.literal_eval('[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]')
    train.loc[13297, 'location'] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

    train.loc[13299, 'annotation'] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
    train.loc[13299, 'location'] = ast.literal_eval('[["79 88"], ["409 418"]]')

    train.loc[13845, 'annotation'] = ast.literal_eval('[["headache global"], ["headache throughout her head"]]')
    train.loc[13845, 'location'] = ast.literal_eval('[["86 94;230 236"], ["86 94;237 256"]]')

    train.loc[14083, 'annotation'] = ast.literal_eval('[["headache generalized in her head"]]')
    train.loc[14083, 'location'] = ast.literal_eval('[["56 64;156 179"]]')

    return train


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
    text = re.sub(':', 'is', text)
    text = re.sub(';', 'and', text)
    text = re.sub('w/o', 'without', text)
    text = re.sub('w/', 'with', text)
    text = re.sub('y/o', 'years old', text)
    text = re.sub('yo ', ' years old ', text)
    text = re.sub(' M ', ' male ', text)
    text = re.sub(' F ', ' female ', text)
    text = re.sub('c/o', 'care of', text)
    text = re.sub('neg', 'negative', text)
    text = re.sub(' hx', ' history', text)
    text = re.sub('mo ', 'month ', text)
    text = re.sub(' mos ', 'months', text)
    text = re.sub('assoc ', 'associated ', text)

    text = re.sub(' SOB ', ' Shortness Of breath ', text)
    text = re.sub(' LMP ', ' Last Menstrual Period ', text)
    text = re.sub(' WNL ', 'Within Normal Limit', text)
    text = re.sub(' CP ', 'Chronic Pancreatitis', text)
    text = re.sub('RLQ', 'Right Lower Quadrant', text)
    text = re.sub(' MI ', ' Myocardial Infarction ', text)
    text = re.sub('abd ', 'abdominal ', text)

    return text


def clean_spaces(text):
    text = re.sub('\n', ' ', text) # delete space
    text = re.sub('\t', ' ', text) # delete intent
    text = re.sub('\r', ' ', text) # delete duplicate
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

def get_max_length(tokenizer, patient_notes, features):
    for i in ["pn_history"]:
        pn_history_lengths = []
        tk0 = tqdm(patient_notes[i].fillna("").values, total=len(patient_notes), desc="Max length")
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            pn_history_lengths.append(length)
    for j in ["feature_text"]:
        feature_lengths = []
        tk0 = tqdm(features[j].fillna("").values, total=len(features))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            feature_lengths.append(length)
    max_len = max(pn_history_lengths) + max(feature_lengths) + 3 # <cls>, <sep>, <sep>
    return max_len