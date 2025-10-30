import re
import os
import pandas as pd
from pathlib import Path
import yaml
import pickle
from PIL import Image
from terminaltables import AsciiTable
from loguru import logger as eval_logger
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


# with open(Path(__file__).parent / "default_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         if "!function" not in line:
#             safe_data.append(line)
# media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]
# embodiedscan_path = yaml.safe_load("".join(safe_data))["metadata"]["embodiedscan_path"]
# with open(embodiedscan_path, "rb") as f:
#     data = pickle.load(f)["data_list"]
#     id2scene = {sample["sample_id"]: sample for sample in data}

def scanqa_doc_to_visual(doc):
    if "image" in doc:
        doc["images"] = doc["image"]
    if "images" in doc: 
        image_files = doc["images"]
        images = [
            Image.open(
                os.path.join(media_dir, image_file)
            ).convert("RGB")
            for image_file in image_files
        ]
        return [images]
    if "video" in doc:
        return [os.path.join(media_dir, doc["video"])]


def scanqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["conversations"][0]["value"].replace("<image>", "").replace("<video>", "")
    return question

def scanqa_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    target = doc["gt_answers"]
    return target

def get_between(ss, st, ed):
    return ss.split("st")[-1].split(ed)[0]

def scanqa_process_results(doc, results):
    doc["pred_response"] = results[0]
    if "<answer>" in doc["pred_response"]:
        doc["pred_response"] = get_between(
            doc["pred_response"],
            "<answer>",
            "</answer>"
        )
    # doc["gt_response"] = doc["annotations"]
    # bangya's version
    doc["gt_response"] = {
        "answers": scanqa_doc_to_target(doc)
    }
    return {"scanqa_score": doc}


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data

# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/evaluator/scanqa_eval.py#L41-L50
def answer_match(pred, gts):
    # return EM and refined EM
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0

def scanqa_aggregate_results(results):

    cider = Cider()
    bleu = Bleu(4)
    meteor = Meteor()
    rouge = Rouge()
    spice = Spice()
    tokenizer = PTBTokenizer()

    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    acc, refined_acc = 0, 0
    for i, item in enumerate(results):
        pred_answer = item['pred_response']
        gt_answers = item['gt_response']['answers']
        pred_answer = clean_answer(pred_answer)
        ref_captions = [clean_answer(gt_answer) for gt_answer in gt_answers]
        tmp_acc, tmp_refined_acc = answer_match(pred_answer, ref_captions)
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[i] = [{'caption': pred_answer}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[i] = [{'caption': caption} for caption in ref_captions]

        # res[i] = ['sos ' + item['pred_response'].replace('.', ' . ').replace(',', ' , ').lower() + ' eos' ]
        # gts[i] = ['sos ' + it.replace('.', ' . ').replace(',', ' , ').lower() + ' eos' for it in item['gt_response']]
    
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    acc = acc / len(results)
    refined_acc = refined_acc / len(results)
    
    cider_score = cider.compute_score(tmp_targets, tmp_preds)
    bleu_score = bleu.compute_score(tmp_targets, tmp_preds)
    meteor_score = meteor.compute_score(tmp_targets, tmp_preds)
    rouge_score = rouge.compute_score(tmp_targets, tmp_preds)
    spice_score = spice.compute_score(tmp_targets, tmp_preds)

    # table_data = [
    #     ["Metric", "Score"],
    #     ["EM1", f"{acc*100:.2f}"],
    #     ["EM1_refined", f"{refined_acc*100:.2f}"],
    #     ["CIDER", f"{cider_score[0]*100:.2f}"],
    #     ["BLEU-1", f"{bleu_score[0][0]*100:.2f}"],
    #     ["BLEU-2", f"{bleu_score[0][1]*100:.2f}"],
    #     ["BLEU-3", f"{bleu_score[0][2]*100:.2f}"],
    #     ["BLEU-4", f"{bleu_score[0][3]*100:.2f}"],
    #     ["METEOR", f"{meteor_score[0]*100:.2f}"],
    #     ["ROUGE", f"{rouge_score[0]*100:.2f}"],
    #     ["SPICE", f"{spice_score[0]*100:.2f}"],
    #     ["Data Num", f"{len(results)}"]
    # ]

    # table = AsciiTable(table_data)
    # table.title = "Evaluation Metrics"
    # eval_logger.info("\n" + table.table)
    # return cider_score[0]*100

    metrics = {
        "EM1": acc * 100,
        "EM1_refined": refined_acc * 100,
        "CIDER": cider_score[0] * 100,
        "BLEU-1": bleu_score[0][0] * 100,
        "BLEU-2": bleu_score[0][1] * 100,
        "BLEU-3": bleu_score[0][2] * 100,
        "BLEU-4": bleu_score[0][3] * 100,
        "METEOR": meteor_score[0] * 100,
        "ROUGE": rouge_score[0] * 100,
        "Data Num": len(results),
    }

    table_data = [["Metric", "Score"]] + [[k, f"{v:.2f}" if isinstance(v, (int,float)) else str(v)] for k, v in metrics.items()]
    table = AsciiTable(table_data); table.title = "Evaluation Metrics"
    eval_logger.info("\n" + table.table)

    return metrics