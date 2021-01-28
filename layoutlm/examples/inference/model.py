import os
import json
import re
from pathlib import Path

import cv2
import torch
import numpy as np
from torch import nn
from imutils import paths
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    WEIGHTS_NAME,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    BertModel,
    BertForTokenClassification,
    LayoutLMTokenizer,
    LayoutLMForTokenClassification,
    LayoutLMModel,
    LayoutLMConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from seqeval.scheme import Entities, IOBES, IOB2, IOE2

from .datasets import get_examples_from_one_sample, get_labels
from .postprocess import convert_kv_format


class LayoutLM:
    def __init__(self, weights_path=None, bert_model=None, max_seq_length=512, device='cuda', labels_path=None):
        if any([item is None for item in (weights_path, bert_model, labels_path)]):
            raise IOError('Model load error: weight {}   bert {}  labels {}'.format(weights_path, bert_model, labels_path))

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.model = LayoutLMForTokenClassification.from_pretrained(weights_path,
                                                                    return_dict=True)
        self.model.to(device)
        self.device = device
        self.model.eval()

        self.labels = get_labels(labels_path)
        self.max_seq_length = max_seq_length

    def process(self, ocr_data, image_path, key_list):
        image = Image.open(image_path)
        image = image.convert('RGB')

        label_map = {i: label for i, label in enumerate(self.labels)}

        features = get_examples_from_one_sample(image, ocr_data, self.tokenizer, self.max_seq_length, datapile_format=False)

        predictions = []
        boxes = []
        input_ids = []
        
        inputs_all = {
            "input_ids": [],
            "bboxes": []
        }

        for f in features:
            with torch.no_grad():
                inputs = {
                    "input_ids": torch.tensor([f.input_ids], dtype=torch.long).to(self.device),
                    "attention_mask": torch.tensor([f.attention_mask], dtype=torch.long).to(self.device),
                    "token_type_ids": torch.tensor([f.segment_ids], dtype=torch.long).to(self.device),
                    "bbox": torch.tensor([f.boxes], dtype=torch.long).to(self.device)
                }

                inputs_all["input_ids"].extend(f.input_ids)
                inputs_all["bboxes"].extend(f.boxes)

                outputs = self.model(**inputs)

                logits = outputs.logits
                pred = logits.detach().cpu().numpy()
                pred = np.argmax(pred, axis=2)[0]

                assert len(pred) == len(f.label_ids)
                assert len(pred) == len(f.actual_bboxes)

                try:
                    for i, label_id in enumerate(f.label_ids):
                        if label_id >= 0:
                            predictions.append(label_map[pred[i]])
                            boxes.append(f.actual_bboxes[i])
                            input_ids.append((f.input_ids[i]))
                except:
                    raise IOError('Model prediction error')

        schemes = [(IOBES, None), (IOB2, hide_eng_tag), (IOE2, hide_begin_tag)]
        entities = []

        inputs = input_ids
    
        for scheme, preprocess_function in schemes:
            if preprocess_function is not None:
                tmp_predictions = preprocess_function(predictions)
            else:
                tmp_predictions = predictions
    
            mask = [False] * len(predictions)
            sequences = Entities([tmp_predictions], scheme=scheme)
    
            for entity in sequences.entities[0]:
                entity_text = tokenizer.decode(inputs[entity.start: entity.end])
                for _id in range(entity.start, entity.end):
                    mask[_id] = True
    
                list_x, list_y = [], []
                for box in boxes[entity.start: entity.end]:
                    x1, y1, x2, y2 = box
                    list_x.extend((x1, x2))
                    list_y.extend((y1, y2))
    
                if scheme == IOB2:
                    tag = predictions[entity.start][2:]
                elif scheme == IOE2:
                    tag = predictions[entity.end-1][2:]
                else:
                    tag = entity.tag
    
                min_x, min_y, max_x, max_y = min(list_x), min(list_y), max(list_x), max(list_y)
                entity_box = [min_x + 8, min_y + 8, max_x - 8, max_y - 8]
    
                if entity.end - entity.start > 1 or scheme == IOBES:
                    entities.append({
                        'text': entity_text,
                        'start': entity.start,
                        'end': entity.end,
                        'tag': tag,
                        'box': entity_box
                    })
    
            new_predictions = []
            new_boxes = []
            new_inputs= []
            for _id, status in enumerate(mask):
                if not status:
                    new_predictions.append(predictions[_id])
                    new_boxes.append(boxes[_id])
                    new_inputs.append(inputs[_id])
                else:
                    new_predictions.append('O')
                    new_boxes.append([0, 0, 0, 0])
                    new_inputs.append('')
            predictions = new_predictions
            boxes = new_boxes
            inputs = inputs

        return convert_kv_format(entities, key_list=key_list)