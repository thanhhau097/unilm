import os
import math
import random
import logging
from copy import deepcopy
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            attention_mask,
            segment_ids,
            label_ids,
            question,
            boxes,
            actual_bboxes,
            file_name,
            page_size,
    ):
        assert (
                0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size
        self.question = question


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f, open(
            box_file_path, encoding="utf-8"
    ) as fb, open(image_file_path, encoding="utf-8") as fi:
        words = []
        boxes = []
        actual_bboxes = []
        file_name = None
        page_size = None
        labels = []
        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size,
                        )
                    )
                    guid_index += 1
                    words = []
                    boxes = []
                    actual_bboxes = []
                    file_name = None
                    page_size = None
                    labels = []
            else:
                splits = line.split("\t")
                bsplits = bline.split("\t")
                isplits = iline.split("\t")
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
                box = bsplits[-1].replace("\n", "")
                box = [int(b) for b in box.split()]
                boxes.append(box)
                actual_bbox = [int(float(b)) for b in isplits[1].split()]
                actual_bboxes.append(actual_bbox)
                page_size = [int(float(i)) for i in isplits[2].split()]
                file_name = isplits[3].strip()
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )
    return examples


def sort_funsd_reading_order(lines):
    """ Sort cell list to create the right reading order using their locations
    Parameters
    ----------
    lines: list of cells to sort
    Returns
    -------
    a list of cell lists in the right reading order that contain no key or start with a key and contain no other key
    """
    sorted_list = []

    if len(lines) == 0:
        return lines

    while len(lines) > 1:
        topleft_line = lines[0]
        for line in lines[1:]:
            topleft_line_pos = topleft_line['box']
            topleft_line_center_y = (topleft_line_pos[1] +
                                     topleft_line_pos[3]) / 2
            x1, y1, x2, y2 = line['box']
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            cell_h = y2 - y1
            if box_center_y <= topleft_line_center_y - cell_h / 2:
                topleft_line = line
                continue
            if box_center_x < topleft_line_pos[2] and box_center_y < topleft_line_pos[3]:
                topleft_line = line
                continue
        sorted_list.append(topleft_line)
        lines.remove(topleft_line)

    sorted_list.append(lines[0])

    return sorted_list


def convert_one_datapile_to_funsd(data, image, tokenizer, datapile_format=True):
    width, height = image.size

    lines = []
    if datapile_format:
        regions = data['attributes']['_via_img_metadata']['regions']
    else:
        regions = data

    for line in regions:
        current_line = {}

        if datapile_format:
            text = line['region_attributes']['label'].strip()
            if not text:
                continue

            if line['shape_attributes']['name'] == 'rect':
                x1 = line['shape_attributes']['x']
                y1 = line['shape_attributes']['y']
                line_width = line['shape_attributes']['width']
                line_height = line['shape_attributes']['height']
            elif line['shape_attributes']['name'] == 'polygon':
                x1 = min(line['shape_attributes']['all_points_x'])
                y1 = min(line['shape_attributes']['all_points_y'])
                line_width = max(line['shape_attributes']['all_points_x']) - x1
                line_height = max(line['shape_attributes']['all_points_y']) - y1
        else:
            text = line.get('text', '').strip()
            if not text:
                continue
            x1, y1 = line['location'][0]
            line_width, line_height = (line['location'][1][0] - line['location'][0][0]), (line['location'][2][1] - line['location'][0][1])
            points = line['location']
            # print(text, x1, y1, line_width, line_height)

        if line_width < len(text):
            continue


        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x1 + line_width, width)
        y2 = min(y1 + line_height, height)

        # if x2 < x1 or y2 < y1:
        #     from IPython import embed
        #     embed()

        current_line['text'] = text.replace('¥', '円')
        current_line['box'] = [x1, y1, x2, y2]
        current_line['label'] = process_label_invoice_full_class(line['region_attributes']) if datapile_format else 'other'
        current_line['points'] = points

        words = []
        start_x = x1
        tokens = [tokenizer.unk_token] if text == 'NotValid' else tokenizer.tokenize(text)
        token_width = int(line_width / len(tokens))
        width_ratio = 1 / len(tokens)

        for _id, w in enumerate(tokens):
            current_word = {}

            # if len(text) == 1:
            #     token_width = round(1 / len(tokens) * line_width)
            # elif w == tokenizer.unk_token:
            #     token_width = round(1 / len(text) * line_width)
            # else:
            #     token_width = round(len(w.replace('##', '', 1)) / len(text) * line_width)

            start_x = min(start_x, x2)
            end_x = min(x2, start_x + token_width - 1)
            if start_x > end_x or y1 > y2:
                # print(line_width, tokenizer.tokenize(text))
                # print(token_width)
                # print('#' * 60, 'ERROR')
                # print(text, w, sep='|')
                continue

            current_word['text'] = w

            # current_word['box'] = [start_x, y1, end_x, y2]

            if current_line['points'] is None:
                current_word['box'] = [start_x, y1, end_x, y2]
                current_word['points'] = None
            else:
                polygon_box = divide_polygon_by_ratio(current_line['points'], 1 - _id * width_ratio, width_ratio)
                current_word['points'] = polygon_box
                x1, y1, x2, y2 = convert_pts2box(polygon_box)
                current_word['box'] = [max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)]

            start_x = end_x + 1
            words.append(current_word)
        # print('------')

        current_line['words'] = words

        # print(current_line)
        lines.append(current_line)

    # lines = sorted(lines, key=lambda x : (x['box'][1], x['box'][0]))
    lines = sort_funsd_reading_order(lines)

    return lines


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def get_examples_from_one_sample(image, annotation, tokenizer, max_seq_length, datapile_format=True):
    def normalize_box(box):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    funsd_annotation = convert_one_datapile_to_funsd(annotation, image, tokenizer, datapile_format=datapile_format)

    width, height = image.size

    examples = []
    token_cnt = 0

    tokens = []
    boxes = []
    actual_bboxes = []
    labels = []

    for item in funsd_annotation:
        words, label = item["words"], item["label"]
        words = [w for w in words if w["text"].strip() != ""]

        if len(words) == 0:
            continue

        current_len = len(words)

        if token_cnt + current_len > max_seq_length - 2:
            examples.append(
                InputExample(
                    guid="%s-%d".format('test', 1),
                    words=tokens,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name='test',
                    page_size=[width, height],
                )
            )

            token_cnt = 0

            tokens = []
            boxes = []
            actual_bboxes = []
            labels = []

        for w in words:
            tokens.append(w['text'])
            labels.append("O")
            actual_bboxes.append(w['box'])
            boxes.append(normalize_box(w['box']))

        token_cnt += current_len

    if token_cnt > 0:
        examples.append(
            InputExample(
                guid="%s-%d".format('test', 1),
                words=tokens,
                labels=labels,
                boxes=boxes,
                actual_bboxes=actual_bboxes,
                file_name='test',
                page_size=[width, height],
            )
        )

    features = convert_examples_to_features(
        examples,
        None,
        max_seq_length,
        tokenizer,
        is_tokenized=True,
        cls_token_at_end=False,
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=-100,
    )

    return features


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        is_tokenized=False,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    if label_list is not None:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = {}

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        token_boxes = []
        actual_bboxes = []
        label_ids = []

        assert len(example.words) == len(example.labels)
        non_blank_label_ids = []

        for word, label, box, actual_bbox in zip(
                example.words, example.labels, example.boxes, example.actual_bboxes
        ):
            assert box[0] <= box[2]
            assert box[1] <= box[3]
            
            if is_tokenized:
                word_tokens = [word]
            else:
                word_tokens = tokenizer.tokenize(word)
            assert len(word_tokens) > 0
            # print(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [label_map.get(label, 0)] + [pad_token_label_id] * (len(word_tokens) - 1)
            )
            non_blank_label_ids.append(label_map.get(label, 0))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
            actual_bboxes = ([pad_token_box] * padding_length) + actual_bboxes
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            token_boxes += [pad_token_box] * padding_length
            actual_bboxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(actual_bboxes) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                question=None,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )

    return features


def write_output_kv(output_path, filename, entities):
    report_path = os.path.join(output_path, filename)
    if not os.path.isdir(report_path):
        os.mkdir(report_path)
    report_dict = convert_kv_format(entities)
    with open(os.path.join(report_path, 'kv.json'), 'w') as fo:
        json.dump(report_dict, fo, indent=4, ensure_ascii=False)


def dist_points(p_a, p_b):
    return np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2)


def get_mid_points(p_a, p_b, ratio):
    return int(p_a[0] * ratio + p_b[0] * (1 - ratio)), int(p_a[1] * ratio + p_b[1] * (1 - ratio))


def get_closest_point(ref_point, point_lists):
    dist_list = [dist_points(ref_point, p) for p in point_lists]
    return point_lists[np.argmin(dist_list)]


def divide_polygon_by_ratio(points, offset_ratio, width_ratio):

    x1, y1, x2, y2 = box = convert_pts2box(points)
    bounding_pts = (x1, y1), (x2, y1), (x2, y2), (x1, y2)

    points = [get_closest_point(bounding_pts[i], points) for i in range(4)]

    top_points = [get_mid_points(points[0], points[1], offset_ratio), get_mid_points(
        points[0], points[1], offset_ratio - width_ratio)]

    bottom_points = [get_mid_points(points[3], points[2], offset_ratio), get_mid_points(
        points[3], points[2], offset_ratio - width_ratio)]

    return top_points + bottom_points[::-1]


def convert_pts2box(pts):
    list_x = [p[0] for p in pts]
    list_y = [p[1] for p in pts]

    return int(min(list_x)), int(min(list_y)), int(max(list_x)), int(max(list_y))