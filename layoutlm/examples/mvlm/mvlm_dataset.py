import os
import math
import random
import logging
from copy import deepcopy

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetForTokenClassification(Dataset):
    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode, augment=False):
        
        model_name_or_path = args.bert_model if args.bert_only else args.layoutlm_model
        self.augment = augment
        
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_train_{}_{}_{}".format(
                mode,
                list(filter(None, model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args.data_dir, mode)
            features = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                args.is_tokenized,
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=pad_token_label_id,
            )

            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    # randomly resize the bounding boxes
    def _augment_boxes(self, boxes, stdev_x=0.15, stdev_y=0.1):
        min_dim = torch.min(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).unsqueeze(dim=-1)
        offset_ratio = torch.normal(torch.zeros(boxes.shape), torch.tensor((stdev_x, stdev_y, stdev_x, stdev_y)))
        new_boxes = boxes + torch.floor(min_dim * offset_ratio)
        min_boxes = new_boxes.clone()
        min_boxes[:, 2] = min_boxes[:, 0] + 1
        min_boxes[:, 3] = min_boxes[:, 1] + 1
        new_boxes = torch.clamp(torch.max(new_boxes, min_boxes), 0, 1000).long()

        return new_boxes


    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_attention_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index] if not self.augment else self._augment_boxes(self.all_bboxes[index]),
        )


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


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        is_tokenized,
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

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        #     logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
        #     logger.info("actual_bboxes: %s", " ".join([str(x) for x in actual_bboxes]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )

    return features


class DatasetForMaskedVisualLM:

    def __init__(self, args, tokenizer, mode, mlm_probability=0.15):
        model_name_or_path = args.bert_model if args.bert_model else args.layoutlm_model
        
        cached_features_file = os.path.join(
            args.data_dir,
            "mvlm_cached_{}_{}_{}".format(
                mode,
                list(filter(None, model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )

        self.tokenizer = tokenizer
        self.is_tokenized = args.is_tokenized
        self.mode = mode
        self.data_dir = args.data_dir
        self.max_seq_length = args.max_seq_length
        self.mlm_probability = mlm_probability
        if mode == 'train':
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.eval_batch_size

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = self.read_examples_from_file()
            features = self.convert_examples_to_features(examples, self.is_tokenized)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        self.features = sorted(features, key=lambda x: len(x.input_ids))

        self.num_batch = math.ceil(len(self.features) / self.batch_size)

        self.remaining_indices = [i for i in range(len(self.features))]
        random.shuffle(self.remaining_indices)

    def __len__(self):
        return len(self.features)

    def next_batch(self):
        if len(self.remaining_indices) == 0:
            self.remaining_indices = [i for i in range(len(self.features))]
            random.shuffle(self.remaining_indices)

        batch = self._collate_batch([self.features[i]
                                     for i in self.remaining_indices[:self.batch_size]])
        batch = self.mask_tokens(batch)
        self.remaining_indices = self.remaining_indices[self.batch_size:]
        return batch

    def read_examples_from_file(self):
        box_file_path = os.path.join(self.data_dir, "{}_box.txt".format(self.mode))
        image_file_path = os.path.join(self.data_dir, "{}_image.txt".format(self.mode))
        guid_index = 1
        examples = []
        with open(box_file_path, encoding="utf-8"
                  ) as fb, open(image_file_path, encoding="utf-8") as fi:
            words = []
            boxes = []
            actual_bboxes = []
            file_name = None
            page_size = None
            for bline, iline in zip(fb, fi):
                if bline.startswith("-DOCSTART-") or bline == "" or bline == "\n":
                    if words:
                        examples.append(
                            InputExample(
                                guid="{}-{}".format(self.mode, guid_index),
                                words=words,
                                labels='default',
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
                else:
                    bsplits = bline.split("\t")
                    isplits = iline.split("\t")
                    assert len(bsplits) == 2
                    assert len(isplits) == 4
                    assert bsplits[0] == isplits[0]
                    words.append(bsplits[0])
                    box = bsplits[-1].replace("\n", "")
                    box = [int(b) for b in box.split()]
                    boxes.append(box)
                    actual_bbox = [int(b) for b in isplits[1].split()]
                    actual_bboxes.append(actual_bbox)
                    page_size = [int(i) for i in isplits[2].split()]
                    file_name = isplits[3].strip()
            if words:
                examples.append(
                    InputExample(
                        guid="%s-%d".format(self.mode, guid_index),
                        words=words,
                        labels='default',
                        boxes=boxes,
                        actual_bboxes=actual_bboxes,
                        file_name=file_name,
                        page_size=page_size,
                    )
                )

        return examples

    def convert_examples_to_features(
            self,
            examples,
            is_tokenized,
            cls_token_at_end=False,
            cls_token_segment_id=0,
            cls_token_box=[0, 0, 0, 0],
            sep_token_box=[1000, 1000, 1000, 1000],
            pad_token_box=[0, 0, 0, 0],
            sequence_a_segment_id=0,
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

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
            for word, box, actual_bbox in zip(
                    example.words, example.boxes, example.actual_bboxes
            ):
                if is_tokenized:
                    word_tokens = [word]
                else:
                    word_tokens = self.tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))
                actual_bboxes.extend([actual_bbox] * len(word_tokens))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 2
            if len(tokens) > self.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.max_seq_length - special_tokens_count)]
                token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
                actual_bboxes = actual_bboxes[: (self.max_seq_length - special_tokens_count)]

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

            # add sep_token
            tokens += [self.tokenizer.sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [self.tokenizer.cls_token]
                token_boxes += [cls_token_box]
                actual_bboxes += [[0, 0, width, height]]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [self.tokenizer.cls_token] + tokens
                token_boxes = [cls_token_box] + token_boxes
                actual_bboxes = [[0, 0, width, height]] + actual_bboxes
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1] * len(input_ids)

            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s", example.guid)
            #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            #     logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            #     logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
            #     logger.info("actual_bboxes: %s", " ".join([str(x) for x in actual_bboxes]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    label_ids=input_ids,
                    boxes=token_boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )

        return features

    def _collate_batch(self, examples, pad_token_box=[0, 0, 0, 0]):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

        # Check if padding is necessary.
        length_of_first = len(examples[0].input_ids)
        are_tensors_same_length = all(len(x.input_ids) == length_of_first
                                      for x in examples)
        if are_tensors_same_length:
            return {
                'input_ids': torch.tensor([e.input_ids for e in examples], dtype=torch.long),
                'label_ids': torch.tensor([e.input_ids for e in examples], dtype=torch.long),
                'attention_mask': torch.tensor([e.attention_mask for e in examples], dtype=torch.long),
                'boxes': torch.tensor([e.boxes for e in examples], dtype=torch.long),
                'special_tokens_mask': torch.tensor([[1] + [0] * (len(e.input_ids) - 2) + [1]
                                                     for e in examples],
                                                    dtype=torch.long)
            }

        # If yes, check if we have a `pad_token`.
        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(len(x.input_ids) for x in examples)
        input_ids = []
        attention_mask = []
        boxes = []
        special_tokens_mask = []
        for e in examples:
            padding_length = max_length - len(e.input_ids)
            input_ids.append(e.input_ids + [self.tokenizer.pad_token_id] * padding_length)
            attention_mask.append(e.attention_mask + [0] * padding_length)
            boxes.append(e.boxes + [pad_token_box for _ in range(padding_length)])
            this_special_tokens_mask = [1] + [0] * (len(e.input_ids) - 2) + [1]
            special_tokens_mask.append(this_special_tokens_mask + [1] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.long),
            'special_tokens_mask': torch.tensor(special_tokens_mask, dtype=torch.long)
        }

    def mask_tokens(self, batch):
        batch_shape = batch['input_ids'].shape

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(batch_shape, self.mlm_probability)
        special_tokens_mask = batch['special_tokens_mask'].bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch['label_ids'][~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(batch_shape, 0.8)).bool() & masked_indices
        batch['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(batch_shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), batch_shape, dtype=torch.long)
        batch['input_ids'][indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return batch