import os
import glob
import json
import time
import logging
import copy
import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from imutils import paths
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    WEIGHTS_NAME,
    LayoutLMTokenizer,
    LayoutLMForMaskedLM,
    LayoutLMModel,
    LayoutLMConfig,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)

# from datasets import DatasetForMaskedVisualLM
from utils import *
from mvlm_dataset import DatasetForMaskedVisualLM

logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer, mode, prefix="", verbose=True):
    eval_dataset = DatasetForMaskedVisualLM(args, tokenizer, mode=mode)

    # Eval!
    logger.info("***** Running evaluation - %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # eval_loss = 0.0
    eval_loss_list = []
    eval_acc = 0
    nb_eval_steps = 0
    # pred_output_report = []

    model.eval()

    for batch in tqdm([eval_dataset.next_batch() for _ in range(eval_dataset.num_batch)],
                      desc="Evaluating", disable=not verbose):
        with torch.no_grad():
            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'labels': batch['label_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'bbox': batch['boxes'].to(args.device)
            }

            # print(batch['input_ids'].shape)
            decoded_text_input = [tokenizer.decode(item) for item in batch['input_ids']]
            # print('\n'.join(decoded_text_input))
            # bbox_input = batch['boxes']

            # for text, bbox in zip(decoded_text_input, bbox_input):
            #     pred_output_report.append((text, bbox))
            #     print(text)

            outputs = model(**inputs)
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            # eval_loss += tmp_eval_loss.item()
            eval_loss_list.append(tmp_eval_loss.item())

        nb_eval_steps += 1

        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=-1)
        labels = inputs['labels'].detach().cpu().numpy()

        if verbose:
            for p, l, inp in zip(preds, labels, decoded_text_input):
                print('Input\n{}\n'.format(inp))
                p = np.array([p[i] for i, x in enumerate(l) if x != -100])
                l = np.array([x for x in l if x != -100])
                print('Pred\n{}\n'.format(tokenizer.decode(p)))
                print(p)
                print('Label\n{}\n'.format(tokenizer.decode(l)))
                print(l)
                print('acc {:.2f}'.format(np.mean(p == l)))
                eval_acc += np.mean(p == l) if len(p) > 0 else 0.0
        else:
            for p, l, inp in zip(preds, labels, decoded_text_input):
                p = np.array([p[i] for i, x in enumerate(l) if x != -100])
                l = np.array([x for x in l if x != -100])
                eval_acc += np.mean(p == l) if len(p) > 0 else 0.0

    # print(eval_loss_list)
    eval_loss = np.average(eval_loss_list) if len(eval_loss_list) > 0 else 0.0  # eval_loss / nb_eval_steps
    eval_acc = eval_acc / len(eval_dataset)

    results = {
        "loss": eval_loss,
        "acc": eval_acc,
        "perplexity": np.exp(eval_loss)
    }

    if verbose:
        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results


def train(args, train_dataset, model, tokenizer):
    summary_writer = SummaryWriter(logdir="runs/" + os.path.basename(args.output_dir))

    t_total = train_dataset.num_batch // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch")

    best_loss = 999
    best_f1 = 0

    if args.use_val:
        val_results = evaluate(
            args,
            model,
            tokenizer,
            mode="test",
            verbose=False
        )
        logger.info('val: %s', val_results)
        for key, value in val_results.items():
            summary_writer.add_scalar(
                "val_{}".format(key), value, global_step
            )

        if val_results['loss'] < best_loss:
            best_loss = val_results['loss']
            print('saving best loss', best_loss)
            # Save model checkpoint
            output_dir = os.path.join(
                args.output_dir,
                ''.join("""ep-{}-val_loss-{:.2f}
                        """.split()).format(-1, val_results['loss'])
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))

            logger.info("Saving model checkpoint to %s", output_dir)

    for i in train_iterator:
        print()
        print('Epoch', i, '=' * 17)
        epoch_iterator = tqdm([train_dataset.next_batch() for _ in range(train_dataset.num_batch)],
                              desc='Epochs {}'.format(i))

        for step, batch in enumerate(epoch_iterator):
            model.train()

            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'labels': batch['label_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'bbox': batch['boxes'].to(args.device)
            }

            outputs = model(**inputs)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    summary_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    summary_writer.add_scalar(
                        "train_loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        train_results = evaluate(
            args,
            model,
            tokenizer,
            mode="train",
            verbose=False
        )
        logger.info('train: %s', train_results)

        if args.use_val and i % 5 == 0:
            val_results = evaluate(
                args,
                model,
                tokenizer,
                mode="test",
                verbose=False
            )
            logger.info('val: %s', val_results)
            for key, value in val_results.items():
                summary_writer.add_scalar(
                    "val_{}".format(key), value, global_step
                )

            if val_results['loss'] < best_loss:
                best_loss = val_results['loss']
                print('saving best loss', best_loss)
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir,
                    ''.join("""ep-{}-val_loss-{:.2f}-train_loss-{:.2f}
                            """.split()).format(i, val_results['loss'], train_results['loss'])
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

        print()
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # print('saving final model')
    # # Save model checkpoint
    # output_dir = os.path.join(
    #     args.output_dir,
    #     'ep-{}-train_loss-{:.2f}-train_f1-{:.2f}'.format(i,
    #                                                      train_results['loss'],
    #                                                      train_results['f1'])
    # )
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
    # logger.info("Saving model checkpoint to %s", output_dir)

    summary_writer.close()

    return global_step, tr_loss / global_step


args = dict(
    data_dir='../seq_labeling/data/', 
    max_seq_length=512, 
    layoutlm_model='microsoft/layoutlm-base-uncased', 
    model_type='layoutlm', 
    num_train_epochs=100, 
    learning_rate=5e-5,
    weight_decay=0.0,
    output_dir='./outputs/', 
    overwrite_cache=False, 
    train_batch_size=1,
    eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    adam_epsilon=1e-8,
    max_steps=-1,
    save_steps=-1, 
    logging_steps=1,
    max_grad_norm=1.0, 
    device='cuda',
    eval_all_checkpoints=True,
    use_val=True,
    load_pretrain=True,
    freeze_lm=False,
    so_only=True,
    bert_model=None,
    bert_only=False,
    is_tokenized=False,
    retrain_word_embedder=False,
    retrain_layout_embedder=False,
    test_only=False
)
class Args:
    def __init__(self, args):
        self.__dict__ = args

args = Args(args)

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(args.output_dir, "train.log"),
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger.addHandler(logging.StreamHandler())

if not args.test_only:
    if args.load_pretrain:
        model = LayoutLMForMaskedLM.from_pretrained(args.layoutlm_model,
                                                    return_dict=True)
        tokenizer = LayoutLMTokenizer.from_pretrained(args.layoutlm_model)
        print('Loading pre-trained model from', args.layoutlm_model)
    else:
        config = LayoutLMConfig.from_pretrained(args.model_name_or_path,
                                                return_dict=True)
        if args.bert_model is not None:
            tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            config.vocab_size = tokenizer.vocab_size

        model = LayoutLMForMaskedLM(config)

    if args.bert_model is None:
        tokenizer = LayoutLMTokenizer.from_pretrained(args.layoutlm_model,
                                                      do_lower_case=True)
    else:
        bert = BertModel.from_pretrained(args.bert_model)
        model.layoutlm.embeddings.word_embeddings = \
            copy.deepcopy(bert.embeddings.word_embeddings)
        del bert

    if args.layoutlm_model is not None and not args.load_pretrain:
        layout_model = LayoutLMForMaskedLM.from_pretrained(args.layoutlm_model, return_dict=True)

        model.layoutlm.embeddings.position_embeddings = \
            copy.deepcopy(layout_model.layoutlm.embeddings.position_embeddings)
        model.layoutlm.embeddings.x_position_embeddings = \
            copy.deepcopy(layout_model.layoutlm.embeddings.x_position_embeddings)
        model.layoutlm.embeddings.y_position_embeddings = \
            copy.deepcopy(layout_model.layoutlm.embeddings.y_position_embeddings)
        model.layoutlm.embeddings.h_position_embeddings = \
            copy.deepcopy(layout_model.layoutlm.embeddings.h_position_embeddings)
        model.layoutlm.embeddings.w_position_embeddings = \
            copy.deepcopy(layout_model.layoutlm.embeddings.w_position_embeddings)
        model.layoutlm.encoder = \
            copy.deepcopy(layout_model.layoutlm.encoder)
        model.layoutlm.pooler = \
            copy.deepcopy(layout_model.layoutlm.pooler)

    print(tokenizer)
    print(model.config)

    model.to(args.device)

    if args.freeze_lm:
        for param in model.layoutlm.parameters():
            param.requires_grad = False

    logger.info("Training/evaluation parameters %s", args.__dict__)

    logger.info('Training...')

    train_dataset = DatasetForMaskedVisualLM(args, tokenizer, mode="train")

    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    #
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving final model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

logger.info('Evaluating...')

checkpoints = [args.output_dir]
if args.eval_all_checkpoints:
    checkpoints = list(
        os.path.dirname(c)
        for c in sorted(
            glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
        )
    )
    logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
        logging.WARN
    )  # Reduce logging

results = {}
for checkpoint in checkpoints:
    print(checkpoint)
    model = LayoutLMForMaskedLM.from_pretrained(checkpoint)
    tokenizer = LayoutLMTokenizer.from_pretrained(checkpoint)

    if args.bert_model is None:
        tokenizer = LayoutLMTokenizer.from_pretrained(args.layoutlm_model,
                                                      do_lower_case=True)
    else:
        bert = BertModel.from_pretrained(args.bert_model)
        model.layoutlm.embeddings.word_embeddings = \
            copy.deepcopy(bert.embeddings.word_embeddings)
        del bert

    _ = model.to(args.device)
    result = evaluate(
        args,
        model,
        tokenizer,
        mode="test",
        prefix=checkpoint,
    )

    print('\n')
    results[checkpoint] = result

import json

output_eval_file = os.path.join(args.output_dir, "eval_results.json")
with open(output_eval_file, "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

import pprint

pprint.pprint(results)