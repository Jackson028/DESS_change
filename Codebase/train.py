import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import transformers
from torch.optim import optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig

from Parameter import train_argparser
from models.D2E2S_Model import D2E2SModel
from models.General import set_seed
from trainer import util, sampling
from trainer.baseTrainer import BaseTrainer
from trainer.entities import Dataset
from trainer.evaluator import Evaluator
from trainer.input_reader import JsonInputReader
from trainer.loss import D2E2SLoss
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


class D2E2S_Trainer(BaseTrainer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._tokenizer = AutoTokenizer.from_pretrained(args.pretrained_deberta_name)
        self._predictions_path = os.path.join(
            self._log_path_predict, "predicted_%s_epoch_%s.json"
        )
        self._examples_path = os.path.join(
            self._log_path_predict, "sample_%s_%s_epoch_%s.html"
        )
        os.makedirs(self._log_path_result)
        os.makedirs(self._log_path_predict)
        self._dump_args(args)
        self.max_pair_f1 = 40
        self.result_path = os.path.join(
            self._log_path_result, "result{}.txt".format(self.args.max_span_size)
        )

    def _dump_args(self, args):
        path = os.path.join(self._log_path, "args.txt")
        with open(path, "w", encoding="utf-8") as f:
            for key, value in sorted(vars(args).items()):
                f.write("{}: {}\n".format(key, value))

    def _preprocess(self, args, input_reader_cls, types_path, train_path, test_path, dev_path=None):

        train_label, test_label, dev_label = "train", "test", "dev"
        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(test_label)
        use_dev = args.select_model_by_dev and dev_path and os.path.exists(dev_path)
        if use_dev:
            self._init_eval_logging(dev_label)

        # loading data
        input_reader = input_reader_cls(
            types_path,
            self._tokenizer,
            args.neg_entity_count,
            args.neg_triple_count,
            args.max_span_size,
        )
        dataset_paths = {train_label: train_path, test_label: test_path}
        if use_dev:
            dataset_paths[dev_label] = dev_path
        input_reader.read(dataset_paths)
        train_dataset = input_reader.get_dataset(train_label)

        # preprocess
        train_sample_count = train_dataset.sentence_count
        updates_epoch = train_sample_count // args.batch_size
        updates_total = updates_epoch * args.epochs

        print("   ", self.args.dataset, "  ", self.args.max_span_size)
        return input_reader, updates_total, updates_epoch

    def _train(
        self,
        train_path: str,
        test_path: str,
        types_path: str,
        input_reader_cls,
        dev_path: str = None,
    ):
        args = self.args

        # set seed
        set_seed(args.seed)

        train_label, test_label, dev_label = "train", "test", "dev"
        input_reader, updates_total, updates_epoch = self._preprocess(
            args, input_reader_cls, types_path, train_path, test_path, dev_path
        )
        train_dataset = input_reader.get_dataset(train_label)
        test_dataset = input_reader.get_dataset(test_label)
        dev_dataset = None
        if args.select_model_by_dev and dev_path and os.path.exists(dev_path):
            dev_dataset = input_reader.get_dataset(dev_label)
        if args.select_model_by_dev and dev_dataset is None:
            raise ValueError(
                "--select_model_by_dev requires an existing dev dataset file"
            )

        # load model
        config = AutoConfig.from_pretrained(args.pretrained_deberta_name)

        model = D2E2SModel(
            config=config,
            cls_token=self._tokenizer.convert_tokens_to_ids("[CLS]"),
            sentiment_types=input_reader.sentiment_type_count - 1,
            entity_types=input_reader.entity_type_count,
            args=args,
        )
        model.to(args.device)
        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.lr_warmup * updates_total,
            num_training_steps=updates_total,
        )

        # create loss function
        entity_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        senti_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        compute_loss = D2E2SLoss(
            senti_criterion,
            entity_criterion,
            model,
            optimizer,
            scheduler,
            args.max_grad_norm,
        )
        # eval validation set
        if args.init_eval:
            self._eval(model, test_dataset, input_reader, 0, updates_epoch)

        # track best dev
        best_dev_f1 = -1.0
        best_dev_test_f1 = 0.0
        best_dev_epoch = 0
        best_model_path = os.path.join(
            args.save_path, "best_model_{}.pt".format(args.dataset)
        )

        # train
        for epoch in range(args.epochs):
            # train epoch
            self.train_epoch(
                model, compute_loss, optimizer, train_dataset, updates_epoch, epoch
            )

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                if args.select_model_by_dev:
                    _, dev_eval, _ = self._eval(
                        model, dev_dataset, input_reader, epoch + 1, updates_epoch,
                        track_best=False,
                    )
                    dev_f1 = float(dev_eval[2])
                    is_new_best = dev_f1 > best_dev_f1
                    _, test_eval, _ = self._eval(
                        model, test_dataset, input_reader, epoch + 1, updates_epoch,
                        track_best=False, save_artifacts=is_new_best,
                    )
                    test_f1 = float(test_eval[2])
                    if is_new_best:
                        best_dev_f1 = dev_f1
                        best_dev_test_f1 = test_f1
                        best_dev_epoch = epoch + 1
                        os.makedirs(args.save_path, exist_ok=True)
                        torch.save(model.state_dict(), best_model_path)
                        print(
                            "Epoch {}: New best dev F1 = {:.4f}, test F1 = {:.4f}, model saved.".format(
                                epoch + 1, best_dev_f1, best_dev_test_f1
                            )
                        )
                    else:
                        print(
                            "Epoch {}: dev F1 = {:.4f}, test F1 = {:.4f}".format(
                                epoch + 1, dev_f1, test_f1
                            )
                        )
                else:
                    self._eval(model, test_dataset, input_reader, epoch + 1, updates_epoch)

        if args.select_model_by_dev:
            print(
                "Training complete. Best dev F1: {:.4f} at epoch {}; dev-selected test F1: {:.4f}".format(
                    best_dev_f1, best_dev_epoch, best_dev_test_f1
                )
            )

    def train_epoch(
        self,
        model: torch.nn.Module,
        compute_loss: D2E2SLoss,
        optimizer: optimizer,
        dataset: Dataset,
        updates_epoch: int,
        epoch: int,
    ):

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        model.zero_grad()

        iteration = 0
        total = dataset.sentence_count // self.args.batch_size
        for batch in tqdm(data_loader, total=total, desc="Train epoch %s" % epoch):
            model.train()
            batch = util.to_device(batch, arg_parser.device)

            # forward step
            entity_logits, senti_logits, batch_loss, auxiliary_losses = model(
                encodings=batch["encodings"],
                context_masks=batch["context_masks"],
                entity_masks=batch["entity_masks"],
                entity_sizes=batch["entity_sizes"],
                sentiments=batch["rels"],
                senti_masks=batch["senti_masks"],
                adj=batch["adj"],
                senti_types=batch.get("senti_types"),
                senti_sample_masks=batch.get("senti_sample_masks"),
            )

            # compute loss and optimize parameters
            epoch_loss = compute_loss.compute(
                entity_logits=entity_logits,
                senti_logits=senti_logits,
                batch_loss=batch_loss,
                senti_types=batch["senti_types"],
                entity_types=batch["entity_types"],
                entity_sample_masks=batch["entity_sample_masks"],
                senti_sample_masks=batch["senti_sample_masks"],
                auxiliary_losses=auxiliary_losses,
            )

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(
                    optimizer,
                    epoch_loss,
                    epoch,
                    iteration,
                    global_iteration,
                    dataset.label,
                )

        return iteration

    def _log_train(
        self,
        optimizer: optimizer,
        loss: float,
        epoch: int,
        iteration: int,
        global_iteration: int,
        label: str,
    ):
        # average loss
        avg_loss = loss / self.args.batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # to_csv
        # log to csv
        self._log_csv(label, "loss", loss, epoch, iteration, global_iteration)
        self._log_csv(label, "loss_avg", avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, "lr", lr, epoch, iteration, global_iteration)

        # log to tensorboard
        self._log_tensorboard(label, "loss", loss, global_iteration)
        self._log_tensorboard(label, "loss_avg", avg_loss, global_iteration)
        self._log_tensorboard(label, "lr", lr, global_iteration)

    def _eval(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        input_reader: JsonInputReader,
        epoch: int = 0,
        updates_epoch: int = 0,
        iteration: int = 0,
        track_best: bool = True,
        save_artifacts: bool = False,
    ):

        # create evaluator
        evaluator = Evaluator(
            dataset,
            input_reader,
            self._tokenizer,
            self.args.sen_filter_threshold,
            self._predictions_path,
            self._examples_path,
            self.args.example_count,
            epoch,
            dataset.label,
        )
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        with torch.no_grad():
            model.eval()
            # iterate batches
            total = math.ceil(dataset.sentence_count / self.args.batch_size)
            for batch in tqdm(
                data_loader, total=total, desc="Evaluate epoch %s" % epoch
            ):
                # move batch to selected device
                batch = util.to_device(batch, self.args.device)

                # run model (forward pass)
                result = model(
                    encodings=batch["encodings"],
                    context_masks=batch["context_masks"],
                    entity_masks=batch["entity_masks"],
                    entity_sizes=batch["entity_sizes"],
                    entity_spans=batch["entity_spans"],
                    entity_sample_masks=batch["entity_sample_masks"],
                    evaluate=True,
                    adj=batch["adj"],
                )
                entity_clf, senti_clf, rels = result
                # evaluate batch, entity:tensor(16, 188, 3), senti_clf:tensor(16, 2, 4), rels:tensor(16, 2, 2)
                evaluator.eval_batch(entity_clf, senti_clf, rels, batch)
            global_iteration = epoch * updates_epoch + iteration
            ner_eval, senti_eval, senti_nec_eval = evaluator.compute_scores()
            # print(self.result_path)
            if track_best:
                self._log_filter_file(ner_eval, senti_eval, evaluator, epoch)
            elif save_artifacts:
                self._store_eval_artifacts(evaluator)
        self._log_eval(
            *ner_eval,
            *senti_eval,
            *senti_nec_eval,
            epoch,
            iteration,
            global_iteration,
            dataset.label
        )
        return ner_eval, senti_eval, senti_nec_eval

    def _log_filter_file(self, ner_eval, senti_eval, evaluator, epoch):
        f1 = float(senti_eval[2])
        if self.max_pair_f1 < f1:
            columns = [
                "mic_precision",
                "mic_recall",
                "mic_f1_score",
                "mac_precision",
                "mac_recall",
                "mac_f1_score",
            ]
            ner_dic = {
                "mic_precision": 0.0,
                "mic_recall": 0.0,
                "mic_f1_score": 0.0,
                "mac_precision": 0.0,
                "mac_recall": 0.0,
                "mac_f1_score": 0.0,
            }
            senti_dic = {
                "mic_precision": 0.0,
                "mic_recall": 0.0,
                "mic_f1_score": 0.0,
                "mac_precision": 0.0,
                "mac_recall": 0.0,
                "mac_f1_score": 0.0,
            }
            # for inx, val in enumerate(ner_eval):
            #     ner_dic[columns[inx]] = val
            for inx, val in enumerate(senti_eval):
                senti_dic[columns[inx]] = val
            self.max_pair_f1 = f1
            with open(self.result_path, mode="a", encoding="utf-8") as f:
                w_str = "No. {} ：....\n".format(epoch)
                f.write(w_str)
                f.write("ner_entity: \n")
                f.write(str(ner_dic))
                f.write("\n rec: \n")
                f.write(str(senti_dic))
                f.write("\n")
            try:
                fileNames = os.listdir(self._log_path_predict)
                # print(fileNames)
                for i in fileNames:
                    os.remove(os.path.join(self._log_path_predict, i))
            except BaseException:
                print(BaseException)
            if self.args.store_predictions:
                evaluator.store_predictions()

            if self.args.store_examples:
                evaluator.store_examples()

    def _store_eval_artifacts(self, evaluator):
        try:
            for name in os.listdir(self._log_path_predict):
                os.remove(os.path.join(self._log_path_predict, name))
        except BaseException:
            print(BaseException)
        if self.args.store_predictions:
            evaluator.store_predictions()
        if self.args.store_examples:
            evaluator.store_examples()

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_params


if __name__ == "__main__":
    arg_parser = train_argparser()
    trainer = D2E2S_Trainer(arg_parser)
    trainer._train(
        train_path=arg_parser.dataset_file["train"],
        test_path=arg_parser.dataset_file["test"],
        types_path=arg_parser.dataset_file["types_path"],
        input_reader_cls=JsonInputReader,
        dev_path=arg_parser.dataset_file.get("dev"),
    )
