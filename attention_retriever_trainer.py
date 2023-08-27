import warnings
import logging
import hydra
import hydra.utils as hu
import numpy as np
from datasets import load_metric
from transformers import Trainer, EvalPrediction, EarlyStoppingCallback, set_seed
from src.utils.collators import AttentionRetrieverDataCollator
from src.models.attention_model import AttentionModel

logger = logging.getLogger(__name__)


class AttentionRetrieverTrainer:

    def __init__(self, cfg) -> None:
        self.training_args = hu.instantiate(cfg.training_args)
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.index_reader = hu.instantiate(cfg.index_reader)
        encoded_index = list(self.index_reader)

        train_dataset, eval_dataset = self.dataset_reader.split_dataset(test_size=0.1, seed=42)
        logger.info(f"train size: {len(train_dataset)}, eval size: {len(eval_dataset)}")

        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path is not None:
            self.model = AttentionModel.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = AttentionModel(model_config)

        data_collator = AttentionRetrieverDataCollator(
            tokenizer=self.dataset_reader.tokenizer,
            encoded_index=encoded_index,
            **cfg.collector,
        )

        self.metric = load_metric('src/metrics/accuracy.py')
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.dataset_reader.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.training_args.output_dir)
        self.trainer.tokenizer.save_pretrained(self.training_args.output_dir)

    def evaluate(self):
        return self.trainer.evaluate()
    
    def compute_metrics(self, p: EvalPrediction):
        print(
            "\n" + "-" * 50
            + f"\n[*] label[0]: {[x for x in p.label_ids[0] if x != -1]}"
            + f"\n[*] pred[0]: {[x for x in p.predictions[0] if x != -1]}"
            + f"\n[*] label[1]: {[x for x in p.label_ids[1] if x != -1]}"
            + f"\n[*] pred[1]: {[x for x in p.predictions[1] if x != -1]}"
            + f"\n[*] label[2]: {[x for x in p.label_ids[2] if x != -1]}"
            + f"\n[*] pred[2]: {[x for x in p.predictions[2] if x != -1]}"
            + f"\n[*] label[3]: {[x for x in p.label_ids[3] if x != -1]}"
            + f"\n[*] pred[3]: {[x for x in p.predictions[3] if x != -1]}"
            + f"\n[*] label[4]: {[x for x in p.label_ids[4] if x != -1]}"
            + f"\n[*] pred[4]: {[x for x in p.predictions[4] if x != -1]}"
            + f"\n[*] label[5]: {[x for x in p.label_ids[5] if x != -1]}"
            + f"\n[*] pred[5]: {[x for x in p.predictions[5] if x != -1]}"
            + f"\n[*] label[6]: {[x for x in p.label_ids[6] if x != -1]}"
            + f"\n[*] pred[6]: {[x for x in p.predictions[6] if x != -1]}"
            + f"\n[*] label[7]: {[x for x in p.label_ids[7] if x != -1]}"
            + f"\n[*] pred[7]: {[x for x in p.predictions[7] if x != -1]}"
            + f"\n[*] label[8]: {[x for x in p.label_ids[8] if x != -1]}"
            + f"\n[*] pred[8]: {[x for x in p.predictions[8] if x != -1]}"
            + f"\n[*] label[9]: {[x for x in p.label_ids[9] if x != -1]}"
            + f"\n[*] pred[9]: {[x for x in p.predictions[9] if x != -1]}"
            "\n" + "-" * 50
        )

        predictions = p.predictions
        references = p.label_ids
        sample_weight = (references != -1).astype(np.float32)

        return self.metric.compute(
            predictions=predictions,
            references=references,
            normalize=True,
            sample_weight=sample_weight,
            batch=True,
        )
    

@hydra.main(config_path="configs", config_name="attention_retriever_trainer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)

    trainer = AttentionRetrieverTrainer(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.training_args.do_train:
            trainer.train()
        if cfg.training_args.do_eval:
            logger.info(trainer.evaluate())


if __name__ == "__main__":
    main()
