from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from src.dataset_readers.training_dsr import InputSample


class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        return self.data


def ignore_pad_dict(features):
    res_dict = {}
    if "metadata" in features[0]:
        res_dict['metadata'] = ListWrapper([x.pop("metadata") for x in features])
    return res_dict


@dataclass
class DataCollatorWithPaddingAndCuda:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchEncoding:
        res_dict = ignore_pad_dict(features)

        has_labels = "labels" in features[0]
        if has_labels:
            labels = [{"input_ids": x.pop("labels")} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )

        # print(features)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        if has_labels:
            batch['labels'] = labels.input_ids
        batch.update(res_dict)

        if self.device:
            batch = batch.to(self.device)

        return batch


@dataclass
class IBNDataCollatorWithPadding:
    """
    In-batch negative data collector for training
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None
    encoded_index: List[Dict] = None
    pos_topk: int = 5
    neg_topk: int = -5
    hard_neg_per_step: int = 1
    pair_wise: bool = False

    def pad(self, list_of_list):
            padded = self.tokenizer.pad(
                {"input_ids": list_of_list},
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return padded['input_ids'], padded['attention_mask']

    def create_infonce_ctxs(self, sample):
        # randomly select 1 positive examples
        pos_candidates = sample.ctxs_candidates[:self.pos_topk]
        positive_ctxs_idx = np.random.choice(len(pos_candidates))
        positive_ctxs = pos_candidates[positive_ctxs_idx]

        # randomly select hard_neg_per_step hard negative examples
        if self.neg_topk == -1:
            neg_candidates = sample.ctxs_candidates[(positive_ctxs_idx + 1):]
        else:
            neg_candidates = sample.ctxs_candidates[self.neg_topk:]

        negative_ctxs_idx = np.random.choice(len(neg_candidates), size=self.hard_neg_per_step, replace=False)
        negative_ctxs = [neg_candidates[i] for i in sorted(negative_ctxs_idx)]

        all_ctxs = [positive_ctxs] + negative_ctxs
        return all_ctxs

    def create_pairwise_ctxs(self, sample):
        return sample.ctxs_candidates

    def __call__(self, samples: List[InputSample]) -> BatchEncoding:
        """
        Args:
            samples:  List of "InputSample":
                                [
                                    "question_ids",
                                    "ctxs_candidates"
                                ]
        Returns:
            data dict in BatchEncoding:
            [
                "questions_tensor",  # 2d, question token id
                "questions_attn_mask",
                "ctxs_tensor",  # 2d, context token id, gather all contexts in a batch
                "ctxs_attn_mask",
                "ctx_indices",  # 2d, value indicates idx in ctxs_tensor, size: [batch*(1+neg), num_ice]
                "labels"  # 1d, indicate pos index in ctx_indices, size: batch
            ]
        """

        questions_tensor = []
        ctxs_tensor = []
        ctx_indices = []
        eid2idx = {}

        for sample in samples:
            if self.pair_wise:
                all_ctxs = self.create_pairwise_ctxs(sample)
            else:
                all_ctxs = self.create_infonce_ctxs(sample)

            for i, ctxs in enumerate(all_ctxs):
                indices = []
                for eid in ctxs:
                    if eid in eid2idx:
                        indices.append(eid2idx[eid])
                    else:
                        indices.append(len(eid2idx))
                        eid2idx[eid] = len(eid2idx)
                        ctxs_tensor.append(self.encoded_index[eid]['input_ids'])
                ctx_indices.append(indices)

            questions_tensor.append(sample.question_ids)

        questions_tensor, questions_attn_mask = self.pad(questions_tensor)
        ctxs_tensor, ctxs_attn_mask = self.pad(ctxs_tensor)
        ctx_indices = torch.tensor(ctx_indices)  # assume num_ice is same for each candidate here

        if self.pair_wise:
            labels = torch.zeros(ctx_indices.shape[0])
        else:
            # labels = torch.zeros(len(samples), dtype=torch.long)
            labels = torch.arange(0, len(samples)) * (1 + self.hard_neg_per_step)

        batch = {"questions_tensor": questions_tensor,
                 "questions_attn_mask": questions_attn_mask,
                 "ctxs_tensor": ctxs_tensor,
                 "ctxs_attn_mask": ctxs_attn_mask,
                 "ctx_indices": ctx_indices,
                 "labels": labels
                 }
        return BatchEncoding(data=batch)
    

@dataclass
class AttentionRetrieverDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None
    encoded_index: List[Dict] = None
    num_ice: int = 50

    def pad(self, list_of_list):
            padded = self.tokenizer.pad(
                {"input_ids": list_of_list},
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return padded['input_ids'], padded['attention_mask']

    def pad_zero(self, input_list, target_length):
        num_pads_needed = target_length - len(input_list)
        return input_list + [0] * num_pads_needed

    def pad_minus_one(self, input_list, target_length):
        num_pads_needed = target_length - len(input_list)
        return input_list + [-1] * num_pads_needed
    
    def __call__(self, samples: List[InputSample]) -> BatchEncoding:
        questions_tensor = []
        ctxs_tensor = []
        ctx_indices = []
        labels = []
        delta_scores = []
        eid2idx = {}

        for sample in samples:  # num_iterations = B
            questions_tensor.append(sample.question_ids)

            indices = []
            for eid in sample.candidates:  # num_iterations = L
                if eid in eid2idx:
                    indices.append(eid2idx[eid])
                else:
                    indices.append(len(eid2idx))
                    eid2idx[eid] = len(eid2idx)
                    ctxs_tensor.append(self.encoded_index[eid]['input_ids'])
            ctx_indices.append(indices)

            delta_score = [item[0] for item in sample.score_gained[:-1]]
            delta_score = self.pad_zero(delta_score, self.num_ice)
            delta_scores.append(delta_score)

            label = [eid2idx[eid] for eid in sample.best_permutations]
            label = self.pad_minus_one(label, self.num_ice)
            labels.append(label)

        questions_tensor, questions_attn_mask = self.pad(questions_tensor)  # (B, T)
        ctxs_tensor, ctxs_attn_mask = self.pad(ctxs_tensor)  # (B*L', T)

        labels = torch.tensor(labels)  # (B, L)
        ctx_indices = torch.tensor(ctx_indices)  # (B, L)
        delta_scores = torch.tensor(delta_scores)  # (B, L)

        batch = {
            "questions_tensor": questions_tensor,  # (B, T)
            "questions_attn_mask": questions_attn_mask,  # (B, T)
            "ctxs_tensor": ctxs_tensor,  # (B*L', T)
            "ctxs_attn_mask": ctxs_attn_mask,  # (B*L', T)
            "ctx_indices": ctx_indices,  # (B, L)
            "labels": labels,  # (B, L)
            "delta_scores": delta_scores,  # (B, L)
        }
        return BatchEncoding(data=batch)