from copy import deepcopy
from datasets import Dataset
import pandas as pd

from src.dataset_readers.base_dsr import encode_field
from src.dataset_readers.scoring_dsr import ScorerDatasetReader
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper


class AdvancedScorerDatasetReader(ScorerDatasetReader):
    def init_dataset(self, task_name, field, dataset_path, dataset_split, ds_size, truncation=False):
        def get_instance(idx, entry):
            # todo, note here we may overwrite original idx field (if exists)
            entry['idx'] = idx  # unique id of original instances, used for grouping instances
            candidates = entry["candidates"]
            best_permutations = entry["best_permutations"]
            useless_candidates = entry["useless_candidates"]

            sampled = deepcopy(candidates)
            for elem in best_permutations:
                sampled.remove(elem)
            for elem in useless_candidates:
                sampled.remove(elem)

            ctxs_cand = [deepcopy(best_permutations)]
            ctxs_cand += [best_permutations + [elem] for elem in sampled]

            for ctxs in ctxs_cand:
                example = deepcopy(entry)
                example["ctxs"] = ctxs
                yield example

        def get_dataset(data):
            for idx, entry in enumerate(data):
                yield from get_instance(idx, entry)

        self.dataset_wrapper = get_dataset_wrapper(task_name, dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)
        df = pd.DataFrame(list(get_dataset(self.dataset_wrapper.dataset)))
        self.dataset_wrapper.dataset = Dataset.from_pandas(df)
        self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, truncation)

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        prompt_len = self.encoded_dataset[index]['metadata']['len']
        prompt = self.encoded_dataset[index]['metadata']['text']

        answer = self.dataset_wrapper.get_field(entry=entry, field="a")
        tokenized_labels = self.tokenizer.encode_plus(
            answer,
            truncation=False,
            add_special_tokens=False,
            return_tensors='pt',
        )
        answer_len = tokenized_labels.attention_mask.shape[1]

        ice_prompt, trunc_ice_prompts_list = self.get_ice_prompt(entry, prompt_len+answer_len)
        prompt = prompt.replace("{ice_prompt}", ice_prompt)
        entry['prompt'] = prompt + answer
        entry['ice_prompts_list'] = trunc_ice_prompts_list

        tokenized_example = self.tokenizer.encode_plus(
            entry['prompt'],
            truncation=False,
            return_tensors='pt',
            add_special_tokens=False,
        )

        return {
            'input_ids': tokenized_example.input_ids[0],
            'labels': tokenized_labels.attention_mask[0],
            "metadata": entry
        }
