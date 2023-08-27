import os
import copy
import glob
import json
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from transformers import set_seed
from torch.utils.data import DataLoader
from src.utils.misc import partial
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.models.attention_model import AttentionModel

logger = logging.getLogger(__name__)


class AttentionRetriever:
    def __init__(self, cfg, accelerator=None) -> None:
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path is not None:
            self.model = AttentionModel.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = AttentionModel(model_config)

        self.model = self.model.to(self.cuda_device)
        self.model.eval()

        self.output_file = cfg.output_file
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"
        self.accelerator = accelerator
        self.index = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        res_list = self.forward(dataloader, encode_ctx=True)
        self.ctx_res_list = res_list

        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.concatenate([normalize(np.expand_dims(res['embed'], axis=0)) for res in res_list])
        norm_embed_list = embed_list  # (D, H)
        index.add_with_ids(norm_embed_list, id_list)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        return index

    def forward(self, dataloader, **kwargs):
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.model.encode(**entry, **kwargs)
            res = res.cpu().detach().numpy()
            res_list.extend([{"embed": r, "attn_mask": at, "metadata": m} for r, at, m in zip(res, entry["attention_mask"], metadata)])
        return res_list

    def shard_list(self, input_list, num_shards):
        length = len(input_list) // num_shards + 1
        shards = []
        for i in range(num_shards):
            shards.append(input_list[i*length:(i+1)*length])
        return shards

    def find(self):
        res_list = self.forward(self.dataloader, encode_ctx=False)
        for i, res in enumerate(res_list):
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]

        func = partial(greedy_nn_retrieval,
                        model=self.model,
                        ctx_entries=self.ctx_res_list,
                        ctx_index=self.index,
                        num_ice=self.num_ice,
                        device=self.accelerator.device)
        data = []
        res_list = self.shard_list(res_list, self.accelerator.num_processes)[self.accelerator.process_index]
        if self.accelerator.is_main_process:
            res_list = tqdm.tqdm(res_list)
        for i, res in enumerate(res_list):
            data.append(func(target_entry=res))

        with open(f"{self.output_file}tmp_{self.accelerator.device}.bin", "w") as f:
            json.dump(data, f)

    def aggregate_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with open(path) as f:
                data.extend(json.load(f))

        with open(self.output_file, "w") as f:
            json.dump(data, f)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data
    

def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def greedy_nn_retrieval(model,
                        ctx_entries,
                        ctx_index,
                        target_entry,
                        num_ice,
                        device):

    def get_id_index_map(entries):
        res_id_to_index = dict()
        for i, res in enumerate(entries):
            res_id_to_index[res['metadata']['id']] = i
        return res_id_to_index

    ctx_index = copy.deepcopy(ctx_index)

    # First-stage retrieval
    question_embed = np.expand_dims(target_entry['embed'], axis=0)  # [1, H]
    norm_question_embed = normalize(question_embed)  # (1, H)
    D, I = ctx_index.search(norm_question_embed, num_ice)
    ctx_idx_map = get_id_index_map(ctx_entries)
    near_res_list = [ctx_entries[ctx_idx_map[near_id]] for near_id in I[0]]

    # Build new faiss index
    id_list = np.array([res['metadata']['id'] for res in near_res_list])
    embed_list = np.stack([res['embed'] for res in near_res_list])
    norm_embed_list = np.concatenate([normalize(np.expand_dims(res['embed'], axis=0)) for res in near_res_list])
    near_index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    near_index.add_with_ids(norm_embed_list, id_list)

    # Initialize embedding for search
    q_vectors = torch.from_numpy(question_embed).to(device)  # (1, H)
    ctx_vectors = torch.from_numpy(embed_list).to(device)  # (num_ice, H)
    ctx_indices = torch.tensor([[i for i in range(num_ice)]]).to(device)  # (1, num_ice)

    pred_ctx_vectors, pred_class_probs, _ = model.predict(q_vectors, ctx_vectors, ctx_indices)
    pred_class_labels = (pred_class_probs[0, :, 1] > 0.2)  # (filtered_num_ice, )
    last_true_index = torch.where(pred_class_labels)[0][-1]
    pred_class_labels[:last_true_index + 1] = True
    filtered_pred_ctx_vectors = pred_ctx_vectors[0][pred_class_labels] # (filtered_num_ice, H)

    ctxs = []
    cos_sim_list = []
    for i in range(len(filtered_pred_ctx_vectors)):
        curr_embed = filtered_pred_ctx_vectors[i].unsqueeze(0).detach().cpu().numpy()
        norm_curr_embed = normalize(curr_embed)  # (1, H)
        D, I = near_index.search(norm_curr_embed, num_ice)

        Ds = []
        Is = []
        for d, i in zip(D, I):
            Ds.extend(d)
            Is.extend(i)

        sorted_di = sorted(zip(Ds, Is), key=lambda x: x[0], reverse=True)
        for cos_sim, near_id in sorted_di:
            if (near_id not in ctxs) and (near_id != -1):
                ctxs.append(int(near_id))
                near_index.remove_ids(np.array([near_id]))
                cos_sim_list.append(cos_sim)
                break

    entry = target_entry['entry']
    entry['ctxs'] = ctxs
    entry['ctxs_candidates'] = [[ctx] for ctx in ctxs]
    return entry


@hydra.main(config_path="configs", config_name="attention_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)

    torch.multiprocessing.set_start_method('spawn')
    accelerator = Accelerator()

    attention_retriever = AttentionRetriever(cfg, accelerator)
    attention_retriever.find()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        attention_retriever.aggregate_results()


if __name__ == "__main__":
    main()
