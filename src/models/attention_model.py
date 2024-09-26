import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class AttentionModelConfig(PretrainedConfig):
    model_type = "AttentionModel"

    def __init__(
            self,
            q_model_name=None,
            ctx_model_name=None,
            q_no_grad=True,
            ctx_no_grad=True,
            norm_embed=True,
            nhead=8,
            num_layers=2,
            max_num_ice=16,
            nonlinear=False,
            causal_mask=True,
            class_ld=1.,
            score_ld=1.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_model_name = q_model_name
        self.ctx_model_name = ctx_model_name
        self.q_no_grad = q_no_grad
        self.ctx_no_grad = ctx_no_grad
        self.norm_embed = norm_embed
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_num_ice = max_num_ice
        self.nonlinear = nonlinear
        self.causal_mask = causal_mask
        self.class_ld = class_ld
        self.score_ld = score_ld


class AutoregressiveTransformerDecoder(nn.Module):
    def __init__(self, hidden_size, nhead=8, num_layers=2):
        super(AutoregressiveTransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead, dropout=0, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None):
        return self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)


class AttentionModel(PreTrainedModel):
    config_class = AttentionModelConfig

    def __init__(self, config):
        super(AttentionModel, self).__init__(config)
        assert config.q_model_name is not None or config.ctx_model_name is not None

        if config.q_model_name is not None:
            self.question_model = AutoModel.from_pretrained(config.q_model_name)
        else:
            self.question_model = None
        if config.ctx_model_name is not None:
            self.ctx_model = AutoModel.from_pretrained(config.ctx_model_name)
        else:
            self.ctx_model = None

        # share q and ctx model if one of them is None
        if self.question_model is None and self.ctx_model is not None:
            self.question_model = self.ctx_model
            logging.info("Sharing ctx_model with question_model")
        if self.question_model is not None and self.ctx_model is None:
            self.ctx_model = self.question_model
            logging.info("Sharing question_model with ctx_model")

        self.q_no_grad = config.q_no_grad
        self.ctx_no_grad = config.ctx_no_grad
        self.norm_embed = config.norm_embed
        self.max_num_ice = config.max_num_ice
        self.causal_mask = config.causal_mask
        self.class_ld = config.class_ld
        self.score_ld = config.score_ld

        hidden_size = self.question_model.config.hidden_size
        self.attention_model = AutoregressiveTransformerDecoder(hidden_size, config.nhead, config.num_layers)
        if config.nonlinear:
            self.learnable_question_query = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(self.max_num_ice)
            ])
        else:
            self.learnable_question_query = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size)
                for _ in range(self.max_num_ice)
            ])

        self.class_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, input_ids, attention_mask, encode_ctx=False, **kwargs):
        if encode_ctx:
            if self.ctx_no_grad:
                with torch.no_grad():
                    enc_emb = self.ctx_model(input_ids, attention_mask)
            else:
                enc_emb = self.ctx_model(input_ids, attention_mask)
        else:
            if self.q_no_grad:
                with torch.no_grad():
                    enc_emb = self.question_model(input_ids, attention_mask)
            else:
                enc_emb = self.question_model(input_ids, attention_mask)

        enc_emb = self.mean_pooling(enc_emb, attention_mask)
        if self.norm_embed:
            enc_emb = enc_emb / enc_emb.norm(p=2, dim=-1, keepdim=True)
        return enc_emb

    def generate_tgt_mask(self, tgt_sz):
        mask = (torch.triu(torch.ones(tgt_sz, tgt_sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def predict(
            self,
            q_vectors: T,  # (B, H)
            ctx_vectors: T,  # (B*L', H)
            ctx_indices: T,  # (B, L)
    ) -> T:
        learnable_q_vectors = torch.stack([self.learnable_question_query[i](q_vectors) for i in range(self.max_num_ice)])  # (L, B, H)
        learnable_q_vectors = learnable_q_vectors.permute(1, 0, 2)  # (B, L, H)

        unpacked_ctx_vectors = torch.stack([ctx_vectors[indices] for indices in ctx_indices])  # (B, L, H)

        # tgt_mask generation
        tgt_mask = None
        if self.causal_mask:
            tgt_mask = self.generate_tgt_mask(self.max_num_ice).to(q_vectors.device)  # (L, L)

        # decoding
        pred_ctx_vectors = self.attention_model(
            learnable_q_vectors,
            unpacked_ctx_vectors,
            tgt_mask=tgt_mask,
        )  # (B, L, H)

        pred_class_logits = self.class_predictor(pred_ctx_vectors)  # (B, L, 2)
        pred_class_probs = F.softmax(pred_class_logits, dim=-1)  # (B, L, 2)
        pred_scores = self.score_predictor(pred_ctx_vectors).squeeze(-1)  # (B, L)
        return pred_ctx_vectors, pred_class_probs, pred_scores

    def pad_minus_one(self, input_list, target_length):
        num_pads_needed = target_length - len(input_list)
        return input_list + [-1] * num_pads_needed
    
    def pred_labels(
            self,
            pred_ctx_vectors: T,  # (B, L, H)
            pred_class_probs: T,  # (B, L, 2)
            ctx_vectors: T,  # (B*L', H)
            ctx_indices: T,  # (B, L)
    ) -> T:
        pred_class_labels = (pred_class_probs[:, :, 1] > 0.2)  # (B, L)

        B = pred_ctx_vectors.shape[0]
        pred_labels = []
        for i in range(B):
            curr_pred_class_labels = pred_class_labels[i]  # (L,)
            curr_pred_ctx_vectors = pred_ctx_vectors[i][curr_pred_class_labels]  # (K_i, H)
            curr_ctx_vectors = ctx_vectors[ctx_indices[i]]  # (L, H)

            local_global_index_map = {local_index: global_index for local_index, global_index in enumerate(ctx_indices[i])}

            if curr_pred_ctx_vectors.size(0) > 0:
                curr_pred_ctx_vectors = F.normalize(curr_pred_ctx_vectors, dim=1)  # (K_i, H)
                curr_ctx_vectors = F.normalize(curr_ctx_vectors, dim=1)  # (L, H)
                cos_sim_matrix = torch.matmul(curr_pred_ctx_vectors, curr_ctx_vectors.T)  # (K_i, L)
                sorted_indices = torch.argsort(cos_sim_matrix, dim=-1, descending=True)  # (K_i, L)
                curr_labels = list()
                for row in sorted_indices:
                    for index in row:
                        if index.item() not in curr_labels:
                            curr_labels.append(index.item())
                            break
                curr_labels = [local_global_index_map[local_index] for local_index in curr_labels]
            else:
                curr_labels = []

            curr_labels = self.pad_minus_one(curr_labels, self.max_num_ice)
            pred_labels.append(curr_labels)

        pred_labels = torch.tensor(pred_labels, device=pred_ctx_vectors.device)  # (B, L)
        return pred_labels
    
    def contrastive_loss(self, preds, labels, temperature=1.0):
        preds = F.normalize(preds, dim=1)  # (K_i, H)
        labels = F.normalize(labels, dim=1)  # (L, H)
        scores = torch.matmul(preds, labels.T) / temperature  # (K_i, L)
        softmax_scores = F.log_softmax(scores)
        pos_mask = torch.triu(torch.ones_like(softmax_scores).T).T.bool()
        loss = - torch.masked_select(softmax_scores, pos_mask).mean()
        return loss

    def calculate_loss(
            self,
            pred_ctx_vectors: T,  # (B, L, H)
            pred_class_probs: T,  # (B, L, 2)
            pred_scores: T, # (B, L)
            ctx_vectors: T,  # (B*L', H)
            labels: T,  # (B, L), minus one padded
            delta_scores: T,  # (B, L), zero padded
    ) -> T:
        B = pred_ctx_vectors.size(0)
        retrieval_loss = 0

        # cosine similarity loss
        for i in range(B):
            curr_labels = labels[i]  # (L,)
            curr_pred_ctx_vectors = pred_ctx_vectors[i]  # (L, H)
            label_mask = (curr_labels != -1)
            filtered_curr_labels = torch.masked_select(curr_labels, label_mask)  # (K_i,)
            curr_pred_ctx_vectors = curr_pred_ctx_vectors[:len(filtered_curr_labels)]  # (K_i, H)
            curr_label_ctx_vectors = ctx_vectors[filtered_curr_labels]  # (K_i, H)
            cos_sim = F.cosine_similarity(curr_pred_ctx_vectors, curr_label_ctx_vectors).mean()
            retrieval_loss += (1 - cos_sim)

        # contrastive loss
        # for i in range(B):
        #     curr_labels = labels[i]  # (L,)
        #     curr_pred_ctx_vectors = pred_ctx_vectors[i]  # (L, H)
        #     filtered_curr_labels = [idx for idx in curr_labels.tolist() if idx != -1]  # (K_i,)
        #     left_curr_labels = [idx for idx in range(len(ctx_vectors)) if idx not in filtered_curr_labels]  # (L - K_i,)
        #     curr_label_ctx_vectors = torch.cat([ctx_vectors[filtered_curr_labels], ctx_vectors[left_curr_labels]], dim=0)  # (L, H)
        #     curr_pred_ctx_vectors = curr_pred_ctx_vectors[:len(filtered_curr_labels)]  # (K_i, H)
        #     retrieval_loss += self.contrastive_loss(curr_pred_ctx_vectors, curr_label_ctx_vectors)

        # class prediciton loss
        class_labels = (labels != -1)  # (B, L)
        one_hot_class_labels = F.one_hot(class_labels.long(), num_classes=2).float()  # (B, L, 2)
        class_loss = F.binary_cross_entropy(pred_class_probs, one_hot_class_labels)

        # score prediction loss
        score_loss = 0
        # nonzero_mask = (delta_scores != 0.0)
        # score_loss = F.mse_loss(pred_scores[nonzero_mask], delta_scores[nonzero_mask])

        # total loss
        loss = retrieval_loss + self.class_ld * class_loss + self.score_ld * score_loss
        return loss
    
    def forward(
            self,
            questions_tensor: T,  # (B, T)
            questions_attn_mask: T,  # (B, T)
            ctxs_tensor: T,  # (B*L', T)
            ctxs_attn_mask: T,  # (B*L', T)
            ctx_indices: T,  # (B, L)
            labels: T,  # (B, L), minus one padded
            delta_scores: T, # (B, L), zero padded
    ) -> Dict:
        q_vectors = self.encode(
            questions_tensor,
            questions_attn_mask,
            encode_ctx=False,
        )  # (B, H)
        ctx_vectors = self.encode(
            ctxs_tensor,
            ctxs_attn_mask,
            encode_ctx=True,
        )  # (B*L', H)
        pred_ctx_vectors, pred_class_probs, pred_scores = self.predict(
            q_vectors,
            ctx_vectors,
            ctx_indices,
        )  # (B, L, H), (B, L, 2)
        pred_labels = self.pred_labels(
            pred_ctx_vectors,
            pred_class_probs,
            ctx_vectors,
            ctx_indices,
        )  # (B, L), minus one padded
        loss = self.calculate_loss(
            pred_ctx_vectors,
            pred_class_probs,
            pred_scores,
            ctx_vectors,
            labels,
            delta_scores,
        )
        return {'loss': loss, 'logits': pred_labels}