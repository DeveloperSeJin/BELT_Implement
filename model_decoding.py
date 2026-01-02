import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math
import numpy as np
from module.VQVAE import VectorQuantizer
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def cos_sim(A, B, dim= -1, eps= 1e-4):

    # ℓ2-norm
    norm_A = F.normalize(A, p=2, dim=dim, eps=eps)
    norm_B = F.normalize(B, p=2, dim=dim, eps=eps)

    # sim = torch.matmul(norm_A, norm_B.transpose(-1, -2))
    return norm_A @ norm_B.transpose(-1, -2)

""" main architecture for open vocabulary EEG-To-Text decoding"""
class ContrastiveLatentLoss(nn.Module):
    def __init__(self, gamma=0.07):
        """
        Contrastive loss for aligning EEG Condition with Text Latent in Latent Diffusion Model (LDM).
        """
        super(ContrastiveLatentLoss, self).__init__()
        self.gamma = gamma
        
        hidden_dim = 512 
        projection_dim = 256

        self.text_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.eeg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )



    def forward(self, text_latent, eeg_condition, mask=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (text_latent, eeg_condition), self.parameters(), True
        )
    
    def _forward(self, text_latent, eeg_condition, mask=None):
        """
        Args:
            text_latent (torch.Tensor): Latent representation of text (batch_size, latent_dim)
            eeg_condition (torch.Tensor): EEG-based Condition Vector (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Contrastive loss value
        """

        batch_size, _, _ = text_latent.shape
        text_sent_emb = text_latent.mean(dim=1)  # (batch_size, hidden_dim)
        eeg_sent_emb = eeg_condition.mean(dim=1) # (batch_size, hidden_dim)

        text_embedding = self.text_projection(text_sent_emb)
        eeg_embedding = self.eeg_projection(eeg_sent_emb)


        # (2) 각 문장 내 단어 간 similarity 계산
        # similarity = cos_sim(text_sent_emb, eeg_sent_emb) / self.gamma
        similarity = cos_sim(text_embedding, eeg_embedding) / self.gamma

        # (3) 정답 labels 생성 (각 단어가 동일 위치 단어와 매칭되도록)
        labels = torch.arange(batch_size).to(eeg_sent_emb.device)
        # labels shape: [batch_size, seq_len] (정답은 항상 [0,1,...,seq_len-1])

        # (5) Contrastive loss 계산 (단어 수준 Contrastive loss)
        loss = F.cross_entropy(similarity, labels)

        return loss


class BELT(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, additional_encoder_nhead=8,
                 additional_encoder_dim_feedforward=2048):
        super().__init__()

        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,
                                                                   dim_feedforward=additional_encoder_dim_feedforward,
                                                                   batch_first=True)
        self.contrastive_learning = ContrastiveLatentLoss(gamma=0.075)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)
        
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)
        self.vq_layer = VectorQuantizer(
            num_embeddings=2048,
            embedding_dim=512,
            beta =self.beta)
        self.d_conformer = {}

    def text_embedding(self, x):
        embeded_context = self.pretrained_LM.model.shared(x)
        condition = self.fc2(embeded_context)

        return condition


    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, dummy_decoder_inputs_ids,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=None,
            assistant_model=None,
            streamer=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
            **kwargs,
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)

        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
            decoder_input_ids = dummy_decoder_inputs_ids,
            **kwargs,)
        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, context, target_ids_batch):

        condition = self.text_embedding(context)

        discrete_embedding = self.d_conformer(input_embeddings_batch)

        z_q, vq_loss = self.vq_layer(discrete_embedding)
        # print(f'forward:{input_embeddings_batch.shape,input_masks_batch.shape,input_masks_invert.shape,target_ids_batch_converted.shape,encoded_embedding.shape}')
        out = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch,
                                labels=target_ids_batch_converted)
        
        # contrastive learning loss
        valid_mask = (target_ids_batch != -100)
        valid_mask_float = valid_mask.float()
        loss = loss + self.beta * self.contrastive_learning(z, condition, valid_mask_float)
        return out