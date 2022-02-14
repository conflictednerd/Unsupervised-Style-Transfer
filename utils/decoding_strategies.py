from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOS_token_id = 5  # for snapp dataset


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def get_memory(input_: Union[Dict, List], emb_layer, encoder):
    input_ = input_['text'] if isinstance(input_, dict) else input_
    input_text = torch.tensor(
        input_ + [EOS_token_id], dtype=torch.int64).unsqueeze(0).to(DEVICE)  # add batch dimension
    src_padding_mask = torch.tensor(
        [False] * (len(input_text[0]))).unsqueeze(0).to(DEVICE)
    embedded_inputs = emb_layer(input_text)
    memory = encoder(embedded_inputs, src_padding_mask)
    return memory, src_padding_mask


@torch.no_grad()
def decode_greedy(
    desired_label,
    tokenizer, emb_layer: nn.Module, encoder: nn.Module, decoder: nn.Module,
    input_: dict = None, memory: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None,
        max_len: int = 128) -> List[torch.Tensor]:
    '''
    Either memory or input_text should be given
    Output is a list of integers (each stored as a 1dim tensor)
    '''
    memory, memory_key_padding_mask = (memory, memory_key_padding_mask) if input_ is None else get_memory(
        input_, emb_layer, encoder)
    assert memory is not None
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = memory_key_padding_mask.to(DEVICE)

    generated_output = []
    tgt_ = emb_layer(torch.tensor([tokenizer.encoder.word_vocab[f'__style{desired_label+1}']]
                                  ).unsqueeze(-1).to(DEVICE))

    while len(generated_output) < max_len:
        decoder_output = decoder(
            tgt=tgt_, memory=memory, memory_key_padding_mask=memory_key_padding_mask)

        # batch size is assumed to be 1
        next_vocab = torch.argmax(decoder_output[:, -1, :])
        generated_output.append(next_vocab)
        if next_vocab == EOS_token_id:
            break
        next_word_emb = emb_layer(torch.tensor(
            [next_vocab]).unsqueeze(0).to(DEVICE))
        tgt_ = torch.cat([tgt_, next_word_emb], axis=1).to(DEVICE)

    return generated_output


@torch.no_grad()
def decode_beam(
        desired_label: int,
        tokenizer, emb_layer: nn.Module, encoder: nn.Module, decoder: nn.Module,
        input_: dict = None, memory: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None,
        K: int = 10, max_len: int = 128) -> List[torch.Tensor]:

    memory, memory_key_padding_mask = (memory, memory_key_padding_mask) if input_ is None else get_memory(
        input_, emb_layer, encoder)
    assert memory is not None
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = memory_key_padding_mask.to(DEVICE)

    def next_most_probables(tgt, memory):
        dec_out = decoder(tgt=tgt, memory=memory,
                          memory_key_padding_mask=memory_key_padding_mask)
        dec_out = F.softmax(dec_out[0, -1, :], dim=-1)
        next_tokens = torch.topk(dec_out, K)
        outputs = []
        for token in range(K):
            outputs.append([next_tokens.indices[token],
                            next_tokens.values[token]])
        return outputs

    target_sequences = [
        [[tokenizer.encoder.word_vocab[f'__style{desired_label+1}']], 0.0] * K]

    min_active_len = 1
    while min_active_len < max_len:
        new_targets = []
        for target in target_sequences:
            target_list = target[0]
            target_score = target[1]

            if target_list[-1] == EOS_token_id or len(target_list) >= max_len:
                new_targets.append(target)
                continue

            tgt_ = emb_layer(torch.tensor(target_list).unsqueeze(0).to(DEVICE))

            next_candidates = next_most_probables(tgt_, memory)
            # mult of probabilities is the sum of their logs
            for candidate in next_candidates:
                new_targets.append(
                    [target_list + [candidate[0].tolist()], target_score - np.log(candidate[1].tolist())])

        target_sequences = sorted(new_targets, key=lambda tup: tup[1])[:K]
        all_ended = True
        for seq in target_sequences:
            if seq[0][-1] != EOS_token_id:
                all_ended = False
                break
        if all_ended:
            break
        min_active_len = min([len(seq[0])
                              for seq in target_sequences if seq[0][-1] != EOS_token_id])

    return target_sequences[0][0]


@torch.no_grad()
def decode_sampling(
    desired_label: int,
    tokenizer, emb_layer: nn.Module, encoder: nn.Module, decoder: nn.Module,
    top_k: int = 10, top_p: float = 0.0, temperature: float = 0.2,
    input_: dict = None, memory: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None,
        max_len: int = 128) -> List[torch.Tensor]:
    '''
    top_k: keep only top k tokens with highest probability (top-k filtering)
    top_p: top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
    '''
    memory, memory_key_padding_mask = (memory, memory_key_padding_mask) if input_ is None else get_memory(
        input_, emb_layer, encoder)
    assert memory is not None
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = memory_key_padding_mask.to(DEVICE)

    generated_output = []
    tgt_ = emb_layer(torch.tensor(
        [tokenizer.encoder.word_vocab[f'__style{desired_label+1}']]).unsqueeze(-1).to(
        DEVICE))

    while len(generated_output) < max_len:
        logits = decoder(
            tgt=tgt_, memory=memory, memory_key_padding_mask=memory_key_padding_mask)[0, -1, :]
        filtered_logits = top_k_top_p_filtering(
            logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)

        generated_output.append(next_token)
        if next_token == EOS_token_id:
            break
        next_word_emb = emb_layer(torch.tensor(
            [next_token]).unsqueeze(0).to(DEVICE))
        tgt_ = torch.cat([tgt_, next_word_emb], axis=1).to(DEVICE)

    return generated_output


@torch.no_grad()
def scheduled_sampling(emb_layer: nn.Module, decoder: nn.Module,
                       memory: torch.Tensor, style_embedding: torch.Tensor, teacher_targets: torch.Tensor,
                       memory_key_padding_mask: torch.Tensor, tgt_mask: torch.Tensor,
                       temperature: float = 0.6, top_k: int = 10, top_p: float = 0.0, iters: int = 5, p=0.25,
                       ) -> torch.Tensor:
    '''
    All tensors are assumed to be on the desired device and models should be on eval mode
    '''
    new_targets = teacher_targets
    for _ in range(iters):
        logits = decoder(tgt=new_targets, memory=memory, memory_key_padding_mask=memory_key_padding_mask,
                         tgt_mask=tgt_mask, tgt_key_padding_mask=memory_key_padding_mask)  # BxTxV
        filtered_logits = top_k_top_p_filtering(
            logits / temperature, top_k=top_k, top_p=top_p)  # BxTxV
        probs = F.softmax(filtered_logits, dim=-1)
        B, T, V = probs.shape
        new_tgt_tokens = torch.multinomial(
            probs.reshape(-1, V), num_samples=1).reshape(B, T)
        new_targets = emb_layer(new_tgt_tokens)  # BxTxh
        new_targets = torch.roll(new_targets, shifts=1, dims=1)
        new_targets[:, 0, :] = style_embedding.squeeze(1)
    idx = torch.empty(B, T, 1, dtype=torch.bool).bernoulli_(p)
    return torch.where(idx, new_targets, teacher_targets)
