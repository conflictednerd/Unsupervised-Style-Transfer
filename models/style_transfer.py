import json
import os
from statistics import mode

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from snapp_dataset import create_snapp_dataset_from_path, data_collator_snapp
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.decoder import Decoder
from models.discriminator import CNNDiscriminator
from models.encoder import EmbeddingLayer, Encoder


class StyleTransferModel():
    def __init__(self, args) -> None:
        self.args = args
        self.log_dir = args.log_dir + '/' + args.exp_name
        self.models_dir = args.models_dir
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = Tokenizer(
            args.tokenizer_name, load=True, models_dir=args.models_dir)
        self.emb_layer = EmbeddingLayer(
            self.tokenizer.vocab_size, args.d_model, batch_first=True).to(self.device)
        self.encoder = Encoder(args.d_model, args.encoder_nhead, args.encoder_ff,
                               args.encoder_layers, self.tokenizer.vocab_size,
                               batch_first=True, device=self.device).to(self.device)
        self.decoder = Decoder(args.d_model, args.decoder_nhead, args.decoder_ff,
                               args.decoder_layers, self.tokenizer.vocab_size,
                               batch_first=True, device=self.device).to(self.device)
        self.disc = CNNDiscriminator(in_channels=1, out_channels=args.disc_channels,
                                     kernel_sizes=args.disc_kernels, hidden_size=args.d_model,
                                     num_classes=args.num_styles).to(self.device)
        self.logger = SummaryWriter(self.log_dir)

        self.encoder_optim = optim.Adam(
            list(self.emb_layer.parameters()) +
            list(self.encoder.parameters()),
            lr=args.encoder_lr)
        self.decoder_optim = optim.Adam(
            self.decoder.parameters(),
            lr=args.decoder_lr)

        self.disc_optim = optim.Adam(self.disc.parameters(), lr=args.disc_lr)

        self.rec_loss_criterion = torch.nn.CrossEntropyLoss(
            reduction='mean', ignore_index=0)
        self.adv_loss_criterion = torch.nn.CrossEntropyLoss(
            reduction='mean', label_smoothing=self.args.label_smoothing)

        if args.load_model:
            self.load()

        # train/dev/test loaders
        self.train_dataset = create_snapp_dataset_from_path(
            "data/snappfood/", "train.csv", self.tokenizer)
        self.dev_dataset = create_snapp_dataset_from_path(
            "data/snappfood/", "dev.csv", self.tokenizer)
        self.test_dataset = create_snapp_dataset_from_path(
            "data/snappfood/", "test.csv", self.tokenizer)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_snapp,
            num_workers=args.num_workers)
        self.dev_loader = DataLoader(
            self.dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_snapp,
            num_workers=args.num_workers)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator_snapp,
            num_workers=args.num_workers)

        print(f'''Embedding layer parameters: {sum(p.numel() for p in self.emb_layer.parameters() if p.requires_grad)}
                Encoder parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}
                Decoder parameters: {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}
                Discriminator parameters: {sum(p.numel() for p in self.disc.parameters() if p.requires_grad)}
                Total number of parameters: {sum(p.numel() for p in list(self.emb_layer.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.disc.parameters()) if p.requires_grad)}''')

        self.log()

    def save(self, ) -> None:
        # save emb_layer, encoder, decoder, disc, their optims
        model_dict = {
            'emb_layer': self.emb_layer,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'disc': self.disc,
            'encoder_optim': self.encoder_optim,
            'decoder_optim': self.decoder_optim,
            'disc_optim': self.disc_optim,
        }
        torch.save(model_dict, os.path.join(
            self.models_dir, self.args.exp_name + '.pt'))
        print('Model saved')

    def load(self, ) -> None:
        model_dict = torch.load(os.path.join(
            self.models_dir, self.args.exp_name + '.pt'), map_location=self.device)
        self.emb_layer = model_dict['emb_layer']
        self.encoder = model_dict['encoder']
        self.decoder = model_dict['decoder']
        self.disc = model_dict['disc']
        self.encoder_optim = model_dict['encoder_optim']
        self.decoder_optim = model_dict['decoder_optim']
        self.disc_optim = model_dict['disc_optim']

    def train(self, ) -> None:
        for epoch in range(self.args.epochs):
            print(f'[Epoch: {epoch+1}/{self.args.epochs}]')
            train_rec_loss, train_disc_loss = self.run_epoch(
                epoch)
            print(
                f'Reconstruction loss: {train_rec_loss:.3f}, Discriminator loss: {train_disc_loss:.3f}')
            self.save()
            # TODO: run evaluation on dev set
            self.evaluate()

    def run_epoch(self, epoch):
        total_rec_loss, total_disc_loss, total_num_samples = 0, 0, 0
        running_rec_loss, running_disc_loss, running_num_samples, running_disc_correct_preds = 0, 0, 0, 0
        for idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            text_batch, labels, src_key_padding_mask, tgt_mask = batch
            text_batch = text_batch.to(self.device)
            labels = labels.to(self.device)
            src_key_padding_mask = src_key_padding_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)

            self.train_mode()
            embedded_inputs = self.emb_layer(text_batch)
            encoder_output = self.encoder(
                embedded_inputs, src_key_padding_mask)
            encoder_output_detached = encoder_output.detach()  # TODO: add noise
            decoder_tgt = torch.roll(encoder_output, shifts=1, dims=1)
            style_embedding = self.emb_layer(torch.tensor(
                [self.tokenizer.encoder.word_vocab[f'__style{label+1}'] for label in labels]).unsqueeze(-1).to(
                self.device))
            decoder_tgt[:, 0, :] = style_embedding.squeeze(1)
            decoder_output = self.decoder(tgt=decoder_tgt, memory=encoder_output,
                                          memory_key_padding_mask=src_key_padding_mask,
                                          tgt_mask=tgt_mask, tgt_key_padding_mask=src_key_padding_mask)
            rec_loss = self.rec_loss_criterion(
                decoder_output.flatten(0, 1), text_batch.flatten())
            disc_logits = self.disc(encoder_output_detached)
            disc_loss = self.adv_loss_criterion(disc_logits, labels)
            enc_loss = - \
                self.adv_loss_criterion(self.disc(encoder_output), labels)

            if self.update_ae(epoch, idx):
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                loss = rec_loss + self.args.lambda_gan * \
                    enc_loss if self.update_disc(epoch, idx) else rec_loss
                loss.backward()
                self.decoder_optim.step()
                self.encoder_optim.step()
            if self.update_disc(epoch, idx):
                self.disc_optim.zero_grad()
                disc_loss.backward()
                self.disc_optim.step()

            # bookkeeping stats
            batch_size = len(labels)
            running_num_samples += batch_size
            running_rec_loss += rec_loss.item() * batch_size
            running_disc_loss += disc_loss.item() * batch_size
            running_disc_correct_preds += torch.count_nonzero(
                torch.argmax(disc_logits, dim=-1) == labels)

            # logging every 100 minibatch
            if (idx + 1) % 100 == 0:
                total_num_samples += running_num_samples
                total_rec_loss += running_rec_loss
                total_disc_loss += running_disc_loss
                disc_acc = running_disc_correct_preds / running_num_samples
                global_step = epoch * \
                    (len(self.train_loader) // 100) + (idx + 1) // 100 - 1
                self.logger.add_scalar(
                    'Train/Loss/reconstruction', running_rec_loss / running_num_samples, global_step=global_step)
                self.logger.add_scalar(
                    'Train/Loss/discriminator', running_disc_loss / running_num_samples, global_step=global_step)
                self.logger.add_scalar(
                    'Train/Accuracy/discriminator', disc_acc, global_step=global_step)
                running_num_samples, running_rec_loss, running_disc_loss, running_disc_correct_preds = 0, 0, 0, 0

            # if (idx + 1) % 900 == 0:
            #     self.evaluate()

            if (idx + 1) % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_rec_loss, total_disc_loss

    # mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    # mask = mask.float().masked_fill(mask == 0, float(
    #     '-inf')).masked_fill(mask == 1, float(0.0))

    def generate_sampling(self, desired_label, top_k: int = 10, top_p: float = 0.0, temperature=0.2, input_=None, memory=None, memory_key_padding_mask=None, max_len=128):
        '''
        top_k: keep only top k tokens with highest probability (top-k filtering)
        top_p: top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        '''
        assert not (input_ is None and memory is None)
        EOS_token_id = 5  # for snapp dataset
        self.eval_mode()
        if input_ is not None:
            input_text = torch.tensor(
                input_['text'] + [EOS_token_id], dtype=torch.int64).unsqueeze(0).to(self.device)
            src_padding_mask = torch.tensor(
                [False] * (len(input_text[0]))).unsqueeze(0).to(self.device)
            embedded_inputs = self.emb_layer(input_text)
            memory = self.encoder(
                embedded_inputs, src_padding_mask)

        generated_output = []
        with torch.no_grad():
            tgt_ = self.emb_layer(torch.tensor(
                [self.tokenizer.encoder.word_vocab[f'__style{desired_label+1}']]).unsqueeze(-1).to(
                self.device))

            while len(generated_output) < max_len:
                # should we use 'decoder_output'? let's talk about it
                logits = self.decoder(
                    tgt=tgt_, memory=memory, memory_key_padding_mask=memory_key_padding_mask.to(self.device))[0, -1, :]
                filtered_logits = self.top_k_top_p_filtering(
                    logits / temperature, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)

                generated_output.append(next_token)
                if next_token == EOS_token_id:
                    break
                next_word_emb = self.emb_layer(torch.tensor(
                    [next_token]).unsqueeze(0).to(self.device))
                tgt_ = torch.cat([tgt_, next_word_emb], axis=1).to(self.device)

        return generated_output

    def generate_greedy(self, desired_label, input_=None, memory=None, memory_key_padding_mask=None, max_len=128):
        '''
        one of memory and input_text should be given though
        '''
        assert not (input_ is None and memory is None)
        EOS_token_id = 5  # for snapp dataset
        self.eval_mode()
        if input_ is not None:
            input_text = torch.tensor(
                input_['text'] + [EOS_token_id], dtype=torch.int64).unsqueeze(0).to(self.device)
            src_padding_mask = torch.tensor(
                [False] * (len(input_text[0]))).unsqueeze(0).to(self.device)
            embedded_inputs = self.emb_layer(input_text)
            memory = self.encoder(
                embedded_inputs, src_padding_mask)

        generated_output = []
        with torch.no_grad():
            tgt_ = self.emb_layer(torch.tensor(
                [self.tokenizer.encoder.word_vocab[f'__style{desired_label+1}']]).unsqueeze(-1).to(
                self.device))

            while len(generated_output) < max_len:
                # should we use 'decoder_output'? let's talk about it
                dec_out = self.decoder(
                    tgt=tgt_, memory=memory, memory_key_padding_mask=memory_key_padding_mask.to(self.device))

                next_vocab = torch.argmax(dec_out[:, -1, :])  # batch size is 1
                generated_output.append(next_vocab)
                if next_vocab == EOS_token_id:
                    break
                next_word_emb = self.emb_layer(torch.tensor(
                    [next_vocab]).unsqueeze(0).to(self.device))
                tgt_ = torch.cat([tgt_, next_word_emb], axis=1).to(self.device)

        return generated_output

    def do_my_shit(self):
        dev_text = self.dev_dataset['text'][0]
        dev_label = self.dev_dataset['label'][0]
        print(self.dev_dataset[0])
        output0 = self.generate_beam(
            desired_label=dev_label, input_=self.dev_dataset[0])
        output1 = self.generate_beam(
            desired_label=1 - dev_label, input_=self.dev_dataset[0])
        print(output0)
        print(output1)
        print(len(output0[0][0]))
        print(len(output1))
        print(output0 == output1)

    def generate_beam(self, desired_label, input_=None, memory=None, memory_key_padding_mask=None, K=5, max_len=128):
        'TODO: This should also return a single sentence, not a list of them'
        # there are actually more ways for this to fail but we ignore them for now
        assert not (input_ is None and memory is None)
        EOS_token_id = 5  # for snapp dataset

        if input_ is not None:
            input_text = torch.tensor(
                input_['text'] + [EOS_token_id], dtype=torch.int64).unsqueeze(0).to(self.device)
            src_padding_mask = torch.tensor(
                [False] * (len(input_text[0]))).unsqueeze(0).to(self.device)
            embedded_inputs = self.emb_layer(input_text)
            memory = self.encoder(
                embedded_inputs, src_padding_mask)
            memory_key_padding_mask = src_padding_mask

        def next_most_probables(tgt, memory):
            with torch.no_grad():
                # should we use 'decoder_output'? let's talk about it
                dec_out = self.decoder(tgt=tgt, memory=memory,
                                       memory_key_padding_mask=memory_key_padding_mask.to(self.device))
                dec_out = F.softmax(dec_out[0, -1, :], dim=-1)
                next_tokens = torch.topk(dec_out, K)
                outputs = []
                for token in range(K):
                    outputs.append([next_tokens.indices[token],
                                    next_tokens.values[token]])
                return outputs

        target_sequences = [[[self.tokenizer.encoder.word_vocab[f'__style{desired_label+1}']], 0.0] * K]

        min_active_len = 1
        while min_active_len < max_len:
            new_targets = []
            for target in target_sequences:
                target_list = target[0]
                target_score = target[1]

                if target_list[-1] == EOS_token_id or len(target_list) >= max_len:  # how to deal with EOS?
                    new_targets.append(target)
                    continue

                tgt_ = self.emb_layer(torch.tensor(target_list).unsqueeze(0).to(
                    self.device))

                next_candidates = next_most_probables(tgt_, memory)
                # mult of probabilities is the sum of their logs
                for candidate in next_candidates:
                    # print(candidate[0].tolist())
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

    def show_results(self, original, original_label, desired_label, result):
        print(f'''Original sentence with label {original_label}:
                    {' '.join([word for word in self.tokenizer.inv_transform([[int(x) for x in original]])[0].split() if word not in ['__pad']])}''')
        print(f'''Generated sentence with label {desired_label}:''')
        print(
            f'''{self.tokenizer.inv_transform([[int(x) for x in result]])[0]}''')
        print("-" * 100)

    def evaluate(self, ):
        decode_mode = self.args.decoding_strategy
        n = 4
        self.eval_mode()
        text_batch, labels, src_key_padding_mask, tgt_mask = next(
            iter(self.dev_loader))
        memories = self.encoder(self.emb_layer(
            text_batch[:2 * n].to(self.device)), src_key_padding_mask[:2 * n].to(self.device))
        print('######### Evaluation #########')
        for i in range(2 * n):
            memory = memories[i].unsqueeze(0)
            desired_label = labels[i] if i < n else (
                labels[i] + 1) % self.args.num_styles
            if decode_mode == 'greedy':
                result = self.generate_greedy(
                    desired_label, memory=memory, memory_key_padding_mask=src_key_padding_mask[i].unsqueeze(0))
            elif decode_mode == 'beam':
                result = self.generate_beam(
                    desired_label, memory=memory, memory_key_padding_mask=src_key_padding_mask[i].unsqueeze(0), K=10)
            else:
                result = self.generate_sampling(
                    desired_label, memory=memory, memory_key_padding_mask=src_key_padding_mask[i].unsqueeze(0))
            
            self.show_results(
                text_batch[i], labels[i].item(), desired_label, result)

    def log(self, ):
        with open(os.path.join('./' + self.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f)

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
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

    def update_disc(self, epoch: int, batch_idx: int) -> bool:
        '''
        For the first copule of epochs, the discriminator will not be updated
        '''
        return epoch >= self.args.ae_pretraining_epochs

    def update_ae(self, epoch: int, batch_idx: int) -> bool:
        '''
        For the first couple of epochs, update the ae. After that, update only on some batches
        '''
        return (epoch < self.args.ae_pretraining_epochs) or (batch_idx % self.args.ae_update_freq == 0)

    def train_mode(self, ) -> None:
        self.emb_layer.train()
        self.encoder.train()
        self.decoder.train()
        self.disc.train()

    def eval_mode(self, ) -> None:
        self.emb_layer.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.disc.eval()
