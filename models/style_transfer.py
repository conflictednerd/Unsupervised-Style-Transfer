import json
import os
from json import encoder

import torch
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
            ignore_index=0).to(self.device)
        self.adv_loss_criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing).to(self.device)

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
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_snapp)
        self.dev_loader = DataLoader(
            self.dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_snapp)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator_snapp)

    def train(self, ) -> None:
        for epoch in range(self.args.epochs):
            print(f'[Epoch: {epoch+1}/{self.args.epochs}]')
            train_rec_loss, train_disc_loss, train_enc_loss = self.run_epoch()
            # Log

    def save(self, ) -> None:
        pass

    def load(self, ) -> None:
        pass

    def run_epoch(self, ):
        total_rec_loss, total_disc_loss, total_enc_loss = 0, 0, 0
        for idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            text_list_input, labels, \
                src_key_padding_mask, text_list_output, \
                tgt_mask, tgt_key_padding_mask, memory_key_padding_mask = batch

            text_list_input = text_list_input.to(self.device)

            # reconstruction loss optimization
            embedded_inputs = self.emb_layer(text_list_input)
            encoder_output = self.encoder(
                embedded_inputs, src_key_padding_mask)  # bsz x seq_len x d_model
            encoder_output = torch.roll(encoder_output, shifts=1, dim=1)

            style_emb = self.emb_layer(torch.tensor(
                [self.tokenizer.encoder.word_vocab[f'__style{label+1}'] for label in labels]).unsqueeze(-1).to(self.device))  # TODO: optimize!
            # shape : bsz, 1, 256
            encoder_output[:, 0, :] = style_emb
            dec_out = self.decoder(encoder_output, memory_key_padding_mask=src_key_padding_mask,
                                   tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # bsz, seq_len, vocab_size
            rec_loss = self.rec_loss_criterion(
                dec_out.flatten(0, 1), text_list_output.flatten())
            total_rec_loss += rec_loss
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            rec_loss.backward()
            self.encoder_optim.step()
            self.decoder_optim.step()

            # adv loss optimization
            embedded_inputs = self.emb_layer(text_list_input)
            encoder_output = self.encoder(
                embedded_inputs, src_key_padding_mask)
            encoder_output_detached = encoder_output.detach()
            # add noise to inputs
            disc_logits = self.disc(
                encoder_output_detached)  # bsz x num_labels
            self.disc_optim.zero_grad()
            adv_loss = self.adv_loss_criterion(disc_logits, labels)
            total_disc_loss += adv_loss
            adv_loss.backward()
            self.disc_optim.step()

            disc_logits = self.disc(encoder_output)
            self.encoder_optim.zero_grad()
            adv_loss = - self.adv_loss_criterion(disc_logits, labels)
            total_enc_loss += adv_loss
            adv_loss.backward()
            self.encoder_optim.step()

        return total_rec_loss, total_disc_loss, total_enc_loss

    def evaluate(self, test_loader, ):
        pass

    def log(self, ):
        with open(os.path.join('./' + self.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f)
