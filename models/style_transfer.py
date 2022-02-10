import json
import os
from statistics import mode

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
            self.save()
            # TODO: run evaluation on dev set

    def run_epoch(self, epoch):
        total_rec_loss, total_disc_loss, total_num_samples = 0, 0, 0
        running_rec_loss, running_disc_loss, running_num_samples, running_disc_correct_preds = 0, 0, 0, 0
        for idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            text_batch, labels, src_key_padding_mask, tgt_mask = batch
            batch_size = len(labels)
            running_num_samples += batch_size
            text_batch = text_batch.to(self.device)
            labels = labels.to(self.device)
            src_key_padding_mask = src_key_padding_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)

            # reconstruction loss optimization
            embedded_inputs = self.emb_layer(text_batch)
            encoder_output = self.encoder(
                embedded_inputs, src_key_padding_mask)  # bsz x seq_len x d_model
            encoder_output = torch.roll(encoder_output, shifts=1, dims=1)

            style_emb = self.emb_layer(torch.tensor(
                [self.tokenizer.encoder.word_vocab[f'__style{label+1}'] for label in labels]).unsqueeze(-1).to(
                self.device))  # TODO: optimize!
            # shape : bsz, 1, 256
            style_emb = style_emb.squeeze(1)
            encoder_output[:, 0, :] = style_emb
            dec_out = self.decoder(tgt=embedded_inputs, memory=encoder_output,
                                   memory_key_padding_mask=src_key_padding_mask,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=src_key_padding_mask)  # bsz, seq_len, vocab_size
            rec_loss = self.rec_loss_criterion(
                dec_out.flatten(0, 1), text_batch.flatten())

            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            rec_loss.backward()
            self.encoder_optim.step()
            self.decoder_optim.step()

            # TODO: should we log and report decoding accuracy? excluding the pads is a bit tough:)
            running_rec_loss += rec_loss.item() * batch_size

            # adv loss optimization
            if epoch > 2:
                # disc adv update
                embedded_inputs = self.emb_layer(text_batch)
                encoder_output = self.encoder(
                    embedded_inputs, src_key_padding_mask)
                encoder_output_detached = encoder_output.detach()
                # TODO add noise to inputs
                disc_logits = self.disc(
                    encoder_output_detached)  # bsz x num_labels

                self.disc_optim.zero_grad()
                adv_loss = self.args.LAMBDA * \
                           self.adv_loss_criterion(disc_logits, labels)
                adv_loss.backward()
                self.disc_optim.step()

                running_disc_loss += adv_loss.item() * batch_size
                disc_preds = torch.argmax(disc_logits, dim=-1)
                running_disc_correct_preds += torch.count_nonzero(
                    disc_preds == labels)

                # encoder adv update
                if (idx + 1) % 5 == 0:
                    disc_logits = self.disc(encoder_output)
                    self.encoder_optim.zero_grad()
                    adv_loss = -self.args.LAMBDA * \
                               self.adv_loss_criterion(disc_logits, labels)
                    adv_loss.backward()
                    self.encoder_optim.step()

            # logging every 100 minibatch
            if (idx + 1) % 100 == 0:
                total_num_samples += running_num_samples
                total_rec_loss += running_rec_loss
                total_disc_loss += running_disc_loss
                disc_acc = running_disc_correct_preds / running_num_samples
                global_step = epoch * \
                              (len(self.train_loader) // 100) + (idx + 1) // 100
                self.logger.add_scalar(
                    'Train/Loss/reconstruction', running_rec_loss / running_num_samples, global_step=global_step)
                self.logger.add_scalar(
                    'Train/Loss/discriminator', running_disc_loss / running_num_samples, global_step=global_step)
                self.logger.add_scalar(
                    'Train/Accuracy/discriminator', disc_acc, global_step=global_step)
                running_num_samples, running_rec_loss, running_disc_loss, running_disc_correct_preds = 0, 0, 0, 0

            if (idx + 1) % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_rec_loss, total_disc_loss

    # mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    # mask = mask.float().masked_fill(mask == 0, float(
    #     '-inf')).masked_fill(mask == 1, float(0.0))

    def generate_greedy(self, desired_label, input=None, memory=None, max_len=128):
        '''
        one of memory and input_text should be given though
        '''
        assert not (input is None and memory is None)
        EOS_token_id = 5  ## for snapp dataset

        if input is not None:
            input_text = torch.tensor(input['text'] + [EOS_token_id], dtype=torch.int64)
            src_padding_mask = torch.tensor([False] * (len(input_text) + 1))
            embedded_inputs = self.emb_layer(input_text)
            memory = self.encoder(
                embedded_inputs, src_padding_mask)

        encoder_output = torch.roll(memory, shifts=1, dims=1)

        style_emb = self.emb_layer(torch.tensor(
            [self.tokenizer.encoder.word_vocab[f'__style{desired_label+1}']]).unsqueeze(-1).to(
            self.device))

        style_emb = style_emb.squeeze(1)
        encoder_output[:, 0, :] = style_emb  ## why are we putting style embedding in encoder output again?? what about
        ## the decoder?

        generated_output = []
        with torch.no_grad():
            print(style_emb.size())
            tgt_ = self.emb_layer(torch.tensor(
                [self.tokenizer.encoder.word_vocab[f'__style{desired_label+1}']]).unsqueeze(-1).to(
                self.device))

            while len(generated_output) < max_len:
                dec_out = self.decoder(tgt=tgt_, memory=memory)  ## should we use 'decoder_output'? let's talk about it

                next_vocab = torch.argmax(dec_out[:, -1, :])  ##batch size is 1
                generated_output.append(next_vocab)
                if next_vocab == EOS_token_id:
                    break
                next_word_emb = self.emb_layer(torch.tensor([next_vocab])).to(self.device)
                tgt_ = torch.cat([tgt_, next_word_emb], axis=-1).to(self.device)

        return generated_output

    def evaluate(self, test_loader, ):
        pass

    def log(self, ):
        with open(os.path.join('./' + self.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f)
