import json
import os

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from snapp_dataset import create_snapp_dataset_from_path, data_collator_snapp
from torch.utils.tensorboard import SummaryWriter

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
        self.hiddenToVocab = torch.nn.Linear(args.d_model, self.tokenizer.vocab_size).to(self.device)
        self.logger = SummaryWriter(self.log_dir)

        self.encoder_optim = optim.Adam(
            list(self.emb_layer.parameters()) +
            list(self.encoder.parameters()),
            lr=args.encoder_lr)
        self.decoder_optim = optim.Adam(
            self.decoder.parameters(),
            lr=args.decoder_lr)  ## Should we not merge encoder and decoder optimizations,
                                 ## and have anotehr one for encoder and discrimator?

        self.disc_optim = optim.Adam(self.disc.parameters(), lr=args.disc_lr)
        
        self.rec_loss_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.adv_loss_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

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
            print(f'[Epoch: {(epoch + 1)}]')
            train_dec_loss, train_disc_loss = self.run_epoch()

    def save(self, ) -> None:
        pass

    def load(self, ) -> None:
        pass

    def run_epoch(self, ):

        '''

        I think we must have two optimizers, one for (encoder, decoder), one for (encoder, discriminator)

        '''
        for idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            text_list_input, label_list, \
            src_key_padding_mask, text_list_output, \
            tgt_mask, tgt_key_padding_mask, memory_key_padding_mask = batch

            ## optimizer.zero_grad()
            embedded_inputs = self.emb_layer(text_list_input.to(self.device))
            encoder_output = self.encoder(embedded_inputs, src_key_padding_mask.to(self.device))
            decoder_output = self.decoder(text_list_output.to(self.device), encoder_output, tgt_mask.to(self.device),
                                          tgt_key_padding_mask.to(self.device), memory_key_padding_mask.to(self.device))

    def run_on_batch(self, batch, ):
        pass

    def evaluate(self, test_loader, ):
        pass

    def log(self, ):
        with open(os.path.join('./' + self.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f)
