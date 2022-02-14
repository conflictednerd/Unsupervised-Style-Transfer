import argparse
from pprint import pprint
from typing import List

from models.style_transfer import StyleTransferModel


def get_parser():
    parser = argparse.ArgumentParser()
    # General Setting
    parser.add_argument('--exp-name', default='snapp_1',
                        help='Experiment name. Also used as the name of checkpoint file.')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='If True, will load model parameters from the file located at models-dir and named as exp-name.pt')
    parser.add_argument('--models-dir', default='./models_dir/',
                        type=str, help='Directory where models are saved to / loaded from')
    parser.add_argument('--data-dir', default='./data_dir/',
                        type=str, help='Directory where training/validation/testing data is located')
    parser.add_argument('--log-dir', default='logs',
                        type=str, help='Directory where logs are stored')
    parser.add_argument('--num-styles', default=2, type=int,
                        help='Number of styles: 2 for snapp and 3 for poems')
    parser.add_argument('--decoding-strategy', default='beam',
                        type=str,
                        help='Decoding strategy used for generating new sentences. One of "greedy", "beam" or "sampling"')
    parser.add_argument('--beam-width', default=15,
                        type=int, help='Beam width used in beam search decoding')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='If True, will run evaluation after training. This will print out a certain number of examples from dev set')

    # Training
    parser.add_argument('--epochs', default=50,  # TODO
                        type=int, help='batch size')
    parser.add_argument('--label-smoothing', default='0.2',
                        type=float, help='label smoothing ratio used in training the discriminator')
    parser.add_argument('--lambda-gan', default=50.0,
                        type=float, help='coefficient of adversarial loss')
    parser.add_argument('--ae-pretraining-epochs', default=10,
                        type=int,
                        help='The number of epochs that the autoencoder part will train without adversarial loss')
    parser.add_argument('--ae-update-freq', default=5,
                        type=int, help='(After pretraining) update the autoencoder once every this many minibatches')
    parser.add_argument('--scheduled-sampling', action='store_true', default=True, type=bool,
                        help='If True, will use scheduled sampling in training')
    parser.add_argument('--scheduled-sampling-iters', default=2,
                        type=int, help='Number of sampling iterations used in scheduled sampling. Larger values slow down training but reduces test time exposure bias.')

    # Dataset Setting
    parser.add_argument('--batch-size', default=32,
                        type=int, help='batch size')
    parser.add_argument('--tokenizer-name', default='snp_tokenizer',
                        type=str, help='snp_tokenizer or poem_tokenizer')
    parser.add_argument('--num-workers', default=2,
                        type=int, help='Number of workers in data loader')

    # Encoder
    parser.add_argument('--d_model', default=128,
                        type=int, help='Embedding dimension (same value is used in encoder, decoder')
    parser.add_argument('--encoder-nhead', default=8,
                        type=int, help='Number of attention heads in the encoder')
    parser.add_argument('--encoder-layers', default=2,
                        type=int, help='Number of transformer layers in the encoder')
    parser.add_argument('--encoder-ff', default=512,
                        type=int, help='Dimension of feed-forward layer in the encoder')
    parser.add_argument('--encoder-lr', default=2e-5, type=float,  # TODO
                        help='Encoder learning rate')

    # Decoder
    parser.add_argument('--decoder-nhead', default=8,
                        type=int, help='Number of attention heads in the decoder')
    parser.add_argument('--decoder-layers', default=2,
                        type=int, help='Number of transformer layers in the decoder')
    parser.add_argument('--decoder-ff', default=512,
                        type=int, help='Dimension of feed-forward layer in the decoder')
    parser.add_argument('--decoder-lr', default=5e-5, type=float,  # TODO
                        help='Decoder learning rate')

    # Discriminator
    parser.add_argument('--disc-channels', default=4,
                        type=int, help='Number of output channels for discriminators conv layers')
    parser.add_argument('--disc-kernels', default=[1, 2, 3, 4, 5, 6, 8, 10],
                        # [1,2,3,4,5,6,8,10,16,32,64,128] for poems
                        type=List[int], help='Size of convolution kernels used in the discriminator')
    parser.add_argument('--disc-lr', default=1e-4, type=float,  # TODO
                        help='Discriminator learning rate')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pprint(f'Arguments are: {vars(args)}')  # For debugging
    model = StyleTransferModel(args)
    model.train()
    if args.evaluate:
        model.evaluate(n=16, train_loader=True, dev_loader=True)
