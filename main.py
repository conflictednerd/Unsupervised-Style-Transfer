import argparse
from pprint import pprint

from models.decoder import Decoder

x = Decoder()
print('Done')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='...',
                        help='...')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='...')
    parser.add_argument('--batch_size', default=8,
                        type=int, help='...')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pprint(f'Arguments are: {vars(args)}')  # For debugging
