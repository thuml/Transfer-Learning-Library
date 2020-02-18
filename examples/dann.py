from io_utils import basic_parser
from trainer import main


if __name__ == '__main__':
    parser = basic_parser()
    args = parser.parse_args()
    # TODO add more arguments for DANN
    main(args, training=True)

