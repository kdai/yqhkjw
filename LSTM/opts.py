import argparse

def parser_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', type=str, help='choose a domain')
    parser.add_argument('-f', '--feature_dim', type=int, default=512, help='dimension of feature vector')
    parser.add_argument('-x', '--hidden_dim', type=int, default=512, help='dimension of hidden vector')
    parser.add_argument('-s', '--seq_len', type=int, default=4, help='number of clips for resampling')
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-3, help='learning rate')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-5, help='weight decay for SGD')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--step_size', type=int, default=50, help='period for updating learning rate')
    parser.add_argument('--gamma', type=float, default=.1, help='multiplicative factor of learning rate decay')
    parser.add_argument('-p', '--proportion', type=float, default=.3, help='proportion of highlight in the entire length'
                                                                           'of the video')
    args = parser.parse_args()

    return args

