import argparse
import torch
import numpy as np

from tllib.utils import CompleteLogger
import utils
import dataset
import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    logger = CompleteLogger(args.log)
    train_data_loader, val_data_loader, test_data_loader, field_dims, numerical_num = dataset.get_datasets(
        args.root, args.domain_names, args.batch_size)

    model = models.get_model(args.model_name, field_dims, numerical_num, args.train_tasks, args.embed_dim).to(device)
    if args.pretrained:
        print("Loading from ", args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'), strict=False)
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopper = utils.EarlyStopper(num_trials=2, save_path=logger.get_checkpoint_path("best"))
    for epoch_i in range(args.epochs):
        utils.train(model, optimizer, train_data_loader, epoch_i, device, args)
        auc = utils.test(model, val_data_loader, device, args)
        print('epoch:', epoch_i, 'test: auc:', auc)
        for task_name in args.test_tasks:
            print('task {}, AUC {}'.format(task_name, auc[task_name]))
        if not early_stopper.is_continuable(model, np.array(list(auc.values())).mean()):
            print(f'test: best auc: {early_stopper.best_accuracy}')
            break

    model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    auc = utils.test(model, test_data_loader, device, args)
    for task_name in args.test_tasks:
        print('task {}, AUC {}'.format(task_name, auc[task_name]))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--root', default='./data/')
    parser.add_argument('--domain_names', type=str, nargs='+', choices=["NL", "ES", "FR", "US"])
    parser.add_argument('-tr', '--train-tasks', help='training tasks(s)', nargs='+', choices=["click", "conversion"])
    parser.add_argument('-ts', '--test-tasks', help='test task(s)', nargs='+', choices=["click", "conversion"])
    # model parameters
    parser.add_argument('--model_name', default='sharedembedding')
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--pretrained', default=None)
    # training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--log', default="logs/")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")
    parser.add_argument('-p', '--print-freq', default=1000, type=int,
                        metavar='N', help='print frequency (default: 1000)')
    args = parser.parse_args()
    main(args)
