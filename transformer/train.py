from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import random
import torch
import multiprocessing

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm

import params
from model.model import PCCoder
from cuda import use_cuda, LongTensor, FloatTensor
from env.env import ProgramEnv
from env.operator import Operator, operator_to_index
from env.statement import Statement, statement_to_index
from dsl.program import Program
from dsl.example import Example

from transformer.statement import parse_args, incomplete_statement_to_index

learn_rate = 0.001
batch_size = 100
num_epochs = 40

test_iterator_size = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to data')
    parser.add_argument('output_path', type=str, help='Output path of trained model')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Optional limit to the dataset size (usually for debugging)')
    args = parser.parse_args()
    train(args)


def generate_prog_data(line):
    data = json.loads(line.rstrip())
    examples = Example.from_line(data)
    env = ProgramEnv(examples)
    program = Program.parse(data['program'])

    operators = []
    variables = []
    variable_mask = []

    for i, statement in enumerate(program.statements):
        # Translate absolute indices to post-drop indices
        f, args = statement.function, list(statement.args)
        for j, arg in enumerate(args):
            if isinstance(arg, int):
                args[j] = env.real_var_idxs.index(arg)

        operator, var, var_mask = parse_args(f, args)
        operators.append(incomplete_statement_to_index[operator])
        variables.append(var)
        variable_mask.append(var_mask)

        env.step(statement)

    return env.get_encoding(), operators, variables, variable_mask


def load_data(fileobj, max_len):
    X = []
    Y = []
    Z = []
    W = []

    print("Loading dataset...")
    lines = fileobj.read().splitlines()
    if max_len is not None:
        lines = lines[:max_len]

    pool = multiprocessing.Pool()
    res = list(tqdm(pool.imap(generate_prog_data, lines), total=len(lines)))
    pool.close()

    for input, target, to_drop, operators in res:
        X += input
        Y += target
        Z += to_drop
        W += operators

    return np.array(X), np.array(Y), np.array(Z), np.array(W)


def train(args):
    with open(args.input_path, 'r') as f:
        data, statement_target, drop_target, operator_target = load_data(f, args.max_len)

    model = PCCoder()

    if use_cuda:
        model.cuda()

    model = nn.DataParallel(model)

    # The cuda types are not used here on purpose - most GPUs can't handle so much memory
    data, statement_target, drop_target, operator_target = torch.LongTensor(data), torch.LongTensor(statement_target), \
                                                    torch.FloatTensor(drop_target), torch.LongTensor(operator_target)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    statement_criterion = nn.CrossEntropyLoss()
    var_criterion = nn.CrossEntropyLoss()

    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)

    dataset_size = data.shape[0]
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_size = int(0.9 * dataset_size)
    train_data = data[indices[:train_size]]
    train_statement_target = statement_target[indices[:train_size]]
    train_drop_target = drop_target[indices[:train_size]]
    train_operator_target = operator_target[indices[:train_size]]

    test_data = Variable(data[indices[train_size:]].type(LongTensor))
    test_statement_target = Variable(statement_target[indices[train_size:]].type(LongTensor))
    test_drop_target = Variable(drop_target[indices[train_size:]].type(FloatTensor))
    test_operator_target = Variable(operator_target[indices[train_size:]].type(LongTensor))

    train_dataset = TensorDataset(train_data, train_statement_target, train_drop_target, train_operator_target)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        print("Epoch %d" % epoch)
        lr_sched.step()

        statement_losses = []
        var_losses = []

        for batch in tqdm(data_loader):
            current_state, next_operator, next_vars, var_mask = batch

            optimizer.zero_grad()
            pred_operator, pred_first_var, pred_second_var = model(current_state)

            statement_loss = statement_criterion(pred_operator, next_operator)
            var_loss = sum(var_criterion(pred_first_var, var) * mask for var, mask in zip(next_vars, var_mask))
            loss = statement_loss + var_loss

            statement_losses.append(statement_loss.item())
            var_losses.append(var_loss.item())

            loss.backward()
            optimizer.step()

        avg_statement_train_loss = np.array(statement_losses).mean()
        avg_var_train_loss = np.array(var_losses).mean()

        print("Train loss: S %f" % avg_statement_train_loss, "V %f" % avg_var_train_loss)
        """
        model.eval()

        with torch.no_grad():
            # Iterate through test set to avoid out of memory issues
            statement_pred, drop_pred, operator_pred = [], [], []
            for i in range(0, len(test_data), test_iterator_size):
                output = model(test_data[i: i + test_iterator_size])
                statement_pred.append(output[0])
                drop_pred.append(output[1])
                operator_pred.append(output[2])

            statement_pred = torch.cat(statement_pred, dim=0)
            drop_pred = torch.cat(drop_pred, dim=0)
            operator_pred = torch.cat(operator_pred, dim=0)

            test_statement_loss = statement_criterion(statement_pred, test_statement_target)
            test_drop_loss = drop_criterion(drop_pred, test_drop_target)
            test_operator_loss = operator_criterion(operator_pred, test_operator_target)

            print("Train loss: S %f" % avg_statement_train_loss, "D %f" % avg_drop_train_loss,
                  "F %f" % avg_operator_train_loss)
            print("Test loss: S %f" % test_statement_loss.item(), "D %f" % test_drop_loss.item(),
                  "F %f" % test_operator_loss.item())

            predict = statement_pred.data.max(1)[1]
            test_error = (predict != test_statement_target.data).sum().item() / float(test_data.shape[0])
            print("Test classification error: %f" % test_error)
        """
        model.module.save(args.output_path + ".%d" % epoch)


if __name__ == '__main__':
    main()
