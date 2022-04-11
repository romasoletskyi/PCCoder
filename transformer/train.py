from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import torch
import multiprocessing

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

import params
from env.env import ProgramEnv
from dsl.program import Program
from dsl.example import Example

from transformer.statement import parse_args, incomplete_statement_to_index, num_incomplete_statements
from transformer.model import PCCoder, generate_mask

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

    # TODO parse program with more than one input - special mask
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
    states = []
    operators = []
    variables = [[] for _ in range(params.num_variable_head)]
    variables_mask = [[] for _ in range(params.num_variable_head)]

    print("Loading dataset...")
    lines = fileobj.read().splitlines()
    if max_len is not None:
        lines = lines[:max_len]

    pool = multiprocessing.Pool()
    res = list(tqdm(pool.imap(generate_prog_data, lines), total=len(lines)))
    pool.close()

    for state, operator_list, variable_list, mask_list in res:
        states.append(state)
        num_empty_steps = params.state_len - len(operator_list)
        operators.append(operator_list + [num_incomplete_statements] * num_empty_steps)

        for i in range(params.num_variable_head):
            variables[i].append([var[i] for var in variable_list] + [0] * num_empty_steps)
            variables_mask[i].append([mask[i] for mask in mask_list] + [0] * num_empty_steps)

    return np.array(states), np.array(operators), np.array(variables), np.array(variables_mask)


class ProgramDataset(Dataset):
    def __init__(self, states, operators, variables, variables_mask):
        self.states = states
        self.operators = operators
        self.variables = variables
        self.variables_mask = variables_mask

    def __getitem__(self, index):
        return self.states[index], self.operators[index], self.variables[:, index], self.variables_mask[:, index]

    def __len__(self):
        return len(self.states)


def program_collate_fn(batch):
    data = [np.array([x[i] for x in batch]) for i in range(4)]
    data[2] = np.swapaxes(data[2], 0, 1)
    data[3] = np.swapaxes(data[3], 0, 1)
    return data


def model_loss(model, device, operator_criterion, var_criterion, batch):
    for i, array in enumerate(batch):
        batch[i] = torch.from_numpy(batch[i]).to(device)
    states, operators, variables, masks = batch

    pred_operators, *pred_variables = model(states, generate_mask(params.state_len))
    operator_loss = operator_criterion(pred_operators.flatten(end_dim=1), operators.flatten())
    t = var_criterion(pred_variables[0].flatten(end_dim=1), variables[0].flatten())
    print([(x.item(), y.item(), z, u.item()) for x, y, z, u in zip(t, masks[0].flatten(), pred_variables[0].flatten(end_dim=1), variables[0].flatten())])
    variables_losses = [torch.sum(var_criterion(pred_head.flatten(end_dim=1),
                                                var_head.flatten()) * mask.flatten()) / torch.sum(mask)
                        for pred_head, var_head, mask in zip(pred_variables, variables, masks)]

    return operator_loss, variables_losses


def train(args):
    with open(args.input_path, 'r') as f:
        data, operator_target, var_target, var_mask = load_data(f, args.max_len)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_size = int(0.9 * len(data))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_dataset = ProgramDataset(data[train_indices], operator_target[train_indices],
                                   var_target[:, train_indices], var_mask[:, train_indices])
    test_dataset = ProgramDataset(data[test_indices], operator_target[test_indices],
                                  var_target[:, test_indices], var_mask[:, test_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=program_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=program_collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PCCoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)

    operator_criterion = nn.CrossEntropyLoss()
    var_criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        print("Epoch %d" % epoch)

        operator_losses = []
        var_losses = []

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            operator_loss, variables_loss = model_loss(model, device, operator_criterion, var_criterion, batch)

            loss = operator_loss + sum(var_loss for var_loss in variables_loss)
            operator_losses.append(operator_loss.item())
            var_losses.append(np.mean([var_loss.item() for var_loss in variables_loss]))

            loss.backward()
            optimizer.step()

        lr_sched.step()

        avg_statement_train_loss = np.array(operator_losses).mean()
        avg_var_train_loss = np.array(var_losses).mean()

        print("Train loss: S %f" % avg_statement_train_loss, "V %f" % avg_var_train_loss)

        model.eval()
        operator_losses = []
        var_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                operator_loss, variables_loss = model_loss(model, device, operator_criterion, var_criterion, batch)

                loss = operator_loss + sum(var_loss for var_loss in variables_loss)
                operator_losses.append(operator_loss.item())
                var_losses.append(np.mean([var_loss.item() for var_loss in variables_loss]))

            avg_statement_test_loss = np.array(operator_losses).mean()
            avg_var_test_loss = np.array(var_losses).mean()

            print("Train loss: O %f" % avg_statement_train_loss, "V %f" % avg_var_train_loss)
            print("Test loss: O %f" % avg_statement_test_loss, "V %f" % avg_var_test_loss)

        model.save(args.output_path + ".%d" % epoch)


if __name__ == '__main__':
    main()
