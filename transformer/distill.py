import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import params

from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm

from transformer.train import ProgramDataset, load_data, program_collate_fn
from transformer.model import PCCoder, Encoder, generate_mask

learn_rate = 0.001
batch_size = 100
num_epochs = 40

test_iterator_size = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to data')
    parser.add_argument('base_path', type=str, help='Path to base model')
    parser.add_argument('output_path', type=str, help='Output path of trained model')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Optional limit to the dataset size (usually for debugging)')
    args = parser.parse_args()
    train(args)


def cross_entropy_loss(inp, target):
    log_prob = torch.where(inp == -torch.inf, torch.tensor(0.0), F.log_softmax(inp, dim=-1))
    return -torch.sum(log_prob * target, dim=-1)


def model_loss(base_model, model, device, operator_criterion, batch):
    for i, array in enumerate(batch):
        batch[i] = torch.from_numpy(batch[i]).to(device)
    states, operators, variables, masks = batch
    transformer_mask = generate_mask().to(device)

    pred_operators, *pred_variables = model(states, transformer_mask)
    pred_operators = pred_operators[:, params.num_inputs:-1]
    pred_variables = [pred_var[:, params.num_inputs:-1] for pred_var in pred_variables]

    base_operators, *base_variables = base_model(states, transformer_mask)
    base_operators_prob = F.softmax(base_operators[:, params.num_inputs:-1], dim=-1)
    base_variables_prob = [F.softmax(base_var[:, params.num_inputs:-1], dim=-1) for base_var in base_variables]

    operator_loss = operator_criterion(pred_operators.flatten(end_dim=1),
                                       base_operators_prob.flatten(end_dim=1))

    variables_losses = [torch.sum(cross_entropy_loss(pred_head.flatten(end_dim=1),
                                                     base_head.flatten(end_dim=1)) * mask.flatten()) / torch.sum(mask)
                        for pred_head, base_head, mask in zip(pred_variables, base_variables_prob, masks)]

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

    base_model = PCCoder(lambda: Encoder())
    base_model.load(args.base_path)
    base_model = base_model.to(device)
    for param in base_model.parameters():
        param.requires_grad = False

    model = PCCoder(lambda: Encoder(layer_num=3, n_head=4, dim_feedforward=128)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    lr_sched = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=8,
        num_training_steps=num_epochs
    )

    operator_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        print("Epoch %d" % epoch)

        operator_losses = []
        var_losses = []

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            operator_loss, variables_loss = model_loss(base_model, model, device,
                                                       operator_criterion, batch)

            loss = operator_loss + sum(var_loss for var_loss in variables_loss)
            operator_losses.append(operator_loss.item())
            var_losses.append(np.mean([var_loss.item() for var_loss in variables_loss]))

            loss.backward()
            optimizer.step()

        lr_sched.step()

        avg_statement_train_loss = np.array(operator_losses).mean()
        avg_var_train_loss = np.array(var_losses).mean()

        model.eval()
        operator_losses = []
        var_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                operator_loss, variables_loss = model_loss(base_model, model, device,
                                                           operator_criterion, batch)

                operator_losses.append(operator_loss.item())
                var_losses.append(np.mean([var_loss.item() for var_loss in variables_loss]))

            avg_statement_test_loss = np.array(operator_losses).mean()
            avg_var_test_loss = np.array(var_losses).mean()

            print("Train loss: O %f" % avg_statement_train_loss, "V %f" % avg_var_train_loss)
            print("Test loss: O %f" % avg_statement_test_loss, "V %f" % avg_var_test_loss)

        model.save(args.output_path + ".%d" % epoch)


if __name__ == '__main__':
    main()
