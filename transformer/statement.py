from dsl.impl import ALL_FUNCTIONS, LAMBDAS
from dsl.types import FunctionType, INT, LIST

import params
import itertools
import torch

from env.statement import index_to_statement


class IncompleteStatement(object):
    def __init__(self, func, args):
        self.func = func
        self.args = tuple(args)

    def __repr__(self):
        return "<IncompleteStatement: %s %s>" % (self.func, self.args)

    def __eq__(self, other):
        if not isinstance(other, IncompleteStatement):
            return False
        return self.func == other.func and self.args == other.args

    def __hash__(self):
        return hash(str(self))


def parse_args(func, args, num_inputs):
    input_type = func.input_type
    if not isinstance(input_type, tuple):
        input_type = (input_type,)

    dropped_args = []
    variables = []
    variable_mask = []

    for type, arg in zip(input_type, args):
        if type in [LIST, INT]:
            dropped_args.append(None)
            if num_inputs is None:
                variables.append(arg)
            else:
                if isinstance(num_inputs, int):
                    variables.append(arg if arg < num_inputs else arg + 1 + (params.num_inputs - num_inputs))
                else:
                    variables.append(torch.where(arg < num_inputs, arg, arg + 1 + (params.num_inputs - num_inputs)))
            variable_mask.append(1)
        else:
            dropped_args.append(arg)

    variables += [0] * (params.num_variable_head - len(variables))
    variable_mask += [0] * (params.num_variable_head - len(variable_mask))

    return IncompleteStatement(func, dropped_args), variables, variable_mask


def build_incomplete_statement_space():
    statements = []
    for func in ALL_FUNCTIONS:
        input_type = func.input_type
        if not isinstance(input_type, tuple):
            input_type = (input_type,)

        argslists = []
        for type in input_type:
            if type in [LIST, INT]:
                argslists.append([None])
            elif isinstance(type, FunctionType):
                argslists.append([x for x in LAMBDAS if x.type == type])
            else:
                raise ValueError("Invalid input type encountered!")
        statements += [IncompleteStatement(func, x) for x in list(itertools.product(*argslists))]

    return statements


def build_tensors():
    statement_to_operator = torch.zeros(len(index_to_statement), dtype=torch.int64)
    statement_to_variables = [torch.zeros(len(index_to_statement), dtype=torch.int64) for _ in
                              range(params.num_variable_head)]
    statement_to_variables_mask = [torch.zeros(len(index_to_statement), dtype=torch.int64) for _ in
                              range(params.num_variable_head)]

    for i, statement in index_to_statement.items():
        func, args = statement.function, statement.args
        incomplete_statement, variables, variables_mask = parse_args(func, args, None)
        statement_to_operator[i] = incomplete_statement_to_index[incomplete_statement]

        for j in range(params.num_variable_head):
            statement_to_variables[j][i] = variables[j]
            statement_to_variables_mask[j][i] = variables_mask[j]

    return statement_to_operator, statement_to_variables, statement_to_variables_mask


incomplete_statement_space = build_incomplete_statement_space()
num_incomplete_statements = len(incomplete_statement_space)
index_to_incomplete_statement = dict([(indx, statement) for indx, statement in enumerate(incomplete_statement_space)])
incomplete_statement_to_index = {v: k for k, v in index_to_incomplete_statement.items()}
statement_to_operator, statement_to_variables, statement_to_variables_mask = build_tensors()
