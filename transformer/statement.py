from dsl.impl import ALL_FUNCTIONS, LAMBDAS
from dsl.types import FunctionType, INT, LIST

import params
import itertools


class IncompleteStatement(object):
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __repr__(self):
        return "<IncompleteStatement: %s %s>" % (self.func, self.args)

    def __eq__(self, other):
        if isinstance(other, IncompleteStatement):
            return False
        return self.func == other.func and self.args == other.args

    def __hash__(self):
        return hash(str(self))


def parse_args(func, args):
    input_type = func.input_type
    if not isinstance(input_type, tuple):
        input_type = (input_type,)

    dropped_args = []
    variables = []
    variable_mask = []

    for type, arg in zip(input_type, args):
        if type in [LIST, INT]:
            dropped_args.append(None)
            variables.append(arg)
            variable_mask.append(1)
        else:
            dropped_args.append(arg)

    variables += [0] * (params.num_variable_head - len(variables))
    variable_mask += [0] * (params.num_variable_head - len(variables))

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


incomplete_statement_space = build_incomplete_statement_space()
num_incomplete_statements = len(incomplete_statement_space)
index_to_incomplete_statement = dict([(indx, statement) for indx, statement in enumerate(incomplete_statement_space)])
incomplete_statement_to_index = {v: k for k, v in index_to_incomplete_statement.items()}
