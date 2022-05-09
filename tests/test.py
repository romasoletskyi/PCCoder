import numpy as np
import torch

import params

from transformer.model import PCCoder, generate_mask
from transformer.solve_problems import load_problems
from dsl.example import Example
from env.env import ProgramEnv


def main():
    model = PCCoder()
    model.load('model_pointer.39')
    model.eval()
    model.clear_cache()
    model.set_mode(True)

    slow_model = PCCoder()
    slow_model.load('model_pointer.39')
    slow_model.eval()

    problems = load_problems('smallval_dataset_gps')

    data = problems[0]
    examples = Example.from_line(data)
    start_env = ProgramEnv(examples)
    envs = [start_env.copy() for _ in range(1)]

    env_encodings = torch.tensor(np.array([env.get_encoding() for env in envs]))
    num_inputs = torch.tensor(np.array([env.states[0].num_inputs for env in envs]))
    num_vars = torch.tensor(np.array([env.num_vars for env in envs]))

    mask = generate_mask()

    """x = torch.randn(1, params.state_len, params.var_encoder_size)
    encoding = model.encoder.transformer.layers[0](x, mask)
    slow_encoding = slow_model.encoder.transformer.layers[0](x, mask)"""

    """x = torch.randn(1, params.state_len, params.dense_output_size)
    weight = model.variables_head[0](x, mask)
    slow_weight = slow_model.variables_head[0](x, mask)"""

    statement_pred, statement_log_probs = model.predict(env_encodings, num_inputs, num_vars)
    slow_statement_pred, slow_statement_log_probs = slow_model.predict(env_encodings, num_inputs, num_vars)

    diff = np.max(np.abs(statement_pred - slow_statement_pred))
    print('end')


if __name__ == "__main__":
    main()
