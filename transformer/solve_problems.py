import argparse
import json
import multiprocessing
import torch
import time
import numpy as np

from transformer.model import PCCoder
from env.env import ProgramEnv
from dsl.example import Example
from dsl.program import Program
from dsl.value import Value
from env.statement import index_to_statement


def load_problems(path):
    problems = []
    with open(path) as fh:
        for line in fh:
            problems.append(json.loads(line.rstrip()))
    return problems


def init_worker(*args):
    global counter, fail_counter, model, timeout, max_program_len
    counter, fail_counter, model, timeout, max_program_len = args


def solve_problems(problems, model, timeout, max_program_len, num_workers):
    """
    Attempts to predict programs for the given I/O sample sets.
    """
    # Prevents deadlocks due to torch's problems with GPUs on multi processes.
    # This line is here for convenience, but it is recommended to solve problems on CPU since the overhead
    # in this case is minimal.
    torch.set_num_threads(1)

    counter = multiprocessing.Value('i', 0)
    fail_counter = multiprocessing.Value('i', 0)

    if num_workers is None or num_workers > 1:
        pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                                    initargs=(counter, fail_counter, model, timeout, max_program_len))
        return pool.map(solve_problem_worker, problems)
    else:
        # Don't run in pool to enable debugging
        init_worker(counter, fail_counter, model, timeout, max_program_len)
        return [solve_problem_worker(data) for data in problems]


def choose_top(probs, top_p):
    """
    Args:
        probs: Tensor, shape [batch_size, hid]
        top_p:

    Returns:

    """
    indices = torch.arange(probs.shape[1])
    bound_indices = torch.argmax((torch.cumsum(probs, 1) > top_p).to(torch.long), 1)

    mask = indices[None, :] <= bound_indices[:, None]
    probs *= mask
    probs /= probs.sum(1)[:, None]

    return probs

def sampling(start_env, max_program_len, model, timeout, top_p, temperature):
    sample_size = 64
    start_time = time.time()
    end_time = start_time + timeout

    num_steps = 0
    invalid_steps = 0

    batch_indices = torch.arange(sample_size)[:, None]
    while time.time() < end_time:
        envs = [start_env.copy() for _ in range(sample_size)]
        statements = [[] for _ in range(sample_size)]

        for _ in range(max_program_len):
            for env, statement_list in zip(envs, statements):
                if env.is_solution():
                    return {'result': statement_list, 'num_steps': num_steps, 'time': time.time() - start_time,
                            'num_invalid': invalid_steps}

            """if time.time() > end_time:
                return {'result': False, 'num_steps': num_steps, 'time': time.time() - start_time,
                        'num_invalid': invalid_steps}"""

            env_encodings = torch.tensor(np.array([env.get_encoding() for env in envs]))
            num_inputs = torch.tensor(np.array([env.states[0].num_inputs for env in envs]))
            num_vars = torch.tensor(np.array([env.num_vars for env in envs]))
            statement_pred, statement_log_probs, drop_indx = model.predict(env_encodings, num_inputs, num_vars)
            statement_pred = torch.flip(torch.tensor(statement_pred), (1,))

            statement_probs = torch.exp(statement_log_probs[batch_indices, statement_pred] / temperature)
            statement_probs /= torch.sum(statement_probs, 1)[:, None]
            statement_probs = choose_top(statement_probs, top_p)
            next_statement = statement_pred[batch_indices, torch.multinomial(statement_probs, 1)].squeeze().detach().numpy()

            for i, env in enumerate(envs):
                envs[i] = env.step_safe(index_to_statement[next_statement[i]])
                statements[i].append(index_to_statement[next_statement[i]])

                num_steps += 1
                if envs[i] is None:
                    invalid_steps += 1

            statements = [statement_list for env, statement_list in zip(envs, statements) if env is not None]
            envs = [env for env in envs if env is not None]

            if not envs:
                break

            repeat_indices = np.random.randint(len(envs), size=sample_size - len(envs))
            envs += [envs[i].copy() for i in repeat_indices]
            statements += [statements[i].copy() for i in repeat_indices]

    return {'result': False, 'num_steps': num_steps, 'time': time.time() - start_time,
            'num_invalid': invalid_steps}


def solve_problem_worker(data):
    examples = Example.from_line(data)
    env = ProgramEnv(examples)

    solution = sampling(env, max_program_len, model, timeout, top_p=0.9, temperature=0.5)

    counter.value += 1
    print("\rSolving problems... %d (failed: %d)" % (counter.value, fail_counter.value), end="")

    if solution['result'] is False:
        solution['result'] = "Failed"
        fail_counter.value += 1
    else:
        values = [Value.construct(x) for x in data['examples'][0]['inputs']]
        value_types = [x.type for x in values]
        solution['result'] = Program(value_types, solution['result']).encode()
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('timeout', type=int)
    parser.add_argument('max_program_len', type=int)
    parser.add_argument('--num_workers', type=int, default=None)

    args = parser.parse_args()

    problems = load_problems(args.input_path)

    model = PCCoder()
    model.load(args.model_path)

    model.eval()

    res = solve_problems(problems, model, args.timeout, args.max_program_len, args.num_workers)
    print("")

    solved = len([x for x in res if x['result'] != 'Failed'])
    print("Solved: %d\\%d:" % (solved, len(res)), str(100.0 * solved / len(res)) + '%')

    open(args.output_path, 'w').write('\n'.join([json.dumps(x) for x in res]))


if __name__ == '__main__':
    main()
