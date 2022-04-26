import json
import numpy as np

from collections import defaultdict
from dsl.program import Program

def analyze(result_name, data_name):
    with open(result_name, 'r') as f:
        result_lines = f.readlines()
       
    with open(data_name, 'r') as f:
        data_lines = f.readlines()
        
    len_to_result = defaultdict(list)
    for result_line, data_line in zip(result_lines, data_lines):
        program = Program.parse(json.loads(data_line.rstrip())['program'])
        program_len = len(program.statements)        
        result = json.loads(result_line.rstrip())['result'] != 'Failed'
        len_to_result[program_len].append(result)
     
    len_to_ratio = {l: np.mean(len_to_result[l]) for l in sorted(len_to_result, key=len_to_result.get)}
    return len_to_ratio
