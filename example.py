import itertools
from Loc_ICR import LocalCausalLearner
import numpy as np

if __name__ == "__main__":

    target_X = 26 
    target_Y = 32

    data_matrix = np.loadtxt(f'Example_data/example_data_1000.csv', delimiter=',')
    learner = LocalCausalLearner(data_matrix, target_X, alpha=0.01, max_k=10, verbose=False)
    causal_relation = learner._causal_relation_identify(target_Y)
    CI_test_num = learner._get_citest_num()
    print(f"target {target_X} and {target_Y}'s causal relation: {causal_relation}, CI_test_num: {CI_test_num}")


