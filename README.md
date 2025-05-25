# ✨ Local Identifying Causal Relations in the Presence of Latent Variables
This is the code repository for the local identify the causal relationship between a pair of variables.

## Examples
We provide an examples of running the LocICR  algorithm in `example.py`.


## Input Arguments

- **data_matrix**: Observation data matrix (n_samples x n_variables), type: `ndarray`
- **target_X**: Index of the target variable, type: `int`
- **alpha**: Significance level for conditional independence tests, type: `float`
- **max_k**: Maximum size of the conditioning set, type: `int`

## Output arguments:
- **causal_relation**: the causal relation between target_X and target_Y, datetype: int

## Dependencies

- numpy
- scipy
- scikit-learn
- statsmodels
- pandas
- networkx



## 📝 Citation
If you use this code, please cite the following paper:

Local Identifying Causal Relations in the Presence of Latent Variables. 
Zheng Li, Zeyu Liu, Feng Xie, Hao Zhang, Chunchen Liu, and Zhi Geng.
ICML, Vancouver, Canada, 2025.

If you have problems or questions, do not hesitate to send an email to zhengli0060@gmail.com or xiefeng009@gmail.com.