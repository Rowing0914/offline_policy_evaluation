# Offline Policy Evaluation: Counterfactual Estimator
This repo is to reproduce the result of Doubly Robust Estimator.


## Dataset: `./data/*`
|   Dataset   | Samples | Features | Labels |
|:-----------:|:-------:|:--------:|:------:|
|       ecoli |     336 |        7 |      8 |
|       glass |     214 |        9 |      6 |
|      letter |   20000 |       16 |     26 |
|   optdigits |    5620 |       64 |     10 |
| page_blocks |    5473 |       10 |      5 |
|   pendigits |   10992 |       16 |     10 |
|    satimage |    6435 |       36 |      6 |
|     vehicle |     846 |       18 |      4 |
|       yeast |    1484 |        8 |     10 |

**All data can be found under UCI's great repositories

## Results
- [Bias](https://github.com/Rowing0914/doubly_robust_estimator/blob/master/results/result_bias.txt): `./results/result_bias.txt`
- [RMSE](https://github.com/Rowing0914/doubly_robust_estimator/blob/master/results/result_rmse.txt): `./results/result_rmse.txt`


## Usage
- Get dependencies: `pip install -r requirements.txt`
- Run the main file: `python main.py`


## Dependencies
- Python: 3.6.8
- Packages: See `./requirements.txt`
- OS: Windows10, MacOS, Ubuntu(18.04 LTS)


## Reference
- [IPS/IS](http://www.stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf)
- [Capped IPS/IS](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/bottou13a.pdf)
- [Self Normalised IPS/IS](https://www.cs.cornell.edu/people/tj/publications/swaminathan_joachims_15d.pdf)
- [Doubly Robust](https://arxiv.org/abs/1103.4601)