# Deep Optimal Stopping

PyTorch implementation of the Deep Optimal Stopping algorithm and all three numerical examples from  
**“Deep Optimal Stopping”** – Becker, Cheridito & Jentzen (2019) – created for my master’s thesis:

> **“The Optimal Stopping Problem: A Deep Learning Approach and its Applications”**  
> University of Zurich, 2025

---

## Quick Run

### One-off run (no install)

```bash
# from the repository root (folder name: deep-optimal-stopping/)
export PYTHONPATH=$(pwd)
```
```bash
# Example 1
python experiments/ex1/sweep.py
# Example 2
python experiments/ex2/sweep.py
```

Runs training for all dimensions (d).  
To reduce runtime or test specific cases, you can restrict dimensions, e.g.:


```bash
# Example 1
python experiments/ex1/sweep.py --d 2 3 5
# Example 2
python experiments/ex2/sweep.py --d 10 15
```

```bash
# Example 3
python experiments/ex3/sweep.py --H 0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40
python experiments/ex3/sweep.py --H 0.45 0.50 0.55
python experiments/ex3/sweep.py --H 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00

```
Training was split across Hurst values (H) due to varying values of J. See my thesis for details.

---

## Directory layout (abridged)

```
deep-optimal-stopping/                            
├── deep_optimal_stopping/
│   ├── __init__.py
│   ├── bounds.py
│   ├── models.py
│   ├── training.py
│   ├── utils.py
│   └── simulate/
│       ├── __init__.py
│       ├── bs.py              # GBM simulators (Ex 1 & 2)
│       └── fbm.py             # fBm simulator    (Ex 3)
├── experiments/
│   ├── __init__.py
│   ├── ex1/
│   │   ├── __init__.py
│   │   ├── driver.py
│   │   ├── sweep.py
│   │   └── helpers.py
│   ├── ex2/ …
│   └── ex3/ …
├── results/                   # CSV outputs
```

---

## Reference

Becker S., Cheridito P., Jentzen A. (2019).  
**Deep Optimal Stopping.** *Journal of Machine Learning Research*.  
<https://jmlr.org/papers/volume20/18-232/18-232.pdf>

---

## License

Released for academic use in conjunction with my master’s thesis.  
