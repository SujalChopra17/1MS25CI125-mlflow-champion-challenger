# Lab 05: Multiple Experiments & Runs — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Demonstrate organizing multiple runs within targeted experiments to analyze:
1. Impact of training vs. test split sizes (0.2, 0.3, 0.4) on model performance with a fixed random seed (`seed = 42`).
2. Sensitivity across different random states (7, 42, 100) with a fixed split size (`test_size = 0.25`).

---

## 2. Execution Command
```bash
python multiple_runs.py
```

---

## 3. Genuine Execution Output

### Part 1: Test Size Comparison (`Lab05_Test_Size_Comparison`) — Fixed Seed 42
| Run Name | Test Split | Training Size ($N_{train}$) | Test Size ($N_{test}$) | MAE | R² Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **test_size_20pct** | 0.20 | 80 | 20 | 9.0197 | 0.9834 |
| **test_size_30pct** | 0.30 | 70 | 30 | 8.7518 | 0.9853 |
| **test_size_40pct** | 0.40 | 60 | 40 | 8.9206 | 0.9854 |

### Part 2: Sensitivity to Random Seed (`Lab05_Random_State_Sensitivity`) — Fixed Test Size 0.25
| Run Name | Random State | Training Size ($N_{train}$) | Test Size ($N_{test}$) | MAE | R² Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **seed_7** | 7 | 75 | 25 | 10.6369 | 0.9798 |
| **seed_42** | 42 | 75 | 25 | 9.0632 | 0.9829 |
| **seed_100** | 100 | 75 | 25 | 12.3879 | 0.9693 |

---

## 4. Key Learnings & Observations
- **Test Size Dynamics:** Moving from 20% to 40% test split maintains stable R² (~0.983 to 0.985), indicating that 60 to 80 samples are sufficient to capture the linear relationship.
- **Random Seed Variance:** Varying the random seed creates noticeable fluctuation in MAE (from 9.0632 up to 12.3879). This demonstrates why fixed random seeds are indispensable for reproducibility in MLOps evaluation.
- **MLflow Multi-Run Advantage:** Storing multiple runs inside the same experiment enables instant side-by-side sorting, parameter vs. metric scatter plots, and parallel coordinates visualization in the MLflow UI.
