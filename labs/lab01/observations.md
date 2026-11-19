# Lab 01: Introduction to MLOps — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Exercise 1: Baseline Linear Regression Model
### Objective
Evaluate how varying the test split size (0.20, 0.25, 0.30, 0.40) impacts sample partitioning and generalization performance measured by Mean Absolute Error (MAE).

### Execution Command
```bash
python baseline_model.py
```

### Genuine Execution Output
| Test Size | Training Samples ($N_{train}$) | Testing Samples ($N_{test}$) | MAE | Learned Slope ($\beta_1$) | Intercept ($\beta_0$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 40 | 10 | **3.6115** | 2.4156 | 1.2454 |
| **0.25** | 37 | 13 | **3.6176** | 2.4250 | 1.2489 |
| **0.30** | 35 | 15 | **3.7512** | 2.4379 | 0.8095 |
| **0.40** | 30 | 20 | **3.9307** | 2.4198 | 0.7644 |

### Observations & Key Learnings
1. As the test split size increases from 0.20 to 0.40, the number of training samples drops from 40 to 30, and test MAE increases from 3.6115 to 3.9307.
2. Smaller training sets yield slightly higher generalization error due to reduced data diversity for parameter estimation.
3. Tracking these variations manually across multiple iterations quickly becomes tedious and error-prone, demonstrating the necessity of MLOps tracking tools like MLflow.

---

## 2. Exercise 2: Delivery Time Predictor
### Objective
Train a linear regression model on the official delivery dataset:
- Distance $X$ (km): 1, 2, 3, 4, 5
- Actual Delivery Time $Y$ (min): 16, 19, 27, 29, 34

### Execution Command
```bash
python delivery_time_predictor.py
```

### Genuine Execution Output
- **Number of training observations:** 4
- **Number of testing observations:** 1
- **Learned Model:** $\text{Predicted Time} = 4.4000 \times \text{Distance} + 12.2000$
- **Test Sample Actual Time:** 19.00 min
- **Test Sample Predicted Time:** 21.00 min
- **Mean Absolute Error (MAE):** 2.0000 min

### Observations & Key Learnings
- The delivery predictor reveals a strong linear correlation between distance and time, with a base overhead of 12.2 minutes (preparation/dispatch) and ~4.4 minutes per km traveled.
- With only 5 total samples, splitting data leaves very few validation samples, highlighting the importance of standardized metrics and version control for production models.
