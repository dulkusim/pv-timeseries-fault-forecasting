# Roadmap - Milestones

1. [x] **M1. Data Labeling**
2. [x] **M2. Repository Foundation**
3. [ ] **M3. Forecasting Code Restructure**
    - Replace the overlapping forecasting notebooks with one tested package under `scripts/`, so data preparation, windowing, the CS-LSTM and the XGBoost helpers are defined once and reused.
    - Leave thin notebooks that only configure and call that code, with centralized paths and a single shared Colab setup.
    - Enforce the chronological split (2015–2016 train, 2017 validation, 2018 test) in one place, so no stage can leak information across years.
4. [ ] **M4. Stage 1–2: CS-LSTM Regression and Fault Classification**
    - Train the CS-LSTM on healthy data only to predict expected AC power, and freeze it. Its residual (measured minus expected) becomes the anomaly signal for every later stage.
    - Train XGBoost on the residuals and engineered features for binary, multiclass and "fault impending" classification, with imbalance handled on training folds only.
    - Report regression and classification metrics on the validation and test years.
5. [ ] **M5. Stage 3: Fault Forecasting and Degradation Heads**
    - Build the fault-forecasting head: an HMM over Normal, Pre-fault and Fault states feeding an LSTM that predicts faults 1 h, 6 h and 24 h ahead.
    - Build the degradation head: STL on weather-normalized performance ratio, a Reliability Score, the degradation rate in %/year and a Remaining Useful Life estimate with confidence intervals.
    - Implement a small set of baselines for each head.
6. [ ] **M6. Evaluation, Baseline Comparison and Cross-Site Generalization**
    - Run every head against its baselines, plus ablations of the pipeline (no context, no HMM, no residuals), using the same splits and metrics for all.
    - Test how well the pipeline transfers to another installation (Ground or Roof) without retraining from scratch.
    - Validate the generated labels against the inverter-reported fault flags and report the agreement.
7. [ ] **M7. Publication Writing**
    - Decide the pipeline's name and logo.
    - Write the paper.

# Candidate Titles
Our pipeline has (keywords that we can use on title):
- **Context-aware:** the CS-LSTM uses season and time of day.
- **Residual-based:** the gap between measured and expected power drives every later stage.
- **Staged and hybrid:** CS-LSTM, then XGBoost, then HMM and LSTM.
- **Two horizons:** short-term fault forecasting and long-term degradation.

## Candidates

| Name | Expansion |
|---|---|
| **PV-CARE** | **C**ontext-**A**ware **R**esidual-based **E**valuation of PV faults and degradation |
| **PV-CRESTA** | **C**ontextual **RES**idual **T**wo-head **A**nalytics for PV |
| **PV-FFORESIGHT** | PV **F**ault **FOR**ecasting and **E**xplainable **S**tate **I**nference for **G**eneration **H**ealth **T**racking |
| **PV-HEAL** | **H**ybrid **E**xplainable **A**nomaly and **L**ifetime analysis for PV |
| **PV-F(F)DA** | PV **F**ault (**F**)/forecasting and **D**egradation **A**nalysis |

**PV**-**F**ault **FOR**casting and **SI**- **H**ealth **T**racking