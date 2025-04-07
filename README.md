# synth-rebound-strength
   synth → Synthetic data  rebound → Rebound hammer method  strength → Concrete strength prediction

**Rebound Hammer Strength Predictor with Synthetic Data Augmentation**
A Python toolkit for predicting concrete compressive strength (f) from rebound hammer tests (R) when only minimal test data is available.

The primary objective of this code is to develop and compare predictive models for concrete compressive strength (denoted as 
f
f) using rebound hammer test data (denoted as 
R
R) under conditions of limited experimental data availability.

Key Goals:
Mitigate Small Training Data Limitations

Train models using only 3 real data points (simulating scarce field data).

Generate synthetic data clusters around these points to augment training.

Evaluate Model Performance

Compare three approaches:

Real Data Model: Trained on original 3-point data.

Synthetic Data Model: Trained purely on generated data.

Mixed Model: Trained on combined real + synthetic data.

Use fixed test sets (77 samples) for fair evaluation.

Optimize for Engineering Accuracy

Prioritize RMSE (Root Mean Squared Error) for model selection, as it penalizes large prediction errors critical in structural safety.

Report standard deviation of residuals to assess prediction consistency.

Ensure Reproducibility

Use fixed random seeds for test data (TEST_SEED = 42).

Allow variable training splits to test robustness.

link  colab https://colab.research.google.com/drive/16-H8f35_cB4swjnoOiLlpMS82Tg_w8cs#scrollTo=Xe_iopSyPWTV&line=4&uniqifier=1


