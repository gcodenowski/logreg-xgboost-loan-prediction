# Critical Report – Machine Learning Project C07045 (Assignment 2)

**Gilbert Kozanowski**
School of Computer and Engineering Sciences
University of Chester, Chester, England, UK
gilbertkozanowski@gmail.com · 2421089@chester.ac.uk

---

## I. Introduction

Loan approval decisions are a high-stakes binary classification problem in which fairness, interpretability, and predictive accuracy are paramount. In this report, the target variable is approval status, while applicant characteristics form the predictor set. Two supervised learning approaches are examined: logistic regression, chosen as the baseline due to its transparent log-odds formulation (Szumilas, 2010; Kleinbaum & Klein, 2010), and XGBoost, selected for its capacity to capture non-linear feature interactions and deliver superior performance (Chen & Guestrin, 2016; Dietterich, 2000). The analysis balances predictive gains against auditability, with particular attention to protected attributes that may act as proxy variables (Barocas & Selbst, 2016).

## II. Machine Learning Approach Justification

### A. Model 1: Logistic Regression

Logistic regression was chosen as a baseline model because it models the log-odds of approval as a linear function of the input features, producing directly interpretable coefficients (Szumilas, 2010). This provides a view of how each feature increases or decreases the likelihood of approval (Kleinbaum & Klein, 2010). In a lending context, this interpretability is a necessary feature: it allows each predictor's contribution to the approval decision to be evaluated individually, assessing whether any feature disproportionately affects outcomes for a particular group (Hosmer et al., 2013). This transparency is valuable given the well-documented risk that seemingly neutral features can act as proxies for protected characteristics, producing indirect discrimination even when a sensitive attribute is excluded (Barocas & Selbst, 2016). Logistic regression, however, is not able to capture the non-linear interactions between features, which the next model chosen for the project does natively.

### B. Model 2: XGBoost

XGBoost (Chen & Guestrin, 2016) was chosen as a comparison model because, despite sharing a supervised classification objective, it learns through a fundamentally different mechanism — sequential ensembles of shallow trees, each correcting the errors of the last. This allows it to capture non-linear relationships and feature interactions. For example, a non-additive effect of income and age on the likelihood of loan approval, without those interactions being manually specified, as they would need to be in logistic regression. Ensemble methods of this kind typically achieve stronger predictive performance compared to a single linear model, for both statistical and representational reasons (Dietterich, 2000).

This advantage in performance comes at an interpretability cost. XGBoost, as a black box model, reports which variables split most frequently or reduce loss the most, but not the direction or predictive effect of a feature on the outcome. Recovering that requires additional tooling such as SHAP (Lundberg & Lee, 2017), which was not implemented here as it lies outside the project's scope. Consequently, model selection was treated less as a simple accuracy comparison and more as a trade-off between predictive performance and the auditability required in a lending setting, evaluated empirically in the Outcome Analysis section.

### C. Alternative Approaches

Two alternative supervised approaches were considered and rejected. Random Forest, another ensemble method, offers similar interpretability benefits through feature importance, but XGBoost's sequential correction mechanism typically achieves stronger predictive performance on structured tabular data of this scale (Shwartz-Ziv & Armon, 2022). Additionally, both methods would provide comparable interpretability limitations, making XGBoost the stronger choice within the ensemble paradigm. Fernández-Delgado et al. (2014) conducted a broad empirical comparison of classifiers across many datasets, finding that performance depends strongly on the problem and that no single classifier is best in all cases; due to the age of the study, XGBoost was not included.

SVMs were considered but ultimately excluded from the final model set due to project scope. While they can be calibrated to provide probability estimates, this adds complexity, and their computational cost can become burdensome depending on kernel choice and dataset characteristics (Platt, 1999).

## III. Data Analysis

### A. Exploratory Data Analysis (EDA)

The dataset presents with 13,734 rows, 7 columns, and a binary classification target. Upon performing EDA, visible issues in the dataset included ghost rows paired with sentinel values, and duplicate pairs of the ID column. For missing values, median data imputation has been chosen, due to its robustness to outliers in the data compared to mean imputation (Alam et al., 2023). Experimental methods such as decision tree imputation have been proven to perform better (Alam et al., 2023); however, median imputation has been chosen due to the project's scope. The dataset has been split into train/test sets before imputation to prevent leakage. Data has been normalised using the StandardScaler library, standardising numeric features to zero mean and unit variance. This is essential for logistic regression, where vastly different scales would cause slow convergence and make coefficients uninterpretable. This has been applied to both models for pipeline consistency.

### B. Feature Selection

When deciding on features necessary for training the models, two columns have been discarded — ID and Postcode. ID has been dismissed due to its non-predictive nature, while Postcode has been dropped due to high cardinality and ethical considerations (further discussed in section C). Features from the dataset chosen for model training ended up as the following: age, income, outgoings, gender. Including gender in the training raises ethical concerns (Centre for Data Ethics and Innovation [CDEI], 2020), which will be discussed further, but it was included to assess its contribution to model classification. Besides this, three additional features have been engineered in hope of improving the models' output metrics after initial tests — income/outgoings ratio, which captures disposable income needed for loan repayment (Finlay, 2006), income per age, and outgoings per age — which account for the fact that income and spending patterns can vary at different career stages. Division by zero was handled by replacing infinite values with the training set median, ensuring the model receives valid numeric inputs without introducing artificial patterns.

### C. Ethical Considerations

Mentioned in the preceding sections, two features in particular raise ethical concerns in the context of machine learning algorithms — postcode and gender. Postcode has been dropped from the dataset due to high cardinality, and due to the fact that geographic location can correlate with race, ethnicity and socioeconomic status (Barocas & Selbst, 2016), which are protected characteristics (Barocas & Selbst, 2016; CDEI, 2020). Including the postcode risks the model learning to discriminate based on where an applicant lived rather than their creditworthiness — a pattern historically associated with redlining in mortgage lending (Barocas & Selbst, 2016).

Gender was retained to allow its influence to be inspected directly via model coefficients/feature importance, rather than assuming it plays no role. This is not a production-ready justification: no disparate impact analysis (e.g. approval rate or false-negative rate by group) was conducted, which the Equality Act 2010, s. 29 would require before deployment. The ordinal encoding of gender in this project introduces an artificial ordering between categories that does not reflect reality. This may cause the model to infer a hierarchical relationship between gender groups. One-hot encoding would avoid this issue by treating each category as independent, but was not adopted here to limit feature space expansion. In a production lending system, this limitation would need to be addressed to prevent gender-based bias in approval decisions.

## IV. Outcome Analysis

### A. Metrics Evaluation

Classification results of both models were evaluated on a test set (20% of data, 2,500 samples), calculating five metrics: accuracy, precision, recall, F1-score (Figures 1 & 2), and AUC-ROC. Confusion matrices (Figures 6 & 7) and ROC curves (Figures 3 & 4) were generated to visualise model performance. When analysing the metrics, both models performed quite similarly, achieving an accuracy of 65–66% (Logistic Regression [LR] – 65.28%, XGBoost – 66.48%). Both models performed better than a random classifier (50%), though there is clear room for improvement. Precision was similar between models (LR – 0.687, XGBoost – 0.683), meaning that approximately 31–32% of predicted approvals were incorrect. In a production environment, this would pose a significant risk of, for example, approving applicants who may default. The ROC curves show XGBoost's curve outperforming LR's across most thresholds, consistent with its higher AUC.

**Logistic Regression Metrics**

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 65.28% |
| Precision | 0.6866 |
| Recall    | 0.7459 |
| F1-score  | 0.7150 |

*Fig. 1 – Logistic Regression Metrics*

<img src="outputs/metrics_table_logreg.png" width="400">

**XGBoost Metrics**

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 66.48% |
| Precision | 0.6827 |
| Recall    | 0.7959 |
| F1-score  | 0.7350 |

*Fig. 2 – XGBoost Metrics*

<img src="outputs/metrics_table_xgboost.png" width="400">

Recall showed the largest divergence: XGBoost achieved 0.796 compared to LR's 0.746. The F1-scores (LR: 0.715, XGBoost: 0.735) confirm XGBoost's superior balance between precision and recall. AUC-ROC scores (LR: 0.706, XGBoost: 0.722) indicate acceptable discrimination ability for both models, with XGBoost showing slightly better ranking performance.

<img src="outputs/roc_curve_logreg.png" width="400"> <img src="outputs/roc_curve_xgboost.png" width="400">

*Fig. 3 – AUC-ROC Curve (Logistic Regression) · Fig. 4 – AUC-ROC Curve (XGBoost)*

### B. Visualisation Analysis

The confusion matrices reveal that XGBoost produces fewer false negatives, but slightly more false positives than LR. The feature importance plot for XGBoost reveals a critical ethical concern: gender emerged as one of the most important features (Figure 5). This suggests the model has learned to associate gender with loan approval decisions, which introduces discriminatory bias. While including gender improved predictive accuracy, it should not be used in production systems due to fairness and legal implications (Barocas & Selbst, 2016; CDEI, 2020). Future implementations should exclude gender entirely, accepting a potential reduction in accuracy to ensure equitable treatment. This finding validates the decision to retain gender for inspection rather than exclude it silently (Section II), as its influence would otherwise have gone undetected.

<img src="outputs/feature_importance_xgboost.png" width="500">

*Fig. 5 – Feature Importance for XGBoost*

<img src="outputs/confusion_matrix_logreg.png" width="400"> <img src="outputs/confusion_matrix_xgboost.png" width="400">

*Fig. 6 – Confusion Matrix (Logistic Regression) · Fig. 7 – Confusion Matrix (XGBoost)*

A pronounced gap emerges in recall: XGB attains 0.796 versus LR's 0.746 — a difference of 0.050 that exceeds the negligible margin seen in other metrics. This larger recall disparity indicates that XGB better captures the minority class (approved applicants), while LR more often mislabels approved cases as not approved.

The addition of three engineered ratios (income to outgoings, income per age, outgoings per age) produced negligible changes in both models: LR's accuracy dipped marginally, and XGB's metrics remained essentially unchanged. Thus, the original four predictors (age, income, outgoings, gender) already encapsulate most predictive signal.

Interpretability remains a decisive factor. LR, as a linear model, offers transparent coefficients and direct interpretability, whereas XGB is an effectively black-box ensemble. In regulated lending environments where auditability is mandatory, LR's transparency outweighs its modest performance deficit. Consequently, for deployment, LR would be a better choice, given a comprehensive fairness audit to mitigate the influence of gender and other proxy variables.

### C. Improvements

Several enhancements could improve model performance. First, hyperparameter tuning using `GridSearchCV` could optimise XGBoost's parameters (`max_depth`, etc.). Second, threshold adjustment could optimise the classification boundary for specific business objectives, such as minimising false negatives. Third, retraining XGBoost without gender and comparing resulting metrics would quantify the accuracy-fairness trade-off directly, informing whether its removal is justified for production use.

## IV. Conclusion

*(numbered as in the original)*

The comparative evaluation demonstrates that Logistic Regression (LR) and XGBoost (XGB) achieve broadly similar accuracy on the loan approval task (LR = 65.28%, XGB = 66.48%). However, the recall metric reveals a more pronounced gap: XGB attains 0.796 versus LR's 0.746 — a difference of 0.050 that exceeds the negligible margin seen in other metrics. This larger recall disparity indicates that XGB better captures the minority class (approved applicants) while LR more often mislabels approved cases as not approved.

The addition of three engineered ratios (income to outgoings, income per age, outgoings per age) produced negligible changes in both models: LR's accuracy dipped marginally, and XGB's metrics remained essentially unchanged. Thus, the original four predictors (age, income, outgoings, gender) already encapsulate most predictive signal.

Interpretability remains a decisive factor. LR, as a linear model, offers transparent coefficients and direct interpretability, whereas XGB is an effectively black-box ensemble. In regulated lending environments where auditability is mandatory, LR's transparency outweighs its modest performance deficit. Consequently, for deployment, LR would be a better choice, given a comprehensive fairness audit to mitigate the influence of gender and other proxy variables.

## References

- Alam, S., Ayub, M. S., Arora, S., & Khan, M. A. (2023). An investigation of the imputation techniques for missing values in ordinal data enhancing clustering and classification analysis validity. *Decision Analytics Journal*, *9*, 100341. https://doi.org/10.1016/j.dajour.2023.100341
- Barocas, S., & Selbst, A. D. (2016). Big Data's Disparate Impact. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.2477899
- Centre for Data Ethics and Innovation (CDEI). (2020, November 27). *Review into bias in algorithmic decision-making.* GOV.UK. https://www.gov.uk/government/publications/cdei-publishes-review-into-bias-in-algorithmic-decision-making/main-report-cdei-review-into-bias-in-algorithmic-decision-making
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. In G. Goos, J. Hartmanis, & J. Van Leeuwen (Eds.), *Multiple Classifier Systems* (Vol. 1857, pp. 1–15). Springer Berlin Heidelberg. https://doi.org/10.1007/3-540-45014-9_1
- Fernandez-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (n.d.). *Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?*
- Finlay, S. M. (2006). Predictive models of expenditure and over-indebtedness for assessing the affordability of new consumer credit applications. *Journal of the Operational Research Society*, *57*(6), 655–669. https://doi.org/10.1057/palgrave.jors.2602030
- GOV UK. (2010). *Equality Act 2010* [Text]. Statute Law Database. https://www.legislation.gov.uk/ukpga/2010/15/contents
- Hosmer, D. W., Jr., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression.* John Wiley & Sons, Incorporated. http://ebookcentral.proquest.com/lib/uocuk/detail.action?docID=1138225
- Kleinbaum, D. G., & Klein, M. (2010). *Logistic Regression.* Springer New York. https://doi.org/10.1007/978-1-4419-1742-3
- Lundberg, S. M., & Lee, S.-I. (n.d.). *A Unified Approach to Interpreting Model Predictions.*
- Shwartz-Ziv, R., & Armon, A. (2021). *Tabular Data: Deep Learning is Not All You Need* (arXiv:2106.03253). arXiv. https://doi.org/10.48550/arXiv.2106.03253
- Szumilas, M. (2010). Explaining Odds Ratios. *Journal of the Canadian Academy of Child and Adolescent Psychiatry*, *19*(3), 227–229.
