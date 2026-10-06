# Diabetes status classification — Decision tree

**Dataset:** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` from DTU Learn. Set `DATA_PATH` below to its local location. Run cells in order. Install dependencies if needed: `pip install numpy pandas scikit-learn matplotlib seaborn`.

The label `Diabetes_binary` denotes reported (pre)diabetes **at survey time**. The dataset was deliberately balanced 50/50. A predicted probability is therefore a model score for this sampled population, **not** an individual's future absolute risk. Feature contributions describe the model, not causal effects or medical advice. Keep the held-out test set untouched while choosing settings.



```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, average_precision_score, RocCurveDisplay,
                             brier_score_loss)
from sklearn.inspection import permutation_importance

```


```python
DATA_PATH = Path('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
SEED = 42
if not DATA_PATH.is_file():
    raise FileNotFoundError(f'Set DATA_PATH to the CSV downloaded from DTU Learn: {DATA_PATH.resolve()}')
df = pd.read_csv(DATA_PATH)
target = 'Diabetes_binary'
if target not in df:
    raise KeyError(f'Missing {target}; columns: {df.columns.tolist()}')
if df.columns.duplicated().any():
    raise ValueError('Duplicate column names')
X = df.drop(columns=target).apply(pd.to_numeric, errors='raise')
y = df[target].astype(int)
assert set(y.unique()) == {0, 1}, 'Expected binary label 0/1'
print('Rows, features:', X.shape, '| positive fraction:', round(y.mean(), 3))
print('Missing values:', int(X.isna().sum().sum()))
display(X.head())

# Identical split in both notebooks. Split before fitting any imputation/scaling.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
features = X.columns.tolist()

```

    Rows, features: (70692, 21) | positive fraction: 0.5
    Missing values: 0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>


## Train a shallow, readable decision tree

Imputation is fitted only on training data. `max_depth` and `min_samples_leaf` constrain complexity; these are baseline settings, not tuned on the test set.


```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree

model = make_pipeline(SimpleImputer(strategy='median'),
                      DecisionTreeClassifier(max_depth=4, min_samples_leaf=100,
                                             random_state=SEED))
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]
plt.figure(figsize=(20, 9))
plot_tree(model[-1], feature_names=features, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title('Decision tree (top four levels)'); plt.show()

```

## Held-out evaluation


```python
def evaluate(y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    print(classification_report(y_true, prediction, target_names=['no (pre)diabetes', '(pre)diabetes']))
    print(f'ROC AUC: {roc_auc_score(y_true, probability):.3f}')
    print(f'Average precision: {average_precision_score(y_true, probability):.3f}')
    print(f'Brier score (balanced sample only): {brier_score_loss(y_true, probability):.3f}')
    ConfusionMatrixDisplay.from_predictions(y_true, prediction, display_labels=['No', 'Yes'], cmap='Blues')
    plt.title('Held-out test set, threshold 0.5'); plt.show()
    RocCurveDisplay.from_predictions(y_true, probability); plt.show()

evaluate(y_test, probabilities)

```

## Global explanation: permutation importance


```python
# Global audit: drop in held-out ROC AUC after shuffling one feature.
# Correlated inputs may share importance; shuffling can produce unrealistic combinations.
sample = X_test.sample(n=min(2000, len(X_test)), random_state=SEED)
result = permutation_importance(model, sample, y_test.loc[sample.index],
                                scoring='roc_auc', n_repeats=5, random_state=SEED, n_jobs=1)
importance = pd.Series(result.importances_mean, index=features).sort_values()
importance.tail(12).plot.barh(figsize=(8, 5), xlabel='Mean decrease in ROC AUC')
plt.title('Permutation importance on held-out sample'); plt.tight_layout(); plt.show()
display(importance.sort_values(ascending=False).head(15).to_frame('mean_auc_drop'))

```

## Local explanation: exact decision path

The conditions below show why the tree assigned its leaf score. They do not estimate the effect of changing a feature.


```python
row_position = 0  # Change to inspect another held-out person
row = X_test.iloc[[row_position]]
filled = model[0].transform(row)
tree_model = model[-1]
node_ids = tree_model.decision_path(filled).indices
for node in node_ids:
    feature_index = tree_model.tree_.feature[node]
    if feature_index < 0:
        break
    value = filled[0, feature_index]
    limit = tree_model.tree_.threshold[node]
    relation = '<=' if value <= limit else '>'
    print(f'{features[feature_index]} = {value:g} {relation} {limit:.3g}')
print('Model score:', round(model.predict_proba(row)[0, 1], 3))
print('True survey label:', int(y_test.iloc[row_position]))

```

## Error audit

Review false negatives and false positives. Features such as `Age`, `Income`, and access to care may encode social patterns; differences in this dataset alone cannot establish fairness or clinical safety.


```python
audit = X_test.copy()
audit['actual'] = y_test
audit['score'] = probabilities
audit['predicted'] = (probabilities >= 0.5).astype(int)
display(audit.loc[(audit.actual == 1) & (audit.predicted == 0)].sort_values('score').head(10))
display(audit.loc[(audit.actual == 0) & (audit.predicted == 1)].sort_values('score', ascending=False).head(10))

```
