"""Validate the submission's inputs and notebook structure without training models."""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'


def main():
    for name in ('model_decision_tree.ipynb', 'model_neural_network.ipynb'):
        notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
        code = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        ast.parse(code, filename=name)
        errors = [o for c in notebook['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'error']
        assert not errors, f'{name}: saved execution errors'
        assert 'EXPLAIN_INDEX = 57608' in code, f'{name}: unexpected reference index'
        assert 'stratify=y, random_state=SEED' in code, f'{name}: check shared split'
        print(f'PASS: {name} syntax and saved outputs')
    df = pd.read_csv(DATA)
    assert df.shape == (70692, 22), df.shape
    assert not df.isna().any().any(), 'Unexpected missing values'
    assert not df.columns.duplicated().any(), 'Duplicate feature names'
    X = df.drop(columns='Diabetes_binary')
    y = df['Diabetes_binary'].astype(int)
    assert y.value_counts().to_dict() == {0: 35346, 1: 35346}
    train, test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert len(test) == 14139 and y_test.sum() == 7069
    assert 57608 in test.index and 57608 not in train.index
    expected = {'Sex': 0, 'Age': 10, 'BMI': 53, 'GenHlth': 4, 'HighBP': 1, 'HighChol': 1}
    assert test.loc[57608, list(expected)].to_dict() == expected
    assert y_test.loc[57608] == 1
    print('PASS: dataset, shared split, held-out patient and reported values')
    print('Dataset SHA-256:', hashlib.sha256(DATA.read_bytes()).hexdigest())
    print('This check does not train the models or validate new explanation results.')


if __name__ == '__main__':
    main()
