from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd


def computeFeatureImportance(df_X, df_Y, model=None, scoring=None):
    if model is None:
        model = RandomForestClassifier(random_state=0)
    print("Compute feature importance using", model)
    model.fit(df_X, df_Y.squeeze())
    result = permutation_importance(model, df_X, df_Y,
            n_repeats=10, random_state=0, scoring=scoring)
    feat_names = df_X.columns.copy()
    feat_ims = np.array(result.importances_mean)
    sorted_ims_idx = np.argsort(feat_ims)[::-1]
    feat_names = feat_names[sorted_ims_idx]
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    df = pd.DataFrame()
    df["feature_importance"] = feat_ims
    df["feature_name"] = feat_names
    return df

def printScores(cv_results):
    """Print the cross-validation results"""
    print("\n================================================")
    print("Accuracy:", cv_results["test_accuracy"])
    print("f1-score micro:",cv_results["test_f1_micro"])
    
    print("Precision micro:", cv_results["test_precision_micro"])
    print("Recall macro:", cv_results["test_recall_macro"])
    print("================================================\n")
