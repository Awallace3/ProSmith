import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_results():
    results = np.load("./plots/results.npy")
    y_true = results[:, 0]
    y_pred = results[:, 1]
    true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    false_negative = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negative}")
    # y_true += np.random.normal(0, 0.1, len(y_true))
    # y_pred += np.random.normal(0, 0.02, len(y_pred))
    # print(results)
    vals = len(y_true)
    correct = np.sum(np.equal(y_true, y_pred))
    binding_ligands_cnt_true = np.sum(y_true)
    binding_ligands_cnt_pred = np.sum(y_pred)
    percent_correct = correct / vals
    print(f"correct / total: {correct} / {vals}\nAccuracy: {percent_correct}")
    print(f"TRUE binding : non-binding = {binding_ligands_cnt_true} : {vals - binding_ligands_cnt_true}")
    print(f"PRED binding : non-binding = {binding_ligands_cnt_pred} : {vals - binding_ligands_cnt_pred}")
    # Plot ROC curve of the model
    display = metrics.RocCurveDisplay.from_predictions(
        y_true,
        y_pred,
    )
    display.plot()
    plt.savefig("./plots/ROC_curve.png", dpi=400)
    return

def example_ROC():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.metrics import RocCurveDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    clf = SVC(random_state=0).fit(X_train, y_train)
    y_pred = clf.decision_function(X_test)
    RocCurveDisplay.from_predictions(
       y_test, y_pred)
    plt.show()

def main():
    # example_ROC()
    plot_results()
    return


if __name__ == "__main__":
    main()
