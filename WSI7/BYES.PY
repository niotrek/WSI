import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0)
            self.priors_[c] = X_c.shape[0] / X.shape[0]
        
        return self
    
    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    
    def _predict_single(self, x):
        posteriors = []
        
        for c in self.classes_:
            prior = np.log(self.priors_[c])
            conditional = np.sum(np.log(self._gaussian_likelihood(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        
        return self.classes_[np.argmax(posteriors)]
    
    def _gaussian_likelihood(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Wczytanie danych
iris = load_iris()
X, y = iris.data, iris.target

# Klasyfikator Gaussowskiego Naiwnego Bayesa
gnb = GaussianNaiveBayes()
skf = StratifiedKFold(n_splits=5)

# Funkcje do obliczania miar jakości
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

# Funkcja do obliczania wyników
def get_results(classifier, X, y, scorers, skf):
    results = {}
    for score_name, scorer in scorers.items():
        scores = cross_val_score(classifier, X, y, cv=skf, scoring=scorer)
        results[score_name] = scores
    return results

# Random state values
random_states = [42, 24, 12, 7, 3]

# Wyniki dla Gaussian Naive Bayes
gnb_results_all = {metric: [] for metric in scorers}
dt_results_all = {metric: [] for metric in scorers}
svm_results_all = {metric: [] for metric in scorers}

# Klasyfikatory do porównania
dt = DecisionTreeClassifier()
svm = SVC(kernel='linear')

# Obliczanie wyników dla każdego random_state
for random_state in random_states:
    skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    
    gnb_results = get_results(gnb, X, y, scorers, skf)
    dt_results = get_results(dt, X, y, scorers, skf)
    svm_results = get_results(svm, X, y, scorers, skf)
    
    for metric in scorers:
        gnb_results_all[metric].append(gnb_results[metric])
        dt_results_all[metric].append(dt_results[metric])
        svm_results_all[metric].append(svm_results[metric])

# Obliczanie średnich i odchylenia standardowego dla każdej miary
gnb_mean_std = {metric: (np.mean([np.mean(rs) for rs in gnb_results_all[metric]]), 
                         np.mean([np.std(rs) for rs in gnb_results_all[metric]])) for metric in scorers}
dt_mean_std = {metric: (np.mean([np.mean(rs) for rs in dt_results_all[metric]]), 
                        np.mean([np.std(rs) for rs in dt_results_all[metric]])) for metric in scorers}
svm_mean_std = {metric: (np.mean([np.mean(rs) for rs in svm_results_all[metric]]), 
                         np.mean([np.std(rs) for rs in svm_results_all[metric]])) for metric in scorers}

# Przygotowanie tabeli wyników
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Gaussian Naive Bayes': [f"{gnb_mean_std[metric][0]:.3f} ± {gnb_mean_std[metric][1]:.3f}" for metric in scorers],
    'Decision Tree': [f"{dt_mean_std[metric][0]:.3f} ± {dt_mean_std[metric][1]:.3f}" for metric in scorers],
    'SVM': [f"{svm_mean_std[metric][0]:.3f} ± {svm_mean_std[metric][1]:.3f}" for metric in scorers]
})

# Wyświetlenie wyników w formacie tekstowym
print("Gaussian Naive Bayes Results:")
for metric, (mean, std) in gnb_mean_std.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")

print("\nDecision Tree Results:")
for metric, (mean, std) in dt_mean_std.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")

print("\nSVM Results:")
for metric, (mean, std) in svm_mean_std.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")

# Eksport do LaTeX
latex_table = results_df.to_latex(index=False)

# Zapis do pliku LaTeX
with open('results.tex', 'w') as f:
    f.write(latex_table)