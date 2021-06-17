
from eq_odds import EqOddsModel
from tradeoff_plots import readData, select_best_cv
from tradeoff_plots import get_bootstrap_sample, sample_scores
from tradeoff_plots import get_score, plot_scatter_sign

from sklearn.linear_model import LogisticRegression
import numpy as np


csvfile = "data/SouthGermanCredit.csv"

# SOUTH GERMAN DATA SET
x_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
y_indices = [21]
z_indices = [13] # young = 0 older = 1

# COMPAS
# x_indices = [0,1,3,4,5,6,7]
# z_indices = [2]
# y_indices = [8]

util_scores = ["Accuracy"]
fairness_scores = ["Equal Opportunity"]
num_samples = 800

# csvfile = sys.argv[1]
# train_n = int(sys.argv[2])
# num_samples = int(sys.argv[3])
# penalty = sys.argv[4]

penalty = "none"
train_n = 900

print("USING PENALTY: %s"%penalty)

if penalty == "none":
    p = False
else:
    p = True
    C = [0.001, 0.01, 0.1, 1, 10, 100]

X_train, Z_train, Y_train, X_test, Z_test, Y_test, features = readData(csvfile, train_n, x_indices, y_indices, z_indices)

# find a policy -> a = 1 means give a loan, Y_pred = 1 means predict good credit

# without sensitive attributes
print("Training without sensitive attributes...")
if penalty == "l2":
    best_c = select_best_cv(C, X_train, Y_train)
    clf = LogisticRegression(penalty = penalty, C=best_c)
    clf.fit(X_train, Y_train)
else:
    clf = LogisticRegression(penalty = penalty)
    clf.fit(X_train,Y_train)



fair_clf = EqOddsModel(clf=clf, group_col=21, group_vals=[0, 1])
fair_clf.train(X_train, Y_train, Z_train)

print("Sampling...")
sample_util_metrics, sample_fair_metrics, sample_sign_metrics = sample_scores(X_test, Y_test, Z_test, fair_clf, num_samples, util_scores, fairness_scores)

util_scores = np.around(np.array(sample_util_metrics)[:,0],decimals=3)
fair_scores = np.around(np.array(sample_fair_metrics)[:,0], decimals=3)

fair_scores_pos = []
util_pos = []
fair_scores_neg = []
util_neg = []
for i,meas in enumerate(sample_sign_metrics):
    if len(meas[0]) > 0:
        fair_scores_pos.append(meas[0][0])
        util_pos.append(util_scores[i])
    elif len(meas[1]) > 0:
        fair_scores_neg.append(meas[1][0])
        util_neg.append(util_scores[i])

fair_scores_pos = np.around(np.array(fair_scores_pos),decimals=3)
fair_scores_neg = np.around(np.array(fair_scores_neg),decimals=3)

print("Plotting...")
plot_scatter_sign("Accuracy","Equal Odds",util_pos, fair_scores_pos, util_neg, fair_scores_neg,penalty = "None", sens=False)


# predicts, labels, new_z_test = fair_clf.predict_proba(X_test, Y_test, Z_test)
# print(predicts)
# print(labels)
# print(new_z_test)
