
from eq_odds import EqOddsModel
from tradeoff_plots import readData, select_best_cv

from sklearn.linear_model import LogisticRegression


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

predicts, labels, new_z_test = fair_clf.predict_proba(X_test, Y_test, Z_test)
print(predicts)
print(labels)
print(new_z_test)


