import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import csv
import functools
import matplotlib.pyplot as plt
import random

random.seed(445)

### Reading in data ###
def getIndex(unique_list, observation):
    for i,u in enumerate(unique_list):
        if functools.reduce(lambda o1, o2 : o1 and o2, map(lambda x,y: x==y,u,observation), True):
            return i

def readData(csvfile, train_n, x_indices, y_indices, z_indices):

    with open(csvfile) as file:
        rows = csv.reader(file)

        data_pos = []
        data_neg = []
        features = []
        for row in rows:
            if row[-1] == 'kredit':
                for i in x_indices:
                    features.append(row[i])
                continue
            if int(row[-1]) == 1:
                data_pos.append(row)
            else:
                data_neg.append(row)


        random.shuffle(data_pos)
        random.shuffle(data_neg)

        X_train = []
        X_test = []
        Z_train = []
        Z_test = []
        Y_train = []
        Y_test = []
        count = 0

        train_prop = train_n/(len(data_pos)+len(data_neg))

        neg_n = int(train_prop*len(data_neg))
        pos_n = int(train_prop*len(data_pos))


        for j,row in enumerate(data_neg):
            if "kredit" in row:
                for i in x_indices:
                    features.append(row[i])
                continue

            #young vs old (under 25 vs. over)
            if (int(row[13])) <= 25: # young
                z = 0
            else: # old
                z = 1

            # black vs white
            # if (int(row[2])) == 1: # black
            #     z = 1
            # else: # white
            #     z = 0

            count+=1

            if j < neg_n:
                X_train.append([int(row[i]) for i in x_indices])
                Z_train.append([z])
                Y_train.append([int(row[i]) for i in y_indices][0])
            else:
                X_test.append([int(row[i]) for i in x_indices])
                Z_test.append([z])
                Y_test.append([int(row[i]) for i in y_indices][0])

        for j,row in enumerate(data_pos):
            if "kredit" in row:
                for i in x_indices:
                    features.append(row[i])
                continue

            #young vs old (under 25 vs. over)
            if (int(row[13])) <= 25: # young
                z = 0
            else: # old
                z = 1

            # black vs white
            # if (int(row[2])) == 1: # black
            #     z = 1
            # else: # white
            #     z = 0

            count+=1

            if j < pos_n:
                X_train.append([int(row[i]) for i in x_indices])
                Z_train.append([z])
                Y_train.append([int(row[i]) for i in y_indices][0])
            else:
                X_test.append([int(row[i]) for i in x_indices])
                Z_test.append([z])
                Y_test.append([int(row[i]) for i in y_indices][0])

        print(len(X_train))
        print(len(X_test))

        return X_train, Z_train, Y_train, X_test, Z_test, Y_test, features

def get_bootstrap_sample(data_x, data_z, data_label):

    n = len(data_x)
    sample_x = []
    sample_z = []
    sample_label = []
    for i in range(n):
        idx = np.random.randint(n)
        sample_x.append(data_x[idx])
        sample_z.append(data_z[idx])
        sample_label.append(data_label[idx])


    return sample_x, sample_z, sample_label

# calibrated
def get_calibration(Y, Z, A):

    a_count = [0, 0]
    y = [[0, 0],[0, 0]]
    z = [[0, 0, 0, 0], [0, 0, 0, 0]]
    yz = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    for i,a in enumerate(A):
        a_count[a] += 1
        y[a][Y[i][0]] += 1
        z[a][Z[i][0]-1] += 1
        yz[a][Y[i][0]][Z[i][0]-1] += 1

    for i in range(len(y[0])):
        y[0][i]/=a_count[0]
        y[1][i]/=a_count[1]

    for i in range(len(z[0])):
        z[0][i]/=a_count[0]
        z[1][i]/=a_count[1]

    for i in range(len(yz[0][0])):
        yz[0][0][i]/=a_count[0]
        yz[0][1][i]/=a_count[0]
        yz[1][0][i]/=a_count[1]
        yz[1][1][i]/=a_count[1]


    calibration_deviation = 0
    # P(y,z|a)-P(y|a)P(z|a)

    for a in range(len(a_count)):
        for y_val in range(len(y[0])):
            for z_val in range(len(z[0])):
                delta = yz[a][y_val][z_val] - y[a][y_val]*z[a][z_val]
                calibration_deviation += abs(delta)

    return calibration_deviation

# disparate mistreatment - assuming Z is binary
def measure_disparate_mistreatment(Y,Z,A):

    z_count = [0,0]
    for z in Z:
        z_count[z[0]] += 1

    misclassified = [0,0] # track misclassification for z = 1 and z = 0
    for i,a in enumerate(A):
        if a != Y[i][0]:
            misclassified[Z[i][0]] += 1

    disp_mistr = 0
    Pmis_z = []
    for i,count in enumerate(z_count):
        Pmis_z.append(misclassified[i]/count)

    return abs(Pmis_z[0]-Pmis_z[1])




# balanced
def get_balance(Y, Z, A):
    y_count = [0, 0]
    a = [[0, 0],[0, 0]]
    z = [[0, 0, 0, 0], [0, 0, 0, 0]]
    az = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    for i,y in enumerate(Y):
        y_count[y[0]] += 1
        a[y[0]][A[i]] += 1
        z[y[0]][Z[i][0]-1] += 1
        az[y[0]][A[i]][Z[i][0]-1] += 1

    for i in range(len(a[0])):
        a[0][i]/=y_count[0]
        a[1][i]/=y_count[1]

    for i in range(len(z[0])):
        z[0][i]/=y_count[0]
        z[1][i]/=y_count[1]

    for i in range(len(az[0][0])):
        az[0][0][i]/=y_count[0]
        az[0][1][i]/=y_count[0]
        az[1][0][i]/=y_count[1]
        az[1][1][i]/=y_count[1]


    balance_deviation = 0
    # P(y,z|a)-P(y|a)P(z|a)

    for y in range(len(y_count)):
        for a_val in range(len(a[0])):
            for z_val in range(len(z[0])):
                delta = az[y][a_val][z_val] - a[y][a_val]*z[y][z_val]
                balance_deviation += abs(delta)

    return balance_deviation

def measure_equal_oppY1(Y, Z, A):
    # P(a=1|z=0,y=1) = P(a=1|z=1,y=1)
    # probability of getting loan if person actually has good credit
    z0y1 = 0
    z1y1 = 0
    a1_1 = 0
    a1_0 = 0
    for i, z in enumerate(Z):
        if Y[i] == 1:
            if z == 0:
                z0y1 += 1
                if A[i] == 1:
                    a1_0 += 1
            elif z == 1:
                z1y1 += 1
                if A[i] == 1:
                    a1_1 += 1

    Pa1_z0y1 = a1_0/z0y1
    Pa1_z1y1 = a1_1/z1y1

    return Pa1_z1y1-Pa1_z0y1

def measure_equal_oppY0(Y, Z, A):
    # P(a=1|z=0,y=0) = P(a=1|z=1,y=0)
    # probability of getting loan if person actually has bad credit
    z0y0 = 0
    z1y0 = 0
    a1_1 = 0
    a1_0 = 0
    for i, z in enumerate(Z):
        if Y[i] == 0:
            if z[0] == 0:
                z0y0 += 1
                if A[i] == 1:
                    a1_0 += 1
            elif z[0] == 1:
                z1y0 += 1
                if A[i] == 1:
                    a1_1 += 1

    Pa1_z0y0 = a1_0/z0y0
    Pa1_z1y0 = a1_1/z1y0

    return Pa1_z1y0-Pa1_z0y0


def get_score(score, y_true, y_pred, y_proba, z):

    if score == "Accuracy":
        return accuracy_score(y_true, y_pred)
    elif score == "auroc":
        return roc_auc_score(y_true, y_proba)
    elif score == "calibration":
        return get_calibration(y_true, z, y_pred)
    elif score == "balance":
        return get_balance(y_true, z, y_pred)
    elif score == "disp mistreatment":
        return measure_disparate_mistreatment(y_true, z, y_pred)
    elif score == "Equal Opportunity":
        meas = measure_equal_oppY1(y_true, z, y_pred)
        return 1-abs(meas), meas > 0, meas < 0
    elif score == "equal opp0":
        meas = measure_equal_oppY0(y_true, z, y_pred)
        return 1-abs(meas), meas > 0, meas < 0

def sample_scores(X, Y, Z, clf, num_samples, util_scores,fairness_scores, with_sens = False):

    sample_util_metrics = []
    sample_fair_metrics = []
    sample_fair_metrics_sign = []
    for j in range(num_samples):
        samp_x, samp_z, samp_y = get_bootstrap_sample(X, Z, Y)
        if with_sens:
            samp_comb = []
            for i in range(len(samp_x)):
                samp_comb.append(samp_x[i] + samp_z[i])
            Y_pred = clf.predict(samp_comb)
            Y_prob = np.max(clf.predict_proba(samp_comb),axis=1)
        else:
            predicts, labels, new_z_test = clf.predict_proba(samp_x, samp_y, samp_z)
            #Y_pred = clf.predict(samp_x)
            #Y_prob = np.max(clf.predict_proba(samp_x),axis=1)
            Y_prob = predicts
            Y_pred = [round(p) for p in predicts]
            samp_z = new_z_test
            samp_y = labels

        util_results = []
        for score in util_scores:
            util_results.append(get_score(score, samp_y, Y_pred, Y_prob, samp_z))

        fair_results_pos = []
        fair_results_neg = []
        fair_results = []
        for score in fairness_scores:
            if score == "Equal Opportunity" or score == "equal opp0":
                meas, pos, neg = get_score(score,samp_y, Y_pred, Y_prob, samp_z)
                if pos:
                    fair_results_pos.append(meas)
                elif neg:
                    fair_results_neg.append(meas)

                fair_results.append(meas)

            else:
                fair_results.append(get_score(score, samp_y, Y_pred, Y_prob, samp_z))

        sample_util_metrics.append(util_results)
        sample_fair_metrics.append(fair_results)

        sample_fair_metrics_sign.append([fair_results_pos,fair_results_neg])

    return sample_util_metrics, sample_fair_metrics, sample_fair_metrics_sign

def plot_scatter(utility, fairness, utility_scores_with, fairness_scores_with, utility_scores_without, fairness_scores_without, with_fair, without_fair,with_orig_util, without_orig_util,penalty = False,sens = False):

    if penalty:
        filename = "TradeoffPlots/"+utility+"_"+fairness+"_penalty_COMPAS.png"
    else:
        filename = "TradeoffPlots/"+utility+"_"+fairness+"_no_penalty_COMPAS.png"

    title = "Utility/Fairness Scatter Plot"


    fig, ax = plt.subplots(1, figsize=(5,5))
    ax.scatter(fairness_scores_with, utility_scores_with, label = "Training with Sensitive")
    ax.scatter(fairness_scores_without, utility_scores_without, label = "Training without Sensitive")
    ax.scatter(with_fair, with_orig_util, label = "Orig with Sensitive", color = "yellow")
    ax.scatter(without_fair, without_orig_util,label = "Orig without Sensitive", color = "black")
    ax.set(xlabel = fairness, ylabel = utility)
    ax.set_title(title)
    ax.legend()

    plt.savefig(filename)

def plot_scatter_sign(utility, fairness, utility_scores_pos, fairness_scores_pos, utility_scores_neg, fairness_scores_neg, penalty=True, sens = False):
    if penalty:
        if sens:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_penalty_sens_SGC.png"
        else:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_penalty_no_sens_SGC.png"
    else:
        if sens:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_no_penalty_sens_SGC.png"
        else:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_no_penalty_no_sens_SGC.png"

    if sens:
        title = "Utility/Fairness Scatter Plot with Sensitive Training"
    else:
        title = "Utility/Fairness Scatter Plot without Sensitive Training"


    fig, ax = plt.subplots(1, figsize=(7,7))
    ax.scatter(fairness_scores_pos, utility_scores_pos, color = "orange", marker="d", label = "Unfair towards Protected Group", zorder=2)
    ax.scatter(fairness_scores_neg, utility_scores_neg, color = "blue", label = "Unfair towards Other Group", zorder=1)
    ax.set(xlabel = fairness, ylabel = utility)
    ax.set(xlim =(0.5,1), ylim=(0.5,1))
    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.rc('legend', prop={"size":8})

    #ax.set_title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.legend(prop={'size':18})

    plt.savefig(filename)



def plot_histogram(utility, fairness, utility_scores, fairness_scores, muF, muU, sigmaF, sigmaU,penalty=False,sens=False):

    if penalty:
        if sens:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_penaltyHistogram_sens_COMPAS.png"
        else:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_penaltyHistogram_no_sens_COMPAS.png"
    else:
        if sens:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_no_penalty_sens_HistogramCOMPAS.png"
        else:
            filename = "TradeoffPlots/"+utility+"_"+fairness+"_no_penalty_no_sens_HistogramCOMPAS.png"

    if sens:
        title = "Utility/Fairness Variance Comparison with Sensitive Training"
    else:
        title = "Utility/Fairness Variance Comparison without Sensitive Training"

    fig, ax = plt.subplots(1, figsize=(5,5))

    _,bins,_ = ax.hist(fairness_scores, bins = 50, label="Fairness Scores")
    ax.plot(bins, 3/(sigmaF * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - muF)**2 / (2 * sigmaF**2) ), linewidth=3, color='y')
    _,bins,_ = ax.hist(utility_scores, bins = 50, label="Utility Scores")
    ax.plot(bins, 2/(sigmaU * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - muU)**2 / (2 * sigmaU**2) ), linewidth=3, color='g')
    ax.set(xlabel= fairness, ylabel = "Number of Occurances")
    ax.set_title("Utility/Fairness Variance Comparison")
    ax.legend()

    plt.savefig(filename)


def cross_validate(clf, X, Y):

    skf = StratifiedKFold(n_splits=5)
    scores = []
    # For each split in the k folds...
    for train, test in skf.split(X,Y):
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        # Fit the data to the training data...
        clf.fit(X_train, Y_train)
        # And test on the ith fold.
        Y_pred = clf.predict(X_test)
        score = accuracy_score(Y_test, Y_pred)
        if not np.isnan(score):
            scores.append(score)
    # Return the average performance across all fold splits.
    return np.array(scores).mean()


def select_best_cv(C,X,Y):

    best_c = 0.001
    best_c_val = 0
    for c in C:
        clf = LogisticRegression(penalty="l2", C=c)

        avg_score = cross_validate(clf, np.array(X),np.array(Y))

        print("C: %f"%c)
        print("Score: %f"%avg_score)

        if avg_score > best_c_val:
            best_c_val = avg_score
            best_c = c

    return best_c





if __name__ == "__main__":

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

    csvfile = sys.argv[1]
    train_n = int(sys.argv[2])
    num_samples = int(sys.argv[3])
    penalty = sys.argv[4]

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

    weights = clf.coef_[0]

    print("Feature weights:")
    for i,f in enumerate(features):
        print("Feature: %s weight: %f"%(f,weights[i]))

    y_pred = clf.predict(X_test)
    y_prob = np.max(clf.predict_proba(X_test), axis = 1)

    without_orig = []
    for score in util_scores+fairness_scores:
        if score == "Equal Opportunity":
            meas,_,_ = get_score(score, Y_test, y_pred, y_prob, Z_test)
        else:
            meas = get_score(score, Y_test, y_pred, y_prob, Z_test)
        print(score+": "+str(meas))
        without_orig.append(meas)

    sample_util_metrics_without, sample_fair_metrics_without, sample_sign_metrics_without = sample_scores(X_test, Y_test, Z_test, clf, num_samples, util_scores, fairness_scores)

    # with sensitive attributes
    print("Training with sensitive attributes...")
    x_train_comb = []
    for i in range(len(X_train)):
        x_train_comb.append(X_train[i] + Z_train[i])

    if penalty == "l2":
        best_c = select_best_cv(C, x_train_comb, Y_train)
        clf = LogisticRegression(penalty = penalty, C=best_c)
        clf.fit(x_train_comb, Y_train)
    else:
        clf = LogisticRegression(penalty = penalty)
        clf.fit(x_train_comb, Y_train)


    weights = clf.coef_[0]

    print("Feature weights:")
    for i,f in enumerate(features):
        print("Feature: %s weight: %f"%(f,weights[i]))

    print("Sensitive weight: %f"%weights[-1])


    x_test_comb = []
    for i in range(len(X_test)):
        x_test_comb.append(X_test[i]+Z_test[i])

    y_pred = clf.predict(x_test_comb)
    y_prob = np.max(clf.predict_proba(x_test_comb), axis = 1)

    with_orig = []
    for score in util_scores+fairness_scores:
        if score == "Equal Opportunity":
            meas,_,_ = get_score(score, Y_test, y_pred, y_prob, Z_test)
        else:
            meas = get_score(score, Y_test, y_pred, y_prob, Z_test)
        print(score+": "+str(meas))
        with_orig.append(meas)

    sample_util_metrics_with, sample_fair_metrics_with, sample_sign_metrics_with  = sample_scores(X_test, Y_test, Z_test, clf, num_samples, util_scores, fairness_scores, with_sens=True)

    # plotting
    for u,util in enumerate(util_scores):
        for f,fair in enumerate(fairness_scores):

            util_scores_with = np.around(np.array(sample_util_metrics_with)[:,u], decimals=3)
            util_scores_without = np.around(np.array(sample_util_metrics_without)[:,u],decimals=3)
            fair_scores_with = np.around(np.array(sample_fair_metrics_with)[:,f], decimals=3)
            fair_scores_without = np.around(np.array(sample_fair_metrics_without)[:,f], decimals=3)


            fair_scores_with_pos = []
            util_with_pos = []
            fair_scores_with_neg = []
            util_with_neg = []
            fair_scores_without_pos = []
            util_without_pos = []
            fair_scores_without_neg = []
            util_without_neg = []
            for i,meas in enumerate(sample_sign_metrics_with):
                if len(meas[0]) > 0:
                    fair_scores_with_pos.append(meas[0][0])
                    util_with_pos.append(util_scores_with[i])
                elif len(meas[1]) > 0:
                    fair_scores_with_neg.append(meas[1][0])
                    util_with_neg.append(util_scores_with[i])

                if len(sample_sign_metrics_without[i][0]) > 0:
                    fair_scores_without_pos.append(sample_sign_metrics_without[i][0][0])
                    util_without_pos.append(util_scores_without[i])
                elif len(sample_sign_metrics_without[i][1]) > 0:
                    fair_scores_without_neg.append(sample_sign_metrics_without[i][1][0])
                    util_without_neg.append(util_scores_without[i])

            fair_scores_with_pos = np.around(np.array(fair_scores_with_pos),decimals=3)
            fair_scores_with_neg = np.around(np.array(fair_scores_with_neg),decimals=3)
            fair_scores_without_pos = np.around(np.array(fair_scores_without_pos),decimals=3)
            fair_scores_without_neg = np.around(np.array(fair_scores_without_neg),decimals=3)

            # rescaling
            # fair_scores_with = [(f+1)/2 for f in fair_scores_with]
            # fair_scores_without = [(f+1)/2 for f in fair_scores_without]

            muFW = np.mean(fair_scores_with)
            muFWO = np.mean(fair_scores_without)
            sigmaFW = np.std(fair_scores_with)
            sigmaFWO = np.std(fair_scores_without)

            muUW = np.mean(util_scores_with)
            muUWO = np.mean(util_scores_without)
            sigmaUW = np.std(util_scores_with)
            sigmaUWO = np.std(util_scores_without)

            print("Covariance Matrix for %s and %s with sensitive training"%(util,fair))
            print("Mean of utility: %f"%(np.mean(util_scores_with)))
            print("Mean of fairness: %f"%(np.mean(fair_scores_with)))
            print("Variance of utility: %f"%(np.std(util_scores_with)**2))
            print("Variance of fairness: %f"%(np.std(fair_scores_with)**2))
            print(np.cov([util_scores_with,fair_scores_with]))

            print("Covariance Matrix for %s and %s without sensitive training"%(util,fair))
            print("Mean of utility: %f"%(np.mean(util_scores_without)))
            print("Mean of fairness: %f"%(np.mean(fair_scores_without)))
            print("Variance of utility: %f"%(np.std(util_scores_without)**2))
            print("Variance of fairness: %f"%(np.std(fair_scores_without)**2))
            print(np.cov([util_scores_without,fair_scores_without]))

            if util == "auroc":
                with_orig_util = with_orig[1]
                without_orig_util = without_orig[1]

            else:
                with_orig_util = with_orig[0]
                without_orig_util = without_orig[0]

            with_fair = with_orig[1]
            without_fair = without_orig[1]

            #plot_scatter(util,fair,util_scores_with, fair_scores_with, util_scores_without, fair_scores_without, with_fair, without_fair, with_orig_util, without_orig_util, penalty = p)

            plot_scatter_sign(util,fair,util_with_pos, fair_scores_with_pos, util_with_neg, fair_scores_with_neg,penalty = p, sens = True)
            plot_scatter_sign(util,fair,util_without_pos, fair_scores_without_pos, util_without_neg, fair_scores_without_neg, penalty = p, sens = False)

            #plot_histogram(util, fair, util_scores_with, fair_scores_with, muFW, muUW,sigmaFW,sigmaUW,penalty=p,sens=True)
            #plot_histogram(util, fair, util_scores_without, fair_scores_without, muFWO, muUWO,sigmaFWO,sigmaUWO,penalty=p,sens=False)
