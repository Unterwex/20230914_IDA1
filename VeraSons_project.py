################ Vera Sons - IDA 1 - 14.09.2023 - Project ##################


################## Import Libraries ###################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns #for heatmap for confusion matrix
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler #normalization


from sklearn.model_selection import GridSearchCV #hyperparameter tuning with cross validation


############## Read and Preprocess Data ###############
def preprocess_data(feature_names):
    """ Read Data + throw out confusing features + deal with missing values + normalize numeric values
    
    ("other" value would maybe appear too often and would be too significant then)
    all values that appear in less than 10% of the data is set to NaN and treated similar to the missing values
    to be able to separate by existing income afterwards we split the dataset and combine it after the modification again.
    Otherwise the missing income values would be overwritten

    some values don't appear in the training dataset and/or are too seldom to be significant:
    print(data["Country_of_birth"].value_counts()) #Hungary,Holand-Netherlands don't appear in 5K data
    print(data["Employment_type"].value_counts())  #Never-worked doesn't appear in 5K data
    """


    ############## Read Data ################
    data = pd.read_csv("einkommen_train.csv", names=feature_names)

    ### get impression of original data
    data.info()
    print(data.head())

    ############ Preprocess Data ############

    ### feature unselection ###
    #weighting_factor = data.Weighting_factor
    data = data.drop("Weighting_factor",axis=1) #almost 1:1 mapping of rows and values


    ### missing values ###
    data = data.replace(to_replace=["\?"], value=np.nan, regex=True)


    ### replace missing or seldom values by mean or most frequent value ###
    income = data.Income
    data = data.drop("Income", axis=1)

    replacements = {}
    for column in data:
        counts = data[column].value_counts(normalize=True)

        for index,value in counts.items():
            if value < 0.01:
                data = data.replace({column: {index:np.nan}}, regex=True)
        if data[column].dtype == object:
            replacements[column] = data[column].mode()[0]
        else:
            replacements[column] = int(data[column].mean())

    data = data.fillna(replacements)


    ############# Normalization ##############

    normalizer = StandardScaler()
    
    numeric=data._get_numeric_data()

    for col in numeric:
        x_array = data[col].values.reshape(-1,1)
        data[col] = normalizer.fit_transform(x_array)

    vis_data(data, income)

    
    data = pd.get_dummies(data) #for non-numeric numbers create column for every entry and set values to True or False

    data["Income"] = income
    #data["Weighting_factor"] = weighting_factor #if weighting_factor should be included unnormalized (also uncomment line 51)
    data = data.replace({"Income": {' <=50K':-1 , ' >50K':1}}, regex=True)

    #print(data)

    return data



############## Visualize original Data ################
def vis_data(data, income):
    """ Visualize all attributes in comparison to Income """

    """
    print("Number of ? in each column:\n")
    for name,values in data.iteritems():
        q=data[name].astype(str).str.count('\?').sum() #question marks in columns
        print(name,": ",q)"""
    
        
    for name,values in data.iteritems():
        crosstab = pd.crosstab(data[name], income)
        crosstab.plot(kind='bar')
        plt.title("Ratio "+name+" - Income", fontsize=26)
        plt.xlabel(name, fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)


#################### Split Data #######################
def split_data(data):
    """ Split data into a learning set (training+testing), a training and a testing set
        and a set that the final model is applied on (application)"""

    #5000 samples
    data_learn_all = data.dropna(subset="Income")

    X_learn = data_learn_all.drop("Income",axis=1)
    
    y_learn = data_learn_all["Income"] #binary

    ### split into training data + testing data ###
    X_train, X_test, y_train, y_test = train_test_split(X_learn, y_learn, train_size=0.8, random_state=0)

    ### 25000 samples application data ###
    data_apply_all = data[data.Income.isna()]
    X_apply = data_apply_all.drop("Income",axis=1)

    
    return X_learn, y_learn, X_train, X_test, y_train, y_test, X_apply



################### Learn Model #######################
### Cross Validation to get accuracy and std ###
def learn_model(model_opt, params, X_learn, y_learn):
    """calculate the accuracy with cross validation and train the optimal model on all learning data"""
    scores = cross_val_score(model_opt, X_learn, y_learn, cv=10)
    #print("Scores of cross validation: ",scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    model = model_opt.fit(X_learn, y_learn)
    test_scores = model.score(X_learn, y_learn)
    print("Score on all learning data: ",test_scores)

    return model, scores.mean(), scores.std()



############### Tune Hyperparameters ################
def tune_hyperparams(clf, params, X_train, y_train):

    grid_search = GridSearchCV(clf, params, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_

    print("Optimal model and parameters:")
    print("Best parameters: ",best_params)

    return best_clf, best_params


########## Evaluate Model ############
def evaluate_model(clf, X_learn, y_learn, X_test, y_test, model_type):
    y_test_pred = clf.predict(X_test)

   

    ### Confusion Matrix ###
    conf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    print("\nConfusion matrix: ")
    print(conf_matrix)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_test_pred).ravel()
    print("True negative: ", tn)
    print("False positive: ", fp)
    print("False negative: ", fn)
    print("True positive: ", tp)

    
    plt.figure()
    ax= plt.subplot()
    ax=sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax, annot_kws={'size': 20});  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted income', fontsize=20)
    ax.set_ylabel('True income', fontsize=20) 
    ax.set_title('Confusion Matrix of '+model_type, fontsize=26)
    ax.xaxis.set_ticklabels(['<=50K','>50K'], fontsize=16)
    ax.yaxis.set_ticklabels(['<=50K','>50K'], fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    # recall
    #recall = tp / float(tp + fn)
    #print("Recall: ",recall)

    # precision
    #precision = tp / float(tp + fp)
    #print("Precision : ",precision)


    # classification report
    print("\nClassification report:")
    print(metrics.classification_report(y_test, y_test_pred, target_names=['<=50K','>50K']))


    """#This part plots ROC curves and Precision-Recall curves of each model for both classes individually
    ### ROC Curve ###
    #classes = clf.classes_
    #print(classes) #[-1: ' <=50K', 1: ' >50K']
    probabilities = clf.predict_proba(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities[:,1]) #>50K
    roc_auc = metrics.roc_auc_score(y_test, probabilities[:,1])
    #print("Roc AUC >50K: ",roc_auc)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_type, pos_label = '>50K')
    display.plot()
    plt.title('ROC Curve for Incomes >50K for '+model_type, fontsize=26)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


    fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities[:,0]) #<=50K
    roc_auc = metrics.auc(fpr, tpr)
    #print("Roc AUC <=50K: ",roc_auc)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_type, pos_label = '<=50K')
    display.plot()
    plt.title('ROC Curve for Incomes <=50K for '+model_type, fontsize=26)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


    ### Precision - Recall Curve ###

    #precision, recall, thresholds = metrics.precision_recall_curve(y_test, probabilities[:,1]) #>50K
    metrics.PrecisionRecallDisplay.from_predictions(y_true=y_test, y_pred=probabilities[:,1],name = '>50K', drawstyle="default")
    plt.title('Precision-Recall Curve for Incomes >50K for '+model_type, fontsize=26)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    
    metrics.PrecisionRecallDisplay.from_predictions(y_true=y_test, y_pred=probabilities[:,0],name = '<=50K', drawstyle="default")
    plt.title('Precision-Recall Curve for Incomes <=50K for '+model_type, fontsize=26)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)"""


def compare_model_evaluation(models, X_test, y_test, model_types):

    plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        probabilities = models[i].predict_proba(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities[:,1]) #>50K
        roc_auc = metrics.roc_auc_score(y_test, probabilities[:,1])

        label = f"Model: {model_types[i]} (AUC = {roc_auc:.3f})"
        plt.plot(fpr, tpr, lw=2, label=label)

    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('ROC Curves for Multiple Models for Incomes >50K', fontsize=26)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot()


    plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        probabilities = models[i].predict_proba(X_test)

        precision, recall, _ = metrics.precision_recall_curve(y_test, probabilities[:,1])

        avg_p = metrics.average_precision_score(y_test, probabilities[:,1])
        
        plt.plot(recall, precision, lw=2, label=model_types[i]+f" (Avg-P = {avg_p:.3f})")

    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('Precision-Recall Curves for Multiple Models for Incomes >50K', fontsize=26)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot()




def predict_25K(clf, X_apply, model_type):
    y_apply = clf.predict(X_apply)
    #print("Predicted values:\n",y_apply)

    poor_count = np.count_nonzero(y_apply == -1)
    rich_count = np.count_nonzero(y_apply == 1)

    plt.figure()
    y_classes = ["<=50K", ">50K"]
    compare = [poor_count,rich_count]

    plt.bar(y_classes, compare)

    #write values into bar
    for i, value in enumerate(compare):
        plt.annotate(value, (y_classes[i], value), ha='center', va='bottom', fontsize=20)
    
    plt.xlabel('Income distribution', fontsize=20)
    plt.ylabel('Number of people', fontsize=20)
    plt.title(model_type+' prediction of income distribution of 25000 people', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    return y_apply, poor_count, rich_count


def main():
    start_time = time.time()

    feature_names= ['Age', 'Employment_type', 'Weighting_factor', 'Level_of_education', 'Schooling_period', 'Marital_status', 'Employment_area', 'Partnership', 'Ethnicity', 'Gender', 'Gains_on_financial_assets', 'Losses_on_financial_assets', 'Weekly_working_time', 'Country_of_birth', 'Income']


    data = preprocess_data(feature_names)


    X_learn, y_learn, X_train, X_test, y_train, y_test, X_apply = split_data(data)
    
    models = []
    model_types = []
    model_scores = []
    model_pred = []
    
    ############ Logistic Regression ############
    print("\n\nLogistic Regression:\n")

    hyperparameters = {
        'penalty': ['l1', 'l2'],  #regularization
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],  #lambda -> how much is the regularizer used? larger C means less regularization -> overfitting
        'solver': ['liblinear'],  #algorithm for optimization, liblinear uses coordinate descent and saga stochastic avg gradient descent but saga reaches max_iter and all other options don't support both regularizations
        'max_iter': [100, 200, 300],
        'random_state': [5],  #random seed
    }
    lr_start_time = time.time()
    
    clf = LogisticRegression()

    lr_model_opt, lr_params = tune_hyperparams(clf, hyperparameters, X_train, y_train)

    lr_model, lr_score, lr_std = learn_model(lr_model_opt, lr_params, X_learn, y_learn)

    evaluate_model(lr_model, X_learn, y_learn, X_test, y_test, "Logistic Regression")
    
    print("Time for learning logistic regression model:", time.time() - lr_start_time, "seconds")


    lr_y_pred, lr_y_pred_poor, lr_y_pred_rich = predict_25K(lr_model, X_apply, "Logistic Regression")

    models.append(lr_model)
    model_types.append("Logistic Regression")
    model_scores.append(lr_score)
    model_pred.append(lr_y_pred)
    
    
    ############ Random Forest ############
    print("\n\nRandom Forest:\n")

    hyperparameters = { 'n_estimators': [50, 100, 200], #how many decision trees are taken into account
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [4, 8, 12],
                        'min_samples_leaf': [2, 4, 6],}

    rf_start_time = time.time()
    
    clf = RandomForestClassifier()
    rf_model_opt, rf_params = tune_hyperparams(clf, hyperparameters, X_train, y_train)

    rf_model, rf_score, rf_std = learn_model(rf_model_opt, rf_params, X_learn, y_learn)

    evaluate_model(rf_model, X_learn, y_learn, X_test, y_test, "Random Forest")

    print("Time for learning random forest model:", time.time() - rf_start_time, "seconds")

    rf_y_pred, rf_y_pred_poor, rf_y_pred_rich = predict_25K(rf_model, X_apply, "Random Forest")


    ##### feature importance #####
    importances = rf_model.feature_importances_
    rf_importances = pd.Series(importances, index=X_learn.columns).sort_values(ascending=False)
    print("Most important features:\n",rf_importances)

    fig, ax = plt.subplots()
    rf_importances.plot.bar(ax=ax)
    ax.set_title("Feature Importances", fontsize=26)
    ax.set_ylabel("Mean decrease in impurity", fontsize=16)
    fig.tight_layout()
    
    
    models.append(rf_model)
    model_types.append("Random Forest")
    model_scores.append(rf_score)
    model_pred.append(rf_y_pred)
    

    ############ Decision Tree ############
    print("\n\nDecision Tree:\n")

    hyperparameters = {
        'criterion': ['gini', 'entropy'],  #separation criterium
        'splitter': ['best', 'random'],  #for best optimizes criterion, for random it doesn't care about criterion and just separates randomly
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [4, 8, 12],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 'log2', None],  #max number of features between which the splitting optimum is checked (depends on total number of features which is the case for None)
        'random_state': [5],  #random seed
    }

    dt_start_time = time.time()
    
    clf = tree.DecisionTreeClassifier()

    dt_model_opt, dt_params = tune_hyperparams(clf, hyperparameters, X_train, y_train)

    dt_model, dt_score, dt_std = learn_model(dt_model_opt, dt_params, X_learn, y_learn)
    
    evaluate_model(dt_model, X_learn, y_learn, X_test, y_test, "Decision Tree")

    print("Time for learning logistic regression model:", time.time() - dt_start_time, "seconds")

    dt_y_pred, dt_y_pred_poor, dt_y_pred_rich = predict_25K(dt_model, X_apply, "Decision Tree")

    models.append(dt_model)
    model_types.append("Decision Tree")
    model_scores.append(dt_score)
    model_pred.append(dt_y_pred)
    
    
    ############### Compare Models ################
    
    
    compare_model_evaluation(models, X_test, y_test, model_types)
    
    y_classes = ("<=50K", ">50K")
    compare = {f"Logistic regression (Acc: {lr_score:.3f})":(lr_y_pred_poor, lr_y_pred_rich), f"Random Forest (Acc: {rf_score:.3f})":(rf_y_pred_poor, rf_y_pred_rich), f"Decision Tree (Acc: {dt_score:.3f})":(dt_y_pred_poor, dt_y_pred_rich)}

    
    comp_income = pd.DataFrame(compare, index=y_classes)

    # plot the dataframe
    ax = comp_income.plot(kind='bar', figsize=(10, 6), rot=0, color=['orange', 'lightblue', 'brown'], fontsize=20)

    #write values into bar
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fontsize=16)
        
    ax.set_title('Comparison of prediction of income distribution of 25000 people of different models', fontsize=26)
    ax.set_xlabel("Income", fontsize=20)
    ax.set_ylabel("Number of people", fontsize=20)
    plt.legend(fontsize=16)


    #### count similar predictions ####
    rf_lr_dt = 0
    lr_rf = 0
    lr_dt = 0
    dt_rf = 0

    for index in range(len(lr_y_pred)):
        if lr_y_pred[index] == rf_y_pred[index] and lr_y_pred[index] == dt_y_pred[index]:
            rf_lr_dt += 1
        elif lr_y_pred[index] == rf_y_pred[index] and lr_y_pred[index] != dt_y_pred[index]:
            lr_rf += 1
        elif lr_y_pred[index] == dt_y_pred[index] and lr_y_pred[index] != rf_y_pred[index]:
            lr_dt +=1
        elif rf_y_pred[index] == dt_y_pred[index] and lr_y_pred[index] != rf_y_pred[index]:
            dt_rf +=1

    lr = len(X_apply) - rf_lr_dt - lr_rf - lr_dt
    rf = len(X_apply) - rf_lr_dt - lr_rf - dt_rf
    dt = len(X_apply) - rf_lr_dt - dt_rf - lr_dt

    plt.figure()
    #venn = venn3(subsets=(lr,rf,lr_rf,dt,lr_dt,dt_rf,rf_lr_dt), 
                 #set_labels=('Logistische Regression', 'Random Forest', 'Decision Tree'))
    #plt.title("Similarity in predictions of all models", fontsize=26)
    venn = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=('Logistische Regression', 'Random Forest', 'Decision Tree'))

    # Entferne den automatischen Text für die Mengengrößen
    venn.get_label_by_id('100').set_text(lr)
    venn.get_label_by_id('010').set_text(rf)
    venn.get_label_by_id('001').set_text(dt)

    # Beschrifte die Schnittmengen
    venn.get_label_by_id('110').set_text(lr_rf)
    venn.get_label_by_id('101').set_text(lr_dt)
    venn.get_label_by_id('011').set_text(dt_rf)
    venn.get_label_by_id('111').set_text(rf_lr_dt)

    plt.title("Similarity in predictions of all models", fontsize=26)


    for label in venn.set_labels:
        label.set_fontsize(20)
    for label in venn.subset_labels:
        label.set_fontsize(20)

    
    ############## return dataset ###################
    max_acc_index = model_scores.index(max(model_scores))
    count = 0
    for index,value in data["Income"].iteritems():
        if np.isnan(value):
            data.loc[data["Income"][index]] = model_pred[max_acc_index][count]
            count +=1

    
    #print(data)
    data.to_csv("pred_income.csv", index=False)
    
    print("Overall program time:", time.time() - start_time, "seconds")
    
    ############### Create all Plots ################
    plt.show()
    

main()
