import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import csv
import matplotlib.pyplot as plt
import seaborn as sns


X_train = None
X_test = None
y_train = None
y_test = None
reduced_X_train = None
reduced_X_test = None


def zad1():
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test
    X_train = pd.read_csv('./Train/X_train.txt', sep=" ", header=None)
    X_test = pd.read_csv('./Test/X_test.txt', sep=" ", header=None)
    y_train = pd.read_csv('./Train/y_train.txt', sep=" ", header=None)
    y_test = pd.read_csv('./Test/y_test.txt', sep=" ", header=None)
    # print(X_train)
    # print(y_train)
    # print('-----')
    # print(y_test)

    #NORMALIZACJA W RAZIE CZEGO:
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)


def zad2():
    global reduced_X_train, reduced_X_test
    # X_train.shape
    # X_test.reshape(-1, 1)

    reduced_X_train = PCA(n_components=2).fit_transform(X_train)
    reduced_X_test = PCA(n_components=2).fit_transform(X_test)

    # reduced_X_train = PCA(n_components=2).fit_transform(X_train, y_train)
    # reduced_X_test = PCA(n_components=2).fit_transform(X_test, y_test)

    reduced_X_train.shape, y_train.shape
    reduced_X_test.shape, y_test.shape

    clf = svm.SVC(kernel='linear', C=1).fit(reduced_X_train, y_train)
    # clf.score(X_test, y_test)
    scores = cross_val_score(clf, reduced_X_test, y_test, cv=5)
    print(scores)

    # acc = accuracy_score(y_train, y_test)
    # print("ACC:")
    # print(acc)

    # print(y_train)
    # print('-----')
    # print(y_test)

def svm_():  # underscore to not override built in function
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test
    X = reduced_X_train
    y = y_train
    clf = svm.SVC()
    clf.fit(X, y.values.ravel())
    predicted = clf.predict(reduced_X_test)
    return predicted


def knn():
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test
    X = reduced_X_train
    y = y_train
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X, y.values.ravel())
    predicted = model.predict(reduced_X_test)
    return predicted


def decisionTree():
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test
    X = reduced_X_train
    y = y_train
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y.values.ravel())
    predicted = clf.predict(reduced_X_test)
    return predicted


def randomForest():
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test
    X = reduced_X_train
    y = y_train
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y.values.ravel())
    predicted = clf.predict(reduced_X_test)
    return predicted

def calculateACC(y, y_test):
    acc = accuracy_score(y, y_test)
    return acc


def calculateRecall(y, y_test):
    rec = recall_score(y, y_test, average='weighted')
    return rec


def calculateF1(y, y_test):
    f1 = f1_score(y, y_test, average='weighted')
    return f1


def calculateAUC(y, y_test):
    auc = roc_auc_score(y, y_test, multi_class='ovo')
    return auc

def zad3():
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test
    assignedLabels = []         # przypisane labele i ich liczniki, potrzebne w do podjęcia decyzji w razie remisu w głosowaniu
    assignedLabelsCount = []
    svmLabels = svm_()              # 0.3
    knnLabels = knn()               # 0.3
    dtLabels = decisionTree()       # 0.2
    rfLabels = randomForest()       # 0.2
    finalLabels = []
    length = len(svmLabels) # wszystkie powinny miec taką samą długość
    for i in range(length):
        labels = []
        votes = []
        # svm 0.3
        if svmLabels[i] in labels:  # jeśli już sie pojawiło to dodaj głos
            index = labels.index(svmLabels[i])
            votes[index] = votes[index] + 0.3
        else:   # jeśli jeszcze nie to wpisz do list
            labels.append(svmLabels[i])
            votes.append(0.3)
        # knn 0.3
        if knnLabels[i] in labels:  # jeśli już sie pojawiło to dodaj głos
            index = labels.index(knnLabels[i])
            votes[index] = votes[index] + 0.3
        else:   # jeśli jeszcze nie to wpisz do list
            labels.append(knnLabels[i])
            votes.append(0.3)
        # dt 0.2
        if dtLabels[i] in labels:  # jeśli już sie pojawiło to dodaj głos
            index = labels.index(dtLabels[i])
            votes[index] = votes[index] + 0.2
        else:   # jeśli jeszcze nie to wpisz do list
            labels.append(dtLabels[i])
            votes.append(0.2)
        # rf 0.2
        if rfLabels[i] in labels:  # jeśli już sie pojawiło to dodaj głos
            index = labels.index(rfLabels[i])
            votes[index] = votes[index] + 0.2
        else:   # jeśli jeszcze nie to wpisz do list
            labels.append(rfLabels[i])
            votes.append(0.2)

        ## sprawdzenie wyników głosowania
        bestLabelIndex = [] # jeśli na końcu długość tej tablicy będzie >1 to znaczy że mamy remis
        for j in range(len(labels)):
            if len(bestLabelIndex) == 0:    # jak nie ma żadnego to w ciemno wpakuj jako najwyższy
                bestLabelIndex.append(j)
            elif votes[j] > votes[bestLabelIndex[0]]: # mamy nowy najwyższy
                bestLabelIndex = [j]
            elif votes[j] == votes[bestLabelIndex[0]]: # mamy kolejny taki sam jak najwyższy
                bestLabelIndex.append(j)

        winnerLabel = None
        if len(bestLabelIndex) == 1:
            # mamy swój najwyższy
            winnerLabel = labels[bestLabelIndex[0]]
        else:
            # mamy remis
            # przy wagach 0.3 0.3 0.2 0.2 remis może być tylko pomiędzy dwoma labelami
            # sprawdz ich ilości wystąpień
            label0Counter = 0
            label1Counter = 0
            if labels[bestLabelIndex[0]] in assignedLabels:
                label0Counter = assignedLabelsCount[assignedLabels.index(labels[bestLabelIndex[0]])]
            if labels[bestLabelIndex[1]] in assignedLabels:
                label1Counter = assignedLabelsCount[assignedLabels.index(labels[bestLabelIndex[1]])]
            if(label1Counter > label0Counter):  # jeśli liczniki wystąpień też są takie same to bierz pierwszy z brzegu
                winnerLabel = labels[bestLabelIndex[1]]
            else:
                winnerLabel = labels[bestLabelIndex[0]]

        if winnerLabel in assignedLabels:
            index = assignedLabels.index(winnerLabel)
            assignedLabelsCount[index] = assignedLabelsCount[index] + 1
        else:
            assignedLabels.append(winnerLabel)
            assignedLabelsCount.append(1)
        finalLabels.append(winnerLabel)

    print("Final labels:")
    print(finalLabels)
    acc = calculateACC(finalLabels, y_test)
    rec = calculateRecall(finalLabels, y_test)
    f1 = calculateF1(finalLabels, y_test)
    auc = 0 #calculateAUC(finalLabels, y_test)
    print("ACC: "+str(acc)+" | REC: "+str(rec)+" | F1: "+str(f1)+" | AUC: "+str(auc))

    writer = csv.writer(open("./ensambled_learning.csv", "w"))
    writer.writerow(['ACC','REC','F1','AUC'])
    writer.writerow([acc, rec, f1, auc])

    return finalLabels

def zad4(finalLabels):
    global X_train, X_test, y_train, y_test, reduced_X_train, reduced_X_test

    # rysowanie wykresów
    datasetTrain = X_train
    datasetTest = X_test
    datasetTrain['cluster'] = y_train
    datasetTest['cluster'] = finalLabels

    fig, axs = plt.subplots(2)
    axs[0].set_title("Zbiór treningowy")
    axs[1].set_title("Zbiór testowy")

    sns.scatterplot(x=datasetTrain[0], y=datasetTrain[1], hue=datasetTrain['cluster'], style=datasetTrain['cluster'],
                    data=datasetTrain, ax=axs[0])
    sns.scatterplot(x=datasetTest[0], y=datasetTest[1], hue=datasetTest['cluster'], style=datasetTest['cluster'],
                    data=datasetTest, ax=axs[1])

    plt.show()

zad1()
zad2()
finalLabels = zad3()
zad4(finalLabels)
