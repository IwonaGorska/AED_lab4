import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

X_train = None
X_test = None
y_train = None
y_test = None


def zad1():
    global X_train, X_test, y_train, y_test
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


zad1()
zad2()
