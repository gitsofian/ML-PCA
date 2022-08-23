
from cifar_loader import CIFAR10
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# Lade CIFAR10 Datensatz
cifar10 = CIFAR10('cifar10')


# Lade Trainings und Testdaten, sowie die Labels der Klassen
images_train, labels_train = cifar10.train_data, cifar10.train_labels
images_test, labels_test = cifar10.test_data, cifar10.test_labels
label_names = cifar10.label_names

# Plotte Beispiele um den Datensatz zu visualisieren
fig, axes = plt.subplots(1, 10, sharey=True)
for i, ax in enumerate(axes):
    subset = np.asarray(labels_train == i).nonzero()[0]
    idx = np.random.choice(subset)
    image = images_train[idx].reshape((3, 32, 32)).transpose((1, 2, 0))
    label = labels_train[idx]

    ax.imshow(image)
    ax.set_title(label_names[i], rotation='vertical')
    ax.axis("off")


# --- Projektarbeit ---


def do_knn():

    # K-Neigh-board Klasifisierung Methode

    # X = images_train
    # y = labels_train

    # Aufgabe a) --> TRAINING
    neighbors = 10*3
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(images_train, labels_train)

    # Aufgabe b) --> PREDICTION
    pred_test = clf.predict(images_test)

    # Aufgabe c) -->
    conf = confusion_matrix(labels_test, pred_test)
    score = clf.score(images_test, labels_test)
    print(f"score : {score:.5f}")

    if conf is not None:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(conf, cmap="Blues")
        # ax.axis("off")
        ax.set_xlabel("prediction")
        ax.set_ylabel("ground truth")

        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # labels = label_names
        # labels[1] = 'Testing'

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        ax.set_xticklabels(label_names, rotation='vertical')
        ax.set_yticklabels(label_names)

        for i in range(10):
            for j in range(10):
                ax.text(j, i, "%d" % conf[i, j], color="red",
                        horizontalalignment='center', verticalalignment='center')

    plt.title(
        f"K-Neigh-board, K-: {neighbors}, score: {score}", loc='right')

    plt.show()


def do_svm():
    # SVM Methode implementieren
    # Erstelle Support Vector Classifier

    kernel = "linear"
    clf = svm.SVC(kernel=kernel, gamma='auto')

    # Training
    clf.fit(images_train, labels_train)

    # Predict the value
    pred_test = clf.predict(images_test)

    # Score
    conf = confusion_matrix(labels_test, pred_test)
    score = clf.score(images_test, labels_test)
    print(f"score : {score:.5f}")

    if conf is not None:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(conf, cmap="Blues")
        # ax.axis("off")
        ax.set_xlabel("prediction")
        ax.set_ylabel("ground truth")

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        ax.set_xticklabels(label_names, rotation='vertical')
        ax.set_yticklabels(label_names)

        for i in range(10):
            for j in range(10):
                ax.text(j, i, "%d" % conf[i, j], color="red",
                        horizontalalignment='center', verticalalignment='center')

    plt.title(
        f"Supprt Vector Machine, Kernel: {kernel}, score: {score}", loc='right')

    plt.show()


def linear():
    # linear Modell implementieren
    pass


wahl = 1


match wahl:
    case 1:
        do_knn()
    case 2:
        do_svm()
    case 3:
        linear()
    case _:
        print(f"kein Auswahl")
