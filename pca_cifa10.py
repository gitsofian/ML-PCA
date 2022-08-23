
from cifar_loader import CIFAR10
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import time


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

# start timer
start = time.time()

# Erstelle ein PCA-Objekt
pca = PCA()

# Führe PCA durch
X = images_train
pca.fit(X)

# nehmen wir nur ein bestimmtes Prozent von pca.explained_variance_ratio
n_features = 0
sum_of_variance = 0

for v in pca.explained_variance_ratio_:
    sum_of_variance += v
    n_features += 1
    if sum_of_variance >= 1:
        break
# printing
print(f'sum of variance {sum_of_variance}, nb features: {n_features}')

# Erstelle ein PCA-Objekt
pca = PCA(n_components=n_features)
pca.fit(X)

transformed_data = pca.transform(X)
transformed_test = pca.transform(images_test)


def do_knn():

    # K-Neigh-board Klasifisierung Methode

    # X = images_train
    # y = labels_train

    # Aufgabe a) --> TRAINING
    neighbors = 1*5
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(transformed_data, labels_train)

    # Aufgabe b) --> PREDICTION
    pred_test = clf.predict(transformed_test)

    # Aufgabe c) -->
    conf = confusion_matrix(labels_test, pred_test)
    score = clf.score(transformed_test, labels_test)
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

    # stop timer
    end = time.time()
    print(f"Zeit verbraucht : {end-start}\n")

    plt.show()


do_knn()
