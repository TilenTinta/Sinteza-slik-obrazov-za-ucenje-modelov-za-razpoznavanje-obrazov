import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Funkcija za orthogonal regularization
def orthogonal_regularization(directions):
    """
    Izračunaj orthogonal regularization za več smeri.
    directions: Seznam smeri v latentnem prostoru.
    """
    loss = 0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            loss += (np.dot(directions[i], directions[j])) ** 2
    return loss

# Funkcija za treniranje SVM in shranjevanje smeri
def train_svm_and_save_direction(labels_csv, output_direction_path, space="W+", pca_components=None, c_value=1.0):
    # Preberi CSV datoteko
    labels = pd.read_csv(labels_csv)

    # Inicializiraj podatke
    X = []
    y = []
    for _, row in labels.iterrows():
        latent_vector = np.loadtxt(row["latent_vector_path"], delimiter=",")
        if space == "W+":
            if latent_vector.ndim == 2:
                X.append(latent_vector)  # Oblika (18, 512) za W+ prostor
            else:
                raise ValueError("Latentni vektorji za W+ morajo imeti dimenzije (18, 512).")
        elif space == "W":
            if latent_vector.ndim == 1:
                X.append(latent_vector)  # Oblika (1, 512) za W prostor
            else:
                raise ValueError("Latentni vektorji za W morajo imeti dimenzije (1, 512).")
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    # Združite dimenzije za W+ prostor
    if space == "W+":
        X = X.reshape(X.shape[0], -1)  # Preoblikovanje v 2D (npr. (18*512))

    # Standardizacija
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA za zmanjšanje dimenzionalnosti
    if pca_components:
        print(f"Zmanjšujem dimenzionalnost na {pca_components} s PCA...")
        pca = PCA(n_components=pca_components)
        X_scaled = pca.fit_transform(X_scaled)

    # Optimizacija parametra C
    best_c = c_value
    best_accuracy = 0
    for c in [0.01, 0.1, 1.0, 10.0]:
        svm = SVC(kernel="linear", C=c, class_weight="balanced")
        svm.fit(X_scaled, y)
        accuracy = svm.score(X_scaled, y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_c = c
    print(f"Najboljša vrednost C: {best_c}, natančnost: {best_accuracy}")

    # Treniranje SVM z najboljšo vrednostjo C
    svm = SVC(kernel="linear", C=best_c)
    svm.fit(X_scaled, y)
    direction = svm.coef_[0]

    # Če je bila uporabljena PCA, pretvori smer nazaj
    if pca_components:
        direction = pca.inverse_transform(direction)

    # Oblika smeri za W+ prostor
    if space == "W+":
        direction = direction.reshape(18, 512)

    # Shrani smer
    np.save(output_direction_path, direction)
    print(f"Smer za lastnost shranjena v: {output_direction_path}")


if __name__ == "__main__":
    # Pot do vhodne CSV datoteke z oznakami in izhodne datoteke za smer
    labels_csv = "labels.csv"
    output_direction_path = "direction.npy"

    # Parametri za izboljšanje SVM
    space = "W+"  # Možnost izbire prostora: "W" ali "W+"
    pca_components = 50 # Število PCA komponent (None za brez PCA), default: 50
    c_value = 0.1       # Regularizacijski parameter SVM, default: 0.1

    # Treniraj SVM in shrani smer
    train_svm_and_save_direction(labels_csv, output_direction_path, space=space, pca_components=pca_components, c_value=c_value)

    # Primer uporabe orthogonal regularization
    # Za demonstracijo uporabimo dve generirani smeri
    direction_1 = np.random.randn(512)  # Naključna smer 1
    direction_2 = np.random.randn(512)  # Naključna smer 2
    loss = orthogonal_regularization([direction_1, direction_2])
    print(f"Orthogonal regularization loss: {loss}")
