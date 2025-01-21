######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program: mereKakovosti.py - računanje mer kakovosti na slikah/vektorjih
# Autor: Tilen Tinta
# Datum: januar 2025
#######################################################################################################

import os
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1. Preberi latentne vektorje iz datotek
def load_latent_vectors(dataset_path):
    latent_vectors = []
    for i in range(1, 1001):  # Številke oseb od 1 do 1000
        file_path = os.path.join(dataset_path, f"oseba{i}", "latent_vector.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                vector = np.array([float(x) for x in f.read().strip().replace(',', ' ').split()])
                latent_vectors.append(vector)
        else:
            print(f"Opozorilo: Datoteka {file_path} ne obstaja.")
    return np.array(latent_vectors)


# 2. Funkcije za izračun mer kakovosti
def calculate_intra_latent_variance(latent_vectors):
    distances = pairwise_distances(latent_vectors)
    variance = np.var(distances)
    return variance

def calculate_average_pairwise_distance(latent_vectors):
    distances = pairwise_distances(latent_vectors)
    mean_distance = np.mean(distances)
    return mean_distance

def perceptual_path_length(latent_vectors, step=1):
    path_length = 0
    for i in range(0, len(latent_vectors) - step, step):
        path_length += np.linalg.norm(latent_vectors[i] - latent_vectors[i + step])
    return path_length / len(latent_vectors)


# 3. Vizualizacija z ravnino
def plot_projection_with_plane(latent_vectors, direction, dims=(0, 1, 2), scale=5):
    if latent_vectors.shape[0] == 0:
        print("Napaka: Ni latentnih vektorjev za vizualizacijo.")
        return

    # Preoblikovanje latentnih vektorjev, če je potrebno
    if latent_vectors.ndim == 3:  # Oblika (n_samples, 18, 512)
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
    if direction.ndim == 2:  # Oblika (18, 512)
        direction = direction.flatten()

    # Povprečni latentni vektor
    avg_latent_vector = np.mean(latent_vectors, axis=0)

    # Projekcija na določene dimenzije
    projected_latents = latent_vectors[:, dims]
    projected_avg = avg_latent_vector[np.array(dims)]
    projected_direction = direction[np.array(dims)]

    # Preverjanje dimenzij za pravokotno smer
    if len(projected_direction) != 3:
        raise ValueError(f"Dimenzije smeri morajo biti 3, ampak so {len(projected_direction)}.")

    # Generiranje dodatnega pravokotnega vektorja za ravnino
    orthogonal_vector = np.cross(projected_direction, [1, 0, 0])
    if np.linalg.norm(orthogonal_vector) == 0:  # Če je vzporeden z [1, 0, 0], vzemi drugo osnovno os
        orthogonal_vector = np.cross(projected_direction, [0, 1, 0])
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    # Mreža za ravnino
    plane_x = np.linspace(-scale, scale, 10)
    plane_y = np.linspace(-scale, scale, 10)
    plane_x, plane_y = np.meshgrid(plane_x, plane_y)
    plane_z = (projected_avg[2] + 
               (plane_x - projected_avg[0]) * orthogonal_vector[0] + 
               (plane_y - projected_avg[1]) * orthogonal_vector[1]) / -orthogonal_vector[2]

    # Prikaz v 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_latents[:, 0], projected_latents[:, 1], projected_latents[:, 2], 
               s=10, alpha=0.4, label="Latentni vektorji")

    # Dodaj ravnino
    ax.plot_surface(plane_x, plane_y, plane_z, color='red', alpha=0.5, label="Ravnina smeri")

    ax.set_xlabel(f"Dimenzija {dims[0]}")
    ax.set_ylabel(f"Dimenzija {dims[1]}")
    ax.set_zlabel(f"Dimenzija {dims[2]}")

    plt.title("Projekcija Latentnega Prostora z Ravnino Smeri")
    plt.legend()
    plt.show()




##### Main #####
if __name__ == "__main__":
    dataset_path = "./Dataset"  # Pot do mape z latentnimi vektorji
    directions_path = "./Smeri"  # Pot do mape z datotekami smeri

    # Naloži latentne vektorje
    latent_vectors = load_latent_vectors(dataset_path)

    if latent_vectors.size == 0:
        print("Ni podatkov za obdelavo.")
    else:
        # Izračun metrik kakovosti
        intra_variance = calculate_intra_latent_variance(latent_vectors)
        avg_pairwise_distance = calculate_average_pairwise_distance(latent_vectors)
        ppl = perceptual_path_length(latent_vectors)

        # Prikaz rezultatov
        print("\nMere kakovosti:")
        print(f"Intra-latent Variance: {intra_variance:.4f}")
        print(f"Average Pairwise Distance: {avg_pairwise_distance:.4f}")
        print(f"Perceptual Path Length: {ppl:.4f}")

        # Izberite eno smer za vizualizacijo (npr. "starost")
        selected_direction_path = os.path.join(directions_path, "rasa_crn.npy")
        if os.path.exists(selected_direction_path):
            selected_direction = np.load(selected_direction_path)
            plot_projection_with_plane(latent_vectors, selected_direction, dims=(0, 1, 2), scale=5)
        else:
            print("Napaka: Izbrana smer ni bila najdena.")
