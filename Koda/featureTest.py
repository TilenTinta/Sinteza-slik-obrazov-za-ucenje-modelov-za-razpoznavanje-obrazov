######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program: featureTest.py - preveri sliko / latentni vektor in glede na izbrano smer pogleda ali ima slika to lastnost
# Autor: Tilen Tinta
# Datum: januar 2025
#######################################################################################################


import os
import numpy as np
import pickle
import dnnlib.tflib as tflib

# Inicializacija omrežja
def initialize_network(model_path):
    print("Inicializacija StyleGAN2...")
    tflib.init_tf()
    with open(model_path, "rb") as f:
        _, _, Gs = pickle.load(f, encoding="latin1")
    print("StyleGAN2 inicializiran.")
    return Gs

# Napovedovanje lastnosti z uporabo latentnega vektorja in smeri
def predict_property(latent_vector, direction, threshold=0):
    """
    Napove, ali ima oseba določeno lastnost glede na smer v latentnem prostoru.

    Parameters:
        latent_vector (numpy.ndarray): Latentni vektor osebe.
        direction (numpy.ndarray): Smer za določeno lastnost.
        threshold (float): Prag za odločitev (privzeto 0).

    Returns:
        bool: True, če ima oseba lastnost; False, sicer.
    """
    # Normalizacija smeri (za stabilnost)
    direction = direction / np.linalg.norm(direction)

    # Izračun projekcije
    projection = np.dot(latent_vector.flatten(), direction.flatten())

    # Napoved lastnosti
    return projection > threshold

# Glavna koda
if __name__ == "__main__":
    # Nastavitve
    lastnost = "Ocala"  # Lastnost, ki jo pregledujemo
    model_path = "stylegan2-ffhq-config-f.pkl"  # Pot do modela
    direction_path = f"results/Sortiranje/{lastnost}/direction.npy"  # Pot do smeri za lastnost
    base_input_dir = f"results/Sortiranje/{lastnost}/Navadna"  # Mapa z latentnimi vektorji

    # Inicializiraj omrežje
    #Gs = initialize_network(model_path)

    # Naloži smer za lastnost
    direction = np.load(direction_path)
    if direction.ndim == 1:
        direction = direction[np.newaxis, :]  # Prilagoditev dimenzij
    print("Smer za lastnost naložena. Oblika:", direction.shape)

    # Poišči prvo mapo v mapi "Navadna"
    person_dirs = sorted([os.path.join(base_input_dir, d) for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))])
    if not person_dirs:
        raise FileNotFoundError(f"Mapa 'Navadna' v {base_input_dir} je prazna ali ne obstaja.")
    first_person_dir = person_dirs[0]  # Prva oseba
    latent_vector_path = os.path.join(first_person_dir, "latent_vector.txt")

    # Preveri, če datoteka latentnega vektorja obstaja
    if not os.path.exists(latent_vector_path):
        raise FileNotFoundError(f"Latentni vektor ni najden na: {latent_vector_path}")

    # Naloži latentni vektor
    latent_vector = np.loadtxt(latent_vector_path, delimiter=",")
    if latent_vector.ndim == 1:
        latent_vector = latent_vector[np.newaxis, :]  # Dodajte dimenzijo za združljivost
    print("Latentni vektor naložen. Oblika:", latent_vector.shape)

    # Napoved lastnosti
    has_property = predict_property(latent_vector, direction)
    print("Oseba ima očala:" if has_property else "Oseba nima očal.")
