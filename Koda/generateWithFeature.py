######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program generateWithFeatures.py - program generira sliko iz latentnega prostora z dodano smerjo in shrani sliko
# Autor: Tilen Tinta
# Datum: januar 2025
#######################################################################################################


import os
import numpy as np
import pickle
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib

# Inicializacija omrežja
def initialize_network(model_path):
    print("Inicializacija StyleGAN2...")
    tflib.init_tf()
    with open(model_path, "rb") as f:
        _, _, Gs = pickle.load(f, encoding="latin1")
    print("StyleGAN2 inicializiran.")
    return Gs

# Generiranje slike iz latentnega vektorja
def generate_image(Gs, latent_vector, output_path, space):
    print("Generiram sliko...")
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    if space == "W+":
        images = Gs.components.synthesis.run(latent_vector, minibatch_size=2, randomize_noise=True, truncation_psi=0.7, output_transform=fmt)
    elif space == "W":
        images = Gs.run(latent_vector, None, truncation_psi=0.5, randomize_noise=True, output_transform=fmt)
    else:
        raise ValueError("Neveljaven prostor. Uporabite 'W' ali 'W+'.")

    image = Image.fromarray(images[0], "RGB")
    image.save(output_path)
    print("Slika shranjena na:", output_path)


# Glavna koda
if __name__ == "__main__":

    lastnost = "Spol"
    space = "W+"
    #space = "W"

    # Pot do modela in smeri za lastnost
    model_path = "stylegan2-ffhq-config-f.pkl"
    direction_path = "results/Sortiranje/" + lastnost + "/direction.npy"

    # Pot do prvega latentnega vektorja iz mape "Navadna"
    base_input_dir = "results/Sortiranje/" + lastnost + "/Navadna"
    person_dirs = sorted([os.path.join(base_input_dir, d) for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))])
    first_person_dir = person_dirs[0]  # Prva oseba
    latent_vector_path = os.path.join(first_person_dir, "latent_vector.txt")

    # Preveri, če datoteka latentnega vektorja obstaja
    if not os.path.exists(latent_vector_path):
        raise FileNotFoundError(f"Latentni vektor ni najden na: {latent_vector_path}")

    # Naloži model in smer za lastnost
    Gs = initialize_network(model_path)
    direction = np.load(direction_path)
    if direction.ndim == 1:
        direction = direction[np.newaxis, :]  # Dodajte dimenzijo za združljivost

    print("Smer za lastnost naložena. Oblika:", direction.shape)

    # Naloži latentni vektor
    latent_vector = np.loadtxt(latent_vector_path, delimiter=",")
    if latent_vector.ndim == 1:
        latent_vector = latent_vector[np.newaxis, :]  # Dodajte dimenzijo za združljivost

    # Dodaj lastnost (npr. očala)
    intensity = 90  # Intenzivnost spremembe

    """
    Min in max vrednosti da je sprememba nam vidna
    + Očala:
        - min: 5 (steklca)
        - 10 (močna črn okvir)
        - max: 15 (črna večja)
        - 20 (črna večja, kvari lase)
        - 30 (začetek sončnih, kvari lase plus kapa)
    + Blond:
        - min: 3
        - max: 10 (prou blond)
    + Spol:
        - min: 20
        - max: 50 (ženska)
    + Starost:
        - min: 10
        - max: 100
    + Nasmeh:
        - min: 8
        - max: 12
    + Rasa_crn:
        - min: 1
        - max: 7
    + Rasa_chn:
        - min: 5
        - max: 20
    + Rotacija:
        - min: 10
        - max: 150
    """
    if space == "W+":
    # Preverjanje in prilagoditev dimenzij
        if direction.shape == (18, 512):
            direction = np.tile(direction, (latent_vector.shape[0], 1, 1))  # Prilagoditev oblike
        else:
            raise ValueError(f"Dimenzije smeri ({direction.shape}) ne ustrezajo W+ prostoru (18, 512).")
        modified_latent = latent_vector + intensity * direction
    elif space == "W":
        if direction.shape[0] != latent_vector.shape[1]:
            raise ValueError(f"Dimenzije smeri ({direction.shape}) ne ustrezajo W prostoru ({latent_vector.shape[1]}).")
        modified_latent = latent_vector + intensity * direction

    # ### Lastnost 2 ##
    # lastnost = "Ocala"
    # direction_path = "results/Sortiranje/" + lastnost + "/direction.npy"

    # # Naloži model in smer za lastnost
    # Gs = initialize_network(model_path)
    # direction = np.load(direction_path)
    # if direction.ndim == 1:
    #     direction = direction[np.newaxis, :]  # Dodajte dimenzijo za združljivost

    # # Dodaj lastnost (npr. očala)
    # intensity = 4  # Intenzivnost spremembe
    # modified_latent = modified_latent + intensity * direction


    ### ANIMACIJA ###
    # for alpha in np.linspace(-3, 3, 10):  # Od mlajšega do starejšega
    #     z_frame = z_original + alpha * direction_age
    #     generate_image(Gs, z_frame, f"frame_{alpha}.png")

    ### Manipulacija smeri samo na določenih plasteh ###
    # for layer in range(8, 18):  # Spremenite samo sloje 8–18 (globoke plasti)
    #     z_modified[:, layer, :] += 1.0 * direction_age



    # Pot za generirano sliko z dodano lastnostjo
    output_path = os.path.join(first_person_dir, "modified.png")

    # Generiraj in shrani novo sliko
    generate_image(Gs, modified_latent, output_path, space)
    print("Nova slika z dodano lastnostjo ustvarjena.")
