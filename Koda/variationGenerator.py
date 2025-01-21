######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program: datasetGenerator.py - generira 1000 identitet in za vsako 50 variant s pomočjo najdenih smeri
# Autor: Tilen Tinta
# Datum: januar 2025
#######################################################################################################


import os
import numpy as np
import pickle
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from urllib.request import urlretrieve

# Prenos modela
def download_model(url, model_path):
    if not os.path.exists(model_path):
        print("Pridobivam model s spleta...")
        urlretrieve(url, model_path)
        print("Model prenesen in shranjen na:", model_path)
    else:
        print("Model že obstaja na:", model_path)


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
    #print("Generiram sliko...")
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    if space == "W+":
        images = Gs.components.synthesis.run(latent_vector, randomize_noise=True, truncation_psi=0.7, output_transform=fmt)
    elif space == "W":
        images = Gs.run(latent_vector, None, truncation_psi=0.5, randomize_noise=True, output_transform=fmt)
    else:
        raise ValueError("Neveljaven prostor. Uporabite 'W' ali 'W+'.")

    image = Image.fromarray(images[0], "RGB")
    image.save(output_path)
    #print("Slika shranjena na:", output_path)


# Shranjevanje latentnega vektorja v datoteko
def save_latent_vector(latent_vector, output_path):
    np.savetxt(output_path, latent_vector, delimiter=",")
    print("Latentni vektor shranjen na:", output_path)


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
    direction = direction / np.linalg.norm(direction)
    projection = np.dot(latent_vector.flatten(), direction.flatten())
    return projection > threshold

##### Main #####
if __name__ == "__main__":
    # Pot do modela in mapa za rezultate
    model_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
    model_path = "stylegan2-ffhq-config-f.pkl"
    base_output_dir = "results/Variacije"

    # Prenesi model, če še ni shranjen lokalno
    download_model(model_url, model_path)

    # Inicializiraj StyleGAN2
    Gs = initialize_network(model_path)

    # Nastavi število oseb za generiranje
    num_people = 1
    num_variations = 20
    space = "W+"

    # Definiraj smeri in njihove omejitve (z vrednosti se je potrebno igrati in gledati generirane slike)
    directions = {
        "starost": (base_output_dir + "/Smeri/starost.npy", (-20, 20)),
        "nasmeh": (base_output_dir + "/Smeri/nasmeh.npy", (-10, 10)),
        "očala": (base_output_dir + "/Smeri/ocala.npy", (-10, 10)),
        "blond": (base_output_dir + "/Smeri/blond.npy", (-8, 8)),
        "rotacija": (base_output_dir + "/Smeri/rotacija.npy", (80, 200)), # 10, 150
        "rasa črna": (base_output_dir + "/Smeri/rasa_crn.npy", (-5, 5)),
        "rasa kitajska": (base_output_dir + "/Smeri/rasa_chn.npy", (0-10, 10)),
        "spol": (base_output_dir + "/Smeri/spol.npy", (-40, 40))
    }

    # Naloži smeri
    loaded_directions = {}
    for name, (path, _) in directions.items():
        direction = np.load(path)
        if direction.ndim == 1:
            direction = direction[np.newaxis, :]
        if direction.shape == (18, 512):
            direction = np.tile(direction, (1, 1, 1))
        loaded_directions[name] = direction

    print("Smeri uspešno naložene.")

    for i in range(1, num_people + 1):
        # Ustvari mapo za osebo
        person_dir = os.path.join(base_output_dir, f"oseba{i}")
        os.makedirs(person_dir, exist_ok=True)

        # Pot za osnovno sliko in latentni vektor
        base_image_path = os.path.join(person_dir, "image1.png")
        base_latent_path = os.path.join(person_dir, "latent_vector1.txt")

        # Ustvari naključni latentni vektor
        latent_vector = np.random.randn(1, Gs.input_shape[1]) # prostor Z (zelo prepleten)
        w_latent = Gs.components.mapping.run(latent_vector, None)  # Pretvorba iz Z v W - (manj prepleten)
        w_latent = w_latent[:, 0, :] # Modifikacija W prostora čene dobimo error (1, 512)
        w_plus = np.tile(w_latent, (1, 18, 1))  # Razširitev v W^+ za 18 slojev - (manj prepleten s plastmi)

        # w_avg = Gs.get_var('dlatent_avg')  # Povprečen W vektor
        # w_plus = w_avg + 0.55 * (w_plus - w_avg)  # Mešanje povprečnega vektorja z W+

        # Shrani osnovno latentno sliko in vektor
        generate_image(Gs, w_plus, base_image_path, space)
        save_latent_vector(w_plus[0, :, :], base_latent_path)

        # Ustvari variacije za vsako osebo
        for j in range(1, num_variations + 1):
            modified_latent = w_plus.copy()

            # Izberi naključno število smeri za to variacijo
            num_directions = np.random.randint(2, len(directions) + 1)  # Izberi vsaj 2 smeri
            selected_directions = np.random.choice(list(directions.keys()), size=num_directions, replace=False)

            # Dodaj intenzivnosti za izbrane smeri
            for name in selected_directions:
                direction, (min_alpha, max_alpha) = directions[name]
                if name in ["očala", "nasmeh", "rotacija", "spol", "rasa črna", "rasa kitajska"]:
                    if predict_property(modified_latent, loaded_directions[name]):
                        #print("Lastnost:", name, " - prisotna")
                        intensity = np.random.uniform(-max_alpha / 2, -min_alpha / 2)  # Odstrani lastnost
                    else:
                        #print("Lastnost:", name, " - NI prisotna")
                        intensity = np.random.uniform(min_alpha / 2, max_alpha / 2)  # Dodaj lastnost
                elif name == "starost":
                    intensity = np.random.uniform(min_alpha, max_alpha) * np.random.choice([-1, 1])  # Spreminjaj starost
                else:
                    intensity = np.random.uniform(min_alpha, max_alpha)

                # Dodaj spremembo k latentnemu vektorju
                modified_latent += intensity * loaded_directions[name]

            # Shranjevanje generirane slike
            variation_image_path = os.path.join(person_dir, f"image_{j+1}.png")
            generate_image(Gs, modified_latent, variation_image_path, space)

            # Shranjevanje modificiranega latentnega vektorja
            variation_latent_path = os.path.join(person_dir, f"latent_vector{j+1}.txt")
            save_latent_vector(modified_latent[0, :, :], variation_latent_path)


    print(f"Generiranih {num_people} oseb s po {num_variations} variacijami.")
