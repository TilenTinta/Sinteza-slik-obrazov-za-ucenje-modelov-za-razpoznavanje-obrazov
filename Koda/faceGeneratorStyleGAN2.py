######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program: faceGeneratorStyleGAN2.py - testno generiranje n obrazov iz StyleGAN2 NN v različnih latentnih prostorih
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
    print("Generiram sliko...")
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    if space == "W+":
        # Uporabite synthesis modul za prostor W^+
        images = Gs.components.synthesis.run(latent_vector, randomize_noise=True, truncation_psi=0.7, output_transform=fmt)
    elif space == "W":
        # Za prostor W uporabite običajen Gs.run()
        images = Gs.run(latent_vector, None, truncation_psi=0.5, randomize_noise=True, output_transform=fmt)
    else:
        raise ValueError("Neveljaven prostor. Uporabite 'W' ali 'W+'.")

    image = Image.fromarray(images[0], "RGB")
    image.save(output_path)
    print("Slika shranjena na:", output_path)


# Shranjevanje latentnega vektorja v datoteko
def save_latent_vector(latent_vector, output_path):
    np.savetxt(output_path, latent_vector, delimiter=",")
    print("Latentni vektor shranjen na:", output_path)

##### Main #####
if __name__ == "__main__":
    # Pot do modela in mapa za rezultate
    model_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
    model_path = "stylegan2-ffhq-config-f.pkl"
    base_output_dir = "results/generirano"

    # Prenesi model, če še ni shranjen lokalno
    download_model(model_url, model_path)

    # Inicializiraj StyleGAN2
    Gs = initialize_network(model_path)

    # Nastavi število oseb za generiranje
    num_people = 600
    space = "W+"
    #space = "W"

    for i in range(1, num_people + 1):
        # Ustvari mapo za osebo
        person_dir = os.path.join(base_output_dir, f"oseba{i}")
        os.makedirs(person_dir, exist_ok=True)

        # Pot za sliko in latentni vektor
        image_path = os.path.join(person_dir, "generated_image.png")
        latent_path = os.path.join(person_dir, "latent_vector.txt")

        # Ustvari naključni latentni vektor
        latent_vector = np.random.randn(1, Gs.input_shape[1]) # prostor Z (zelo prepleten)

        # Ustvarjanje latentnih vektorjev okrog osnove - "podobne slike in dolgočasne slike"
        w_avg = Gs.get_var('dlatent_avg')  # Povprečen W vektor
        latent_vector = w_avg + np.random.randn(1, Gs.input_shape[1]) * 0.1  # Dodajte manjše naključne variacije


        w_latent = Gs.components.mapping.run(latent_vector, None)  # Pretvorba iz Z v W - (manj prepleten)
        w_latent = w_latent[:, 0, :] # Modifikacija W prostora čene dobimo error (1, 512)
        w_plus = np.tile(w_latent, (1, 18, 1))  # Razširitev v W^+ za 18 slojev - (manj prepleten s plastmi)

        # for layer in range(6, 8):  # Prilagodimo le srednje sloje (6-7)
        #     w_plus[:, layer, :] += np.random.randn(1, 512) * 0.1  # Dodamo majhen šum

        w_avg = Gs.get_var('dlatent_avg')  # Povprečen W vektor
        w_plus = w_avg + 0.55 * (w_plus - w_avg)  # Mešanje povprečnega vektorja z W+


        #print(f"Latent vector shape: {w_plus.shape}")

        # Generiraj sliko in shrani latentni vektor
        if space == "W+":
            generate_image(Gs, w_plus, image_path, space)
            save_latent_vector( w_plus[0, :, :], latent_path)
        else:
            generate_image(Gs, w_latent, image_path, space)
            save_latent_vector( w_latent, latent_path)

        #print(f"Latent vector shape: {w_plus[0, :, :].shape}")

    print(f"Generiranih {num_people} oseb.")
