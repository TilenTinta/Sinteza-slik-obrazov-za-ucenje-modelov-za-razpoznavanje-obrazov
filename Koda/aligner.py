######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program: aligner.py - poravnava obraze na slikah. počasen proces zato se izvede samostojno
# Autor: Tilen Tinta
# Datum: januar 2025
#######################################################################################################

import os
from PIL import Image
from tqdm import tqdm
import dlib
import cv2
import numpy as np

# Funkcija za poravnavo obrazov
def align_face(image, predictor, detector):
    # Preveri, če je slika v PIL formatu, in jo pretvori v numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Preveri obliko slike
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a color image with 3 channels")

    # Pretvori sliko v sivinsko
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detekcija obrazov
    faces = detector(gray)
    if len(faces) > 0:
        # Pridobi prvi zaznani obraz
        face = faces[0]
        
        # Koordinate pravokotnika zaznanega obraza
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Obrezovanje slike (dodajte obrobo, če želite več prostora okoli obraza)
        margin = int(0.2 * w)  # Dodajte 20% prostora okoli obraza
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        cropped_face = image[y1:y2, x1:x2]

        # Poravnava na podlagi oči
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(68):  # 68 ključnih točk
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        left_eye_center = np.mean(landmarks_points[36:42], axis=0)
        right_eye_center = np.mean(landmarks_points[42:48], axis=0)

        # Izračun kota in transformacije
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))  # Kot med očmi

        center = tuple(np.array(cropped_face.shape[1::-1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(cropped_face, M, (cropped_face.shape[1], cropped_face.shape[0]))

        # Pretvori nazaj v PIL
        aligned_face = Image.fromarray(aligned_face.astype(np.uint8))
        return aligned_face

    # Če obrazov ni, vrni originalno sliko kot PIL
    return Image.fromarray(image)


# Glavna funkcija za predobdelavo slik
def process_images(input_dir, output_dir, predictor_path):
    # Preveri, če so poti pravilne
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Naloži dlib modele
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Sprehod po vseh slikah v mapi
    for subdir, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            input_file = os.path.join(subdir, file)
            output_file = os.path.join(output_dir, os.path.relpath(input_file, input_dir))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            try:
                image = Image.open(input_file)
                aligned_image = align_face(image, predictor, detector)
                aligned_image.save(output_file)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    # Pot do vhodne mape z originalnimi slikami
    input_directory = "./Dataset"

    # Pot do izhodne mape za poravnane slike
    #output_directory = "./LFW/aligned_img"
    output_directory = "./Dataset"

    # Pot do datoteke za ključne točke
    predictor_file = "./shape_predictor_68_face_landmarks.dat"

    # Izvedi obdelavo slik
    process_images(input_directory, output_directory, predictor_file)
