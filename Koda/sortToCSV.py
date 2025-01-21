import os
import csv

def generate_labels_csv(feature_dir, normal_dir, output_csv):
    """
    Ustvari CSV datoteko z oznakami za latentne vektorje.

    feature_dir: Pot do mape z latentnimi vektorji, ki imajo lastnost (označeni z 1).
    normal_dir: Pot do mape z latentnimi vektorji brez lastnosti (označeni z 0).
    output_csv: Pot do izhodne CSV datoteke.
    """
    rows = []

    # Procesiranje map z lastnostjo
    for folder in os.listdir(feature_dir):
        folder_path = os.path.join(feature_dir, folder)
        if os.path.isdir(folder_path):
            latent_vector_path = os.path.join(folder_path, "latent_vector.txt")
            if os.path.exists(latent_vector_path):
                rows.append([latent_vector_path, 1])

    # Procesiranje map brez lastnosti
    for folder in os.listdir(normal_dir):
        folder_path = os.path.join(normal_dir, folder)
        if os.path.isdir(folder_path):
            latent_vector_path = os.path.join(folder_path, "latent_vector.txt")
            if os.path.exists(latent_vector_path):
                rows.append([latent_vector_path, 0])

    # Pisanje v CSV datoteko
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["latent_vector_path", "label"])
        writer.writerows(rows)

    print(f"CSV datoteka ustvarjena na: {output_csv}")

if __name__ == "__main__":
    # Pot do map z latentnimi vektorji
    feature_dir = "Lastnost"
    normal_dir = "Navadna"
    output_csv = "labels.csv"

    # Ustvari CSV datoteko
    generate_labels_csv(feature_dir, normal_dir, output_csv)
