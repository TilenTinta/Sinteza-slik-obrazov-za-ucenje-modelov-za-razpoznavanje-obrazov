import os
import shutil
import cv2

def move_or_delete_images(source_dir, target_dir_lastnost, target_dir_navadna):
    """
    Program za razvrščanje slik iz map glede na uporabnikovo izbiro.

    source_dir: Pot do mape, ki vsebuje podmape z datotekami.
    target_dir_lastnost: Ciljna mapa za slike z lastnostjo.
    target_dir_navadna: Ciljna mapa za slike brez lastnosti.
    """
    # Ustvari ciljna direktorija, če še ne obstajata
    os.makedirs(target_dir_lastnost, exist_ok=True)
    os.makedirs(target_dir_navadna, exist_ok=True)

    # Preglej vse podmape v izvorni mapi
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, "generated_image.png")

            if not os.path.exists(image_path):
                print(f"Slika {image_path} ne obstaja, preskakujem...")
                continue

            # Prikaz slike
            img = cv2.imread(image_path)
            cv2.imshow("Image", img)
            print("Pritisnite 'l' za lastnost, 'n' za navadno, 'd' za brisanje.")
            key = cv2.waitKey(0)

            # Razvrsti sliko glede na uporabnikov vnos
            if key == ord('l'):
                target_path = os.path.join(target_dir_lastnost, folder)
                shutil.move(folder_path, target_path)
                print(f"Mapa {folder} premaknjena v {target_dir_lastnost}.")

            elif key == ord('n'):
                target_path = os.path.join(target_dir_navadna, folder)
                shutil.move(folder_path, target_path)
                print(f"Mapa {folder} premaknjena v {target_dir_navadna}.")

            elif key == ord('d'):
                shutil.rmtree(folder_path)
                print(f"Mapa {folder} izbrisana.")

            else:
                print("Napačen vnos. Preskakujem...")

            # Zapri prikazano sliko
            cv2.destroyAllWindows()

if __name__ == "__main__":
    source_dir = "vsi"  # Izvorna mapa z mapami, ki vsebujejo slike
    target_dir_lastnost = "Lastnost"  # Ciljna mapa za slike z lastnostjo
    target_dir_navadna = "Navadna"  # Ciljna mapa za slike brez lastnosti

    move_or_delete_images(source_dir, target_dir_lastnost, target_dir_navadna)
