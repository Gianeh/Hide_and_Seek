import os
import matplotlib.pyplot as plt
import numpy as np



agents = ["alpha_0", "alpha_1", "hivemind_0", "alpha_2"]
for a in range(len(agents)):

    reward_path = "./"+agents[a]+"/reward"
    plot_path = "./"+agents[a]+"/reward_plots"

    # Ottieni una lista di tutti i file nella cartella corrente

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    files = os.listdir(reward_path)

    # Filtra solo i file con estensione .txt
    txt_files = [file for file in files if file.endswith('.txt')]

    # Itera sui file .txt nella cartella
    for txt_file in txt_files:
        file_path = os.path.join(reward_path, txt_file)

        # Leggi i dati dal file .txt
        with open(file_path, 'r') as file:

            line = file.readline().strip()

            if line:

                values = line.split(';')
                # Crea un array numpy per i dati
                data = np.array(values)

                # Crea un nuovo grafico
                plt.figure()

                # Plotta i dati
                plt.plot(data)

                # Aggiungi titolo e etichette degli assi se necessario
                plt.title(f'{txt_file}')
                plt.xlabel('Generation')
                plt.ylabel('Reward')


                """new_ticks = [val for val in range(-100, 100 , 2)]
                plt.yticks(new_ticks)"""

                """
                 y_ticks = np.arange(min(data), max(data)+1, step=2)  
                plt.yticks(y_ticks)
                """

                # Salva il grafico come immagine
                image_name = os.path.join(plot_path, f'grafico_{os.path.splitext(txt_file)[0]}.png')
                plt.savefig(image_name)

                # Chiudi il grafico corrente per passare al successivo
                plt.close()
            else:
                print(f"Il file {txt_file} contiene una riga vuota.")



