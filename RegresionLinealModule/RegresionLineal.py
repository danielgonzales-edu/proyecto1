import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

class RegresionLinealClass:
    def __init__(self, path, segmentacion=0.8):
        self.path = path
        self.data_set = np.load(self.path, 'r')
        self.segmentacion = segmentacion
        self.df_regresion =  pd.DataFrame(self.data_set, columns = ['PrecioVenta', 'Calificación', 'ÁreaPP', 'Habitaciones', 'AñoConstrucción', 'Frente'])
        self.set_entrenamiento = self.df_regresion.sample(frac=segmentacion)
        self.set_prueba = self.df_regresion.drop(self.set_entrenamiento.index)
        
    def mostrarDatos(self):
        set_entrenamiento = self.set_entrenamiento
        return set_entrenamiento
    
    def graficarVariable(self, variable, x, y, titulo):
        histograma = sb.displot(variable, kde=True)
        histograma.set(xlabel=x, ylabel=y, title=titulo)
        return histograma
    
    def graficarCorrelacion(self, set_entrenamiento, correlacion):
        fig, axs = plt.subplots(len(set_entrenamiento.columns), len(set_entrenamiento.columns), figsize=(30, 30))
        fig.suptitle("Diagramas de dispersión", fontsize=32)
        for x in range(len(set_entrenamiento.columns)):
            for y in range(len(set_entrenamiento.columns)):
                axs[x,y].scatter(set_entrenamiento[set_entrenamiento.columns[x]], set_entrenamiento[correlacion.columns[y]])
                axs[x,y].set_xlabel(set_entrenamiento.columns[y])
                val_corr = str(set_entrenamiento.corr().iloc[x,y])
                titulo = "Val. Correlación " + val_corr
                axs[x,y].set_title(titulo)
                if y == 0:
                    axs[x,y].set_ylabel(set_entrenamiento.columns[x])
        pass

    def graficarPotencialPredictivo(self, set1, set2, x, y):
        valor = plt.scatter(set1, set2)
        plt.xlabel("Precio de Venta")
        plt.ylabel("Calificación")
        plt.show()