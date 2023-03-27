import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
        set_prueba = self.set_prueba
        return set_entrenamiento, set_prueba
    
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
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
        return valor
    
    def entrenamiento(self, dataset , epochs=5000, imprimir_error_cada=1000, learning_rate=0.001):
        x = dataset.iloc[:,0].to_numpy()
        y = dataset.iloc[:,1].to_numpy()
        v_unos =np.ones_like(x).reshape(-1,1)
        x = np.reshape(x, (-1,1))
        matriz = np.hstack([x, v_unos])
        b1 = 0.2
        b0 = 0.1
        betas = np.array([b1,b0])
        errores = []
        estructura = {}
        parametros = []
        error_df = pd.DataFrame(columns=["Epoch", "Error", "B0", "B1"])
        for i in range(epochs+1):
            y_estimada = np.matmul(matriz, betas)
            error = np.mean(np.power(y-y_estimada,2))
            #delta_b1 = np.mean(np.dot((y_estimada-y), x))
            #delta_b0 = np.mean(y_estimada-y)
            errores.append(error)
            b1 = b1 - learning_rate*b1
            b0 = b0 - learning_rate*b0
            betas = np.array([b1,b0])
            parametros.append([y_estimada, error, b1, b0])
            estructura[i] = parametros
            error_df = pd.concat([error_df, pd.DataFrame({"Epoch": [i], "Error":error, "B0":b0,"B1":b1})])
            if i % imprimir_error_cada == 0:
                print(f" Epoca: {i}, Error: {error}, b0: {b0}, b1: {b1} ")
        return error_df, estructura
    
    def grafica_errores(self, errores):
        errores.plot(x="Epoch", y="Error")
        plt.ylabel("Error")
        plt.title("Gráfica de errores - Función de entrenamiento")
    
    def prediccion(self, x1, x2x, x2y, b1, b0):
        y1_estimada = np.dot(x1, b1) + b0
        regresion = LinearRegression().fit(x2x, x2y)
        y2_estimada = regresion.predict(x2x)
        return y1_estimada, y2_estimada

    def graficar_modelos(self, estructura, datos_entrenamiento, n):
        plt.figure(figsize=(10,8))
        plt.scatter(datos_entrenamiento[:,0], datos_entrenamiento[:,1], s=10)
        for iteracion, parametros in estructura.items():
            try:
                if int(iteracion) % n == 0:
                    x = np.linspace(datos_entrenamiento[:,0].min(), datos_entrenamiento[:,0].max(), 100)
                    y = parametros[0]*x**3 + parametros[1]*x**2 + parametros[2]*x + parametros[3]
                    plt.plot(x, y, '--', label=f'Iteración {iteracion}')
            except ValueError:
                pass
        plt.legend(loc='best')
        plt.show()
    
    def visualizacionComparacion(self, modelo_manual, modelo_scikit, x):
        prediccion_modelo_manual = modelo_manual[0]*x + modelo_manual[1]
        #prediccion_modelo_scikit = modelo_scikit.predict(x.values)
        prediccion_modelo_scikit = modelo_scikit.predict(x.values.reshape(-1,1))
        promedio = (prediccion_modelo_manual+prediccion_modelo_scikit)/2
        return prediccion_modelo_manual, prediccion_modelo_scikit, promedio
