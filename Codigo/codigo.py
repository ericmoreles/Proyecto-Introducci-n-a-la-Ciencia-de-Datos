import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

#Cargamos los datos

dfEdu = pd.read_excel("datosIndMargEstat.xlsx")
dfIng = pd.read_excel("datosIngCorrAño.xlsx")
dfProm = pd.read_excel("datosAñosPromEsc.xlsx")

dfEdu = dfEdu[["NOM_ENT", "ANALF", "SBASC", "OVSEE", "OVSAE", "IM_2020"]]

dfProm = dfProm[["Área geográfica", 2020]].iloc[1:].reset_index(drop=True)

dfIng = dfIng[["entidad_federativa", "ingreso_corriente"]].iloc[64:96].reset_index(drop=True)

dfEdu[["ANALF", "SBASC", "OVSEE", "OVSAE", "IM_2020"]] = (dfEdu[["ANALF", "SBASC", "OVSEE", "OVSAE", "IM_2020"]] - dfEdu[["ANALF", "SBASC", "OVSEE", "OVSAE", "IM_2020"]].mean())/dfEdu[[ "ANALF", "SBASC", "OVSEE", "OVSAE", "IM_2020"]].std()
dfProm[2020] = (dfProm[2020] - dfProm[2020].mean())/dfProm[2020].std()
dfIng["ingreso_corriente"] = (dfIng["ingreso_corriente"] - dfIng["ingreso_corriente"].mean())/dfIng["ingreso_corriente"].std()

#Ponemos todo en una sola dataframe para tener todo en el mismo lugar

#Diccionario:
#PAE: Promedio de Años de Escolaridad
#PPA: Porcentaje de la poblacion analfabeta de 15 años y mas
#PPEB: Porcentaje de la poblacion sin educacion basica de 15 años o mas
#PPEE: Porcentaje de la poblacion sin Energía elecétrica
#PPAE: Porcenta de la poblacion sin agua entubada
#IM: Indice de Marginación
#ICpC: Ingreso Corriente per capita

#Se probaron varias variables, al final se decide usar solo las que no estan como comentarios, pero de querer se puede probar con las otras variables

dfFinal = pd.DataFrame({
    "Entidad Federativa": dfEdu["NOM_ENT"],
    "PAE": dfProm[2020],
    #"PPA": dfEdu["ANALF"],
    "IM": dfEdu["IM_2020"],
    #"PPEB": dfEdu["SBASC"],
    #"PPEE": dfEdu["OVSEE"],
    #"PPAE": dfEdu["OVSAE"],
    "ICpC": dfIng["ingreso_corriente"],
})

#Pasamos todo a numerico para tener mejor comportamiento
numeric_df = dfFinal.select_dtypes(include=[np.number])

#Creamos y graficamos la matriz de correlacion para ver que tan relacionadas estan las variables
corr = dfFinal.corr(numeric_only=True)

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")

plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

#Hacemos regresion lineal mediante OLS, con una variable dependiente y todas las demas variables como respuesta

#Esta es la variable respuesta, cambiar esta para ver diferentes resultados
var_y = "ICpC"

y = dfFinal[var_y]
X = sm.add_constant(numeric_df.drop(columns=[var_y]))

model = sm.OLS(y, X).fit()
print(model.summary())

print("\n")

X_no_const = numeric_df.drop(columns=[var_y])
X_const = sm.add_constant(X_no_const)

vif = pd.DataFrame()
vif["Variable"] = X_no_const.columns
vif["VIF"] = [variance_inflation_factor(X_const.values, i+1) for i in range(X_no_const.shape[1])]

print(vif)


print("\n")

#Regresion lineal para solo dos variables por visualizacion

#Variable predictiva para regresion lineal simple, cuidado de elegir una diferente de y
var_x = "PAE"

x = dfFinal[var_x]
labels = dfFinal["Entidad Federativa"]

plt.figure(figsize=(11,7))
plt.scatter(x, y, label="Datos reales")

for xi, yi, label in zip(x, y, labels):
    plt.text(xi, yi, label, fontsize=8, alpha=0.8)

X_single = sm.add_constant(x)
model_single = sm.OLS(y, X_single).fit()
print(model_single.summary())


print("\n")

x_sorted = np.linspace(x.min(), x.max(), 100)
X_line = sm.add_constant(x_sorted)
predictions = model_single.get_prediction(X_line)
pred_summary = predictions.summary_frame()

plt.plot(x_sorted, pred_summary["mean"], linewidth=2, label="Línea de regresión")
plt.fill_between(x_sorted, pred_summary["mean_ci_lower"], pred_summary["mean_ci_upper"], alpha=0.3, label="Intervalo de confianza (95%)")
plt.xlabel(f"{var_x}")
plt.ylabel(f"{var_y}")
plt.title(f"{var_y}\n vs {var_x}")
plt.legend()
plt.tight_layout()
plt.show()