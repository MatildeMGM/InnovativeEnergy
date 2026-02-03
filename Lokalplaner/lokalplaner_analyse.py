import pandas as pd

# Indlæs CSV-filen
filnavn = "planer.csv" #stinavn 
df = pd.read_csv(filnavn, sep=",", header=0)  # header=0 bruger første linje som kolonnenavne
df = df.fillna("NA")
df_sorted = df.sort_values(by="datovedt", ascending=True) # Sorter efter en kolonne, fx "PlanDato" stigende
print(df_sorted.columns.tolist()) #se kolonnenavne

# Ny csv fil med udvalgte kolonner
kolonner = ["id","datovedt", "distrikt", "doklink"]  #udvalgte kolonner
df_udvalgte = df_sorted[kolonner] #ny dataframe
df_udvalgte.to_csv("lokalplaner_udvalgte.csv", index=False)  #lav til ny csv, index=False for at undgå ekstra index-kolonne

# Tilføj nye kolonner
df_udvalgte["Restriktioner"] = "Ingen"
df_udvalgte["RestriktionerTekst"] = ""