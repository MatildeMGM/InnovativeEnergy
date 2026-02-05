import pandas as pd
import os

# Indlæs CSV-filen på en robust måde
filnavn = "planer.csv" #stinavn 

# Sikrer at filen læses fra samme mappe som scriptet kører fra
sti = os.path.join(os.path.dirname(__file__), filnavn)

df = pd.read_csv(sti, sep=",", header=0, encoding="utf-8")  # robust læsning med encoding
df = df.fillna("NA")

# Sorter rækkerne så Rønne og Nexø kommer øverst, derefter resten alfabetisk og efter dato
prioritet = ["Rønne", "Nexø"]

df["distrikt_sort"] = df["distrikt"].apply(
    lambda x: (0, prioritet.index(x)) if x in prioritet else (1, x)
)

df_sorted = df.sort_values(by=["distrikt_sort", "datovedt"], ascending=[True, True])

print(df_sorted.columns.tolist()) #se kolonnenavne

# Ny csv fil med udvalgte kolonner
kolonner = ["id","datovedt", "distrikt", "doklink"]  #udvalgte kolonner
df_udvalgte = df_sorted[kolonner] #ny dataframe

# Gem som CSV (præcis som før – ingen Excel-ændringer)
df_udvalgte.to_csv("lokalplaner_udvalgte.csv", index=False)  #lav til ny csv, index=False for at undgå ekstra index-kolonne

# Tilføj nye kolonner
df_udvalgte["Restriktioner"] = "Ingen"
df_udvalgte["RestriktionerTekst"] = ""
