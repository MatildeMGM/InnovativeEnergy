import pandas as pd
from pathlib import Path

# Indlæs CSV-filen
BASE_DIR = Path(__file__).resolve().parent  # mappen hvor scriptet ligger
filnavn = BASE_DIR / "planer.csv"           # stien til inputfilen

df = pd.read_csv(filnavn, sep=",", header=0)  # header=0 bruger første linje som kolonnenavne
df = df.fillna("NA")

# Definer prioriterede distrikter
prioritet = ["Rønne", "Nexø"]

# Lav hjælpekolonne til sortering så Rønne og Nexø kommer øverst
df["distrikt_sort"] = df["distrikt"].apply(
    lambda x: (0, x) if x in prioritet else (1, x)
)

# Sorter først efter distrikt-prioritet og derefter efter dato
df_sorted = df.sort_values(by=["distrikt_sort", "datovedt"], ascending=[True, True])

print(df_sorted.columns.tolist()) #se kolonnenavne

# Ny csv fil med udvalgte kolonner
kolonner = ["id","datovedt", "distrikt", "doklink"]  #udvalgte kolonner
df_udvalgte = df_sorted[kolonner] #ny dataframe

# Tilføj nye kolonner
df_udvalgte["Restriktioner"] = "Ingen"
df_udvalgte["RestriktionerTekst"] = ""

# Gem output i samme mappe som scriptet
output_fil = BASE_DIR / "lokalplaner_udvalgte.csv"
df_udvalgte.to_csv(output_fil, index=False)  #lav til ny csv, index=False for at undgå ekstra index-kolonne

print("CSV-fil opdateret:", output_fil)
