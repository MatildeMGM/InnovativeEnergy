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

# Liste over nøgleord, der indikerer restriktioner
nokleord = ["restriktion", "begrænsning", "bebyggelse", "bevaringsværdig", "højde", "afstand", "byggeri"]

def tjek_restriktioner(pdf_url):
    try:
        response = requests.get(pdf_url)
        pdf_file = io.BytesIO(response.content)
        tekst = ""
        with pdfplumber.open(pdf_file) as pdf:
            for side in pdf.pages:
                tekst += side.extract_text() + " "
        tekst_lower = tekst.lower()
        # Tjek om nogle nøgleord er til stede
        fundet = [ord for ord in nokleord if ord in tekst_lower]
        if fundet:
            # Her kan man fx returnere første 200 tegn omkring ordet
            return "Ja", tekst[:500]  # eksempel: de første 500 tegn
        else:
            return "Ingen", ""
    except:
        return "Fejl", ""

# Loop igennem alle rækker (kan tage tid)
for i, row in df.iterrows():
    status, txt = tjek_restriktioner(row["doklink"])
    df.at[i, "Restriktioner"] = status
    df.at[i, "RestriktionerTekst"] = txt

# Gem ny CSV
df.to_csv("lokalplaner_med_restriktioner.csv", index=False)