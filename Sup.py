import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from crewai import Crew, Agent, Task, Process
from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
import requests
import matplotlib.pyplot as plt
import pandas as pd

# Impostazioni di base per i grafici di Matplotlib
plt.style.use("ggplot")
plt.rcParams.update({"figure.figsize": (8, 4), "figure.dpi": 100})

openai_api_key = st.secrets["OPENAI_API_KEY"]
serpapi_api_key = st.secrets["SERPAPI_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_api_key

# Inizializzazione del modello LLM (aggiorna modello, temperature, ecc.)
GPT4_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

# ---------------------------------------------------------------------
# FUNZIONI UTILI
# ---------------------------------------------------------------------
def fetch_data(query: str, api_key: str = serpapi_api_key):
    """
    Esempio di funzione per recuperare dati reali dal web usando SerpAPI.
    Ritorna una lista di risultati con 'title', 'link' e 'snippet'.
    """
    if not api_key:
        st.warning("Nessuna API Key per SerpAPI disponibile.")
        return []

    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("organic_results", [])
    else:
        return [{"title": "Nessun dato trovato", "link": "", "snippet": ""}]

# ---------------------------------------------------------------------
# DEFINIZIONE DEGLI AGENTI (ESEMPIO DI SUPPLY CHAIN)
# ---------------------------------------------------------------------
supplier = Agent(
    role="Supplier",
    goal="Fornire materie prime affidabili rispettando costi e tempi di consegna.",
    backstory="""Backstory for Supplier
        Education & Experience:
        - Laurea in Ingegneria Industriale con 10+ anni in approvvigionamento di materie prime.
        - Rete consolidata di fornitori Tier-2.

        Job Description:
        - Gestisce programmi di produzione, costi e lead time.
        - Monitora fluttuazioni di prezzo delle materie prime.

        Guiding Principles:
        - Reliability, Transparency, Adaptability.

        Safety Guidelines:
        - Conformità a standard etici e regolamenti sul lavoro.
    """,
    llm=GPT4_LLM,
)

manufacturer = Agent(
    role="Manufacturer",
    goal="Assemblare i prodotti finiti, ottimizzando capacità e costi.",
    backstory="""Backstory for Manufacturer
        Education & Experience:
        - MBA in Operations Management, 8+ anni di ottimizzazione della linea di produzione.
        
        Job Description:
        - Gestisce capacità produttive, schedule e controlli di qualità.
        - Collabora con fornitori per Just-In-Time delivery.

        Guiding Principles:
        - Efficienza, Qualità, Flessibilità.

        Safety Guidelines:
        - Conformità a norme ISO, OSHA.
    """,
    llm=GPT4_LLM,
)

logistics_provider = Agent(
    role="Logistics Provider",
    goal="Assicurare trasporti e distribuzione efficienti e puntuali.",
    backstory="""Backstory for Logistics Provider
        Education & Experience:
        - Laurea in Logistics Management, esperienza in shipping globale.
        
        Job Description:
        - Pianifica rotte, gestisce magazzini, minimizza ritardi e costi.
        - Coordina con vettori e si occupa di eventuali disagi.

        Guiding Principles:
        - Affidabilità, Economicità, Trasparenza.

        Safety Guidelines:
        - Rispetto delle normative doganali e di sicurezza cargo.
    """,
    llm=GPT4_LLM,
)

retailer = Agent(
    role="Retailer",
    goal="Prevedere la domanda, gestire lo stock e massimizzare le vendite.",
    backstory="""Backstory for Retailer
        Education & Experience:
        - Laurea in Business Administration, specializzazione in Retail & Merchandising.
        
        Job Description:
        - Monitora trend di mercato e comportamenti dei consumatori.
        - Lancia promozioni e definisce prezzi strategici.

        Guiding Principles:
        - Customer Centricity, Data-Driven Forecasting, Agilità.

        Safety Guidelines:
        - Rispetto privacy clienti e norme protezione dati.
    """,
    llm=GPT4_LLM,
)

competitor_analyst = Agent(
    role="Competitor Analyst",
    goal="Analizzare catene di approvvigionamento concorrenti, pricing e strategie.",
    backstory="""Backstory for Competitor Analyst
        Education & Experience:
        - Master in Business Intelligence & Competitive Analysis, esperienza con Fortune 500.
        
        Job Description:
        - SWOT e ricerche su competitor (prezzi, distribuzione, marketing).
        - Consiglia strategie di differenziazione.

        Guiding Principles:
        - Accuratezza Fattuale, Analisi Etica, Allineamento Strategico.

        Safety Guidelines:
        - Rispetta confini legali e fair competition.
    """,
    llm=GPT4_LLM,
)

# ---------------------------------------------------------------------
# TASKS DI ESEMPIO (SUPPLY CHAIN)
# ---------------------------------------------------------------------
supplier_task = Task(
    description="Raccogli dati lato fornitore: disponibilità materie prime, costi, potenziali colli di bottiglia.",
    expected_output="Report su materie prime, capacità produttiva e rischi di procurement.",
    agent=supplier
)

manufacturer_task = Task(
    description="Valuta capacità produttive, costo unitario e pianificazione della produzione.",
    expected_output="Analisi della produzione, costi e suggerimenti di ottimizzazione.",
    agent=manufacturer
)

logistics_task = Task(
    description="Elabora una strategia logistica, incluse rotte, costi e possibili interruzioni.",
    expected_output="Piano logistico con opzioni di trasporto, proiezioni costi e rischi.",
    agent=logistics_provider
)

retailer_task = Task(
    description="Prevedi la domanda a livello retail e gestisci lo stock di magazzino.",
    expected_output="Forecast di domanda, consigli su ordini, strategie di pricing.",
    agent=retailer
)

competitor_task = Task(
    description="Analizza i concorrenti: strategie di supply chain, pricing e posizionamento.",
    expected_output="Analisi competitiva con raccomandazioni di differenziazione.",
    agent=competitor_analyst
)

# ---------------------------------------------------------------------
# CREW: GESTIONE DEI TASK
# ---------------------------------------------------------------------
crew = Crew(
    agents=[supplier, manufacturer, logistics_provider, retailer, competitor_analyst],
    tasks=[supplier_task, manufacturer_task, logistics_task, retailer_task, competitor_task],
    process=Process.sequential  # O "parallel" se vuoi eseguirli contemporaneamente
)

# ---------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------
def main():
    st.title("Simulazione Avanzata di Supply Chain")
    st.markdown("""
    Benvenuto in un'applicazione di **simulazione agent-based** per la tua supply chain.
    Inserisci i parametri richiesti e avvia la simulazione per ottenere
    previsioni, analisi e raccomandazioni personalizzate.
    ---
    """)

    with st.form("simulation_form"):
        st.subheader("Parametri di Ingresso per la Simulazione")
        
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input("Nome del prodotto/scenario *", value="Prodotto Esempio")
            competitor_name = st.text_input("Nome di un concorrente (opzionale)", value="")
            
            # Parametri del fornitore
            supplier_lead_time = st.number_input("Lead time medio del Fornitore (giorni)", min_value=1, max_value=60, value=15)
            supplier_cost_factor = st.slider("Fattore costo fornitore", min_value=0.1, max_value=10.0, value=1.0)
            
        with col2:
            # Parametri di produzione
            manufacturing_capacity = st.number_input("Capacità produttiva (unità al mese)", min_value=100, max_value=100000, value=10000)
            manufacturing_cost_per_unit = st.number_input("Costo di produzione per unità (€)", min_value=1, max_value=2000, value=50)
            
            # Parametri di logistica
            logistics_cost_factor = st.slider("Fattore costo logistico", min_value=0.1, max_value=10.0, value=1.0)
        
        # Parametri Retail
        retail_demand_forecast = st.slider("Domanda prevista (unità al mese)", min_value=100, max_value=200000, value=5000)
        retail_price = st.number_input("Prezzo di vendita (€)", min_value=1, max_value=5000, value=120)
        
        st.form_submit_button("Applica Parametri")

    if st.button("Avvia Simulazione"):
        st.write("**Elaborazione in corso...**")
        
        # --- Se vuoi integrare SerpAPI per product/competitor
        combined_data = ""
        
        if product_name:
            product_data = fetch_data(product_name)
            st.subheader("Dati reali sul prodotto/scenario:")
            for item in product_data:
                st.markdown(f"**{item['title']}**\n{item['snippet']}\n[Approfondisci]({item['link']})")
                combined_data += f"{item['title']}. {item['snippet']}\n"
        
        if competitor_name:
            competitor_data = fetch_data(competitor_name)
            st.subheader(f"Dati reali sul concorrente '{competitor_name}':")
            for item in competitor_data:
                st.markdown(f"**{item['title']}**\n{item['snippet']}\n[Approfondisci]({item['link']})")
                combined_data += f"{item['title']}. {item['snippet']}\n"

        # --- Prompt finale per il LLM (integra i parametri personalizzati)
        full_prompt = f"""
        Stiamo simulando una supply chain per il prodotto '{product_name}' con i seguenti parametri:
        
        - Lead time fornitore: {supplier_lead_time} giorni
        - Fattore costo fornitore: {supplier_cost_factor}
        - Capacità produttiva mensile: {manufacturing_capacity} unità
        - Costo di produzione per unità: {manufacturing_cost_per_unit} €
        - Fattore costo logistico: {logistics_cost_factor}
        - Domanda prevista al dettaglio: {retail_demand_forecast} unità/mese
        - Prezzo di vendita: {retail_price} €
        
        Dati reali (SerpAPI) sul prodotto e/o concorrenti:
        {combined_data}

        Svolgi la seguente analisi:
        1. Supplier: Disponibilità materie prime, costi e rischi di procurement.
        2. Manufacturer: Rischi di capacity overload, costi totali di produzione.
        3. Logistics Provider: Costi di trasporto e possibili colli di bottiglia.
        4. Retailer: Analisi di domanda vs. offerta, possibili stockout o surplus.
        5. Competitor Analyst: Come la concorrenza potrebbe rispondere o quali differenziazioni adottare.

        Fornisci:
        - Un prospetto di costi totali (materie prime + produzione + logistica).
        - Un'analisi dei principali rischi e strategie di mitigazione.
        - Suggerimenti per ottimizzare la supply chain (ridurre costi, lead time, ecc.).
        - Un focus sulla competitività rispetto a '{competitor_name}' (se inserito).
        """

        # Chiediamo al LLM di generare la simulazione/analisi
        simulation_output = GPT4_LLM.predict(full_prompt)
        
        # --- Output del LLM
        st.subheader("Risultati della Simulazione e Analisi")
        st.write(simulation_output)

        # --- Visualizzazione di alcuni grafici di esempio
        st.subheader("Grafici di Sintesi")

        # 1) Curva di domanda vs. produzione
        months = ["M1", "M2", "M3", "M4", "M5"]
        
        # Generiamo dati fittizi giusto per esempio
        production_plan = [manufacturing_capacity * 0.8,  # ipotesi
                           manufacturing_capacity * 0.85,
                           manufacturing_capacity * 0.9,
                           manufacturing_capacity * 0.9,
                           manufacturing_capacity * 0.95]
        demand_curve = [retail_demand_forecast * 1.0, 
                        retail_demand_forecast * 1.05,
                        retail_demand_forecast * 0.95,
                        retail_demand_forecast * 1.1,
                        retail_demand_forecast * 1.0]

        fig, ax = plt.subplots()
        ax.plot(months, production_plan, label="Produzione Pianificata", marker="o")
        ax.plot(months, demand_curve, label="Domanda Stimata", marker="s")
        ax.set_title("Confronto Domanda vs. Produzione")
        ax.set_xlabel("Mesi")
        ax.set_ylabel("Unità")
        ax.legend()
        st.pyplot(fig)

        # 2) Stima costi totali su base mensile (fittizi, calcolati in base ai parametri)
        monthly_costs = []
        for i in range(5):
            # logica fittizia per costi totali
            raw_mat_cost = production_plan[i] * (manufacturing_cost_per_unit * supplier_cost_factor)
            logistics_cost = (production_plan[i] / 1000) * 1000 * logistics_cost_factor
            total_cost = raw_mat_cost + logistics_cost
            monthly_costs.append(total_cost)
        
        cost_df = pd.DataFrame({
            "Mese": months,
            "Costi Totali (€)": monthly_costs
        })

        st.table(cost_df)

        fig2, ax2 = plt.subplots()
        ax2.bar(months, monthly_costs, color="cadetblue")
        ax2.set_title("Costi Totali Stimati")
        ax2.set_xlabel("Mesi")
        ax2.set_ylabel("€ (migliaia)")
        st.pyplot(fig2)

        st.success("Simulazione completata con successo!")

    st.markdown("---")
    st.markdown("© 2024 - La tua Azienda. Tutti i diritti riservati.")

if __name__ == "__main__":
    main()
