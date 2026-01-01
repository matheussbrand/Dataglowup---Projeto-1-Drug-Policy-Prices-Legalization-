import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA


# ======================================================
# FUNÇÕES DE DETECÇÃO ROBUSTA (UNODC SAFE)
# ======================================================
def detect_country_column(df):
    candidates = [
        "Country", "Country_Territory", "Country/Territory",
        "Country_or_territory", "Territory"
    ]
    for col in df.columns:
        for c in candidates:
            if col.lower() == c.lower():
                return col
    return None


def detect_year_column(df):
    candidates = ["Year", "Reference_year", "Reference_period"]
    for col in df.columns:
        for c in candidates:
            if col.lower() == c.lower():
                return col
    return None


# ======================================================
# CLASSIFICAÇÃO ECONÔMICA (OCDE)
# ======================================================
OECD_COUNTRIES = {
    "United States", "Canada", "Mexico",
    "Germany", "France", "United Kingdom", "Italy", "Spain",
    "Netherlands", "Belgium", "Luxembourg",
    "Switzerland", "Austria", "Sweden", "Norway", "Denmark",
    "Finland", "Ireland", "Portugal", "Greece",
    "Japan", "South Korea", "Australia", "New Zealand",
    "Israel", "Turkey", "Chile", "Colombia",
    "Poland", "Czech Republic", "Slovakia", "Hungary",
    "Estonia", "Latvia", "Lithuania", "Slovenia"
}

def classify_country(country):
    if country in OECD_COUNTRIES:
        return "Desenvolvido"
    return "Em desenvolvimento"


# ======================================================
# DRUG TAGS (PADRONIZAÇÃO)
# ======================================================
DRUG_TAGS = {
    "MDMA": ["mdma", "ecstasy"],
    "Cocaine": ["cocaine"],
    "Heroin": ["heroin"],
    "Cannabis": ["cannabis", "marijuana", "hashish"],
    "Amphetamine": ["amphetamine", "ats"]
}

def tag_drug(name):
    if pd.isna(name):
        return None
    name = str(name).lower()
    for tag, keys in DRUG_TAGS.items():
        if any(k in name for k in keys):
            return tag
    return None


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Drug Policy Analysis", layout="wide")
st.title("💊 Drug Policy, Prices & Legalization")
st.caption("Fonte: UNODC | Projeto de Portfólio em Ciência de Dados")


# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return (
        pd.read_excel("data/prices.xlsx"),
        pd.read_excel("data/seizures.xlsx"),
        pd.read_excel("data/crimes.xlsx")
    )

prices, seizures, crimes = load_data()


# ======================================================
# ETL — PRICES
# ======================================================
prices.columns = prices.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")

year_col = detect_year_column(prices)
country_col = detect_country_column(prices)

if not year_col or not country_col:
    st.error("❌ Colunas obrigatórias não encontradas em prices.")
    st.write(prices.columns.tolist())
    st.stop()

prices = prices.rename(columns={
    year_col: "Year",
    country_col: "Country",
    "Typical_USD": "Price_USD"
})

prices["Year"] = pd.to_numeric(prices["Year"], errors="coerce")
prices["Price_USD"] = pd.to_numeric(prices["Price_USD"], errors="coerce")

prices = prices.dropna(subset=["Year", "Price_USD", "Drug", "Region"])
prices["Drug_Tag"] = prices["Drug"].apply(tag_drug)
prices["Economic_Status"] = prices["Country"].apply(classify_country)


# ======================================================
# ETL — SEIZURES
# ======================================================
seizures.columns = seizures.columns.str.strip().str.replace(" ", "_")

year_col = detect_year_column(seizures)
country_col = detect_country_column(seizures)

seizures = seizures.rename(columns={
    year_col: "Year",
    country_col: "Country"
})

seizures["Year"] = pd.to_numeric(seizures["Year"], errors="coerce")
seizures["Kilograms"] = pd.to_numeric(seizures["Kilograms"], errors="coerce")

seizures = seizures.dropna(subset=["Year", "Kilograms", "DrugName", "Region", "Country"])
seizures["Drug_Tag"] = seizures["DrugName"].apply(tag_drug)
seizures["Economic_Status"] = seizures["Country"].apply(classify_country)


# ======================================================
# ETL — CRIMES
# ======================================================
crimes.columns = crimes.columns.str.strip().str.replace(" ", "_").str.replace(":", "")

year_col = detect_year_column(crimes)
country_col = detect_country_column(crimes)

crimes = crimes.rename(columns={
    year_col: "Year",
    country_col: "Country"
})

crimes["Year"] = pd.to_numeric(crimes["Year"], errors="coerce")
crimes = crimes.dropna(subset=["Year", "Country"])
crimes["Economic_Status"] = crimes["Country"].apply(classify_country)


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("🎛️ Filtros")

# ⬇️ EDITE AQUI para mudar quais drogas aparecem
drug_tags = st.sidebar.multiselect(
    "Droga",
    sorted(prices["Drug_Tag"].dropna().unique()),
    default=sorted(prices["Drug_Tag"].dropna().unique())
)

# ⬇️ EDITE AQUI para mudar classificação econômica
econ_status = st.sidebar.multiselect(
    "Status Econômico",
    ["Desenvolvido", "Em desenvolvimento"],
    default=["Desenvolvido", "Em desenvolvimento"]
)

# ⬇️ EDITE AQUI para permitir seleção manual de países
countries = st.sidebar.multiselect(
    "País",
    sorted(prices["Country"].dropna().unique())
)


# ======================================================
# DATASET FILTRADO
# ======================================================
df_price = prices[
    (prices["Drug_Tag"].isin(drug_tags)) &
    (prices["Economic_Status"].isin(econ_status))
]

if countries:
    df_price = df_price[df_price["Country"].isin(countries)]
# ======================================================
# KPIs
# ======================================================
st.subheader("📊 KPIs Econômicos")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Preço Médio (USD)", f"${df_price['Price_USD'].mean():.2f}")
c2.metric("Países", df_price["Country"].nunique())
c3.metric("Anos", df_price["Year"].nunique())
c4.metric("Status Econômicos", df_price["Economic_Status"].nunique())


# ======================================================
# BOXPLOT — PREÇOS POR DROGA
# ======================================================
st.subheader("📦 Distribuição de Preços por Droga")

fig_box = px.box(
    df_price,
    x="Drug_Tag",
    y="Price_USD",
    color="Drug_Tag"
)
st.plotly_chart(fig_box, use_container_width=True)


# ======================================================
# PRICE TREND
# ======================================================
st.subheader("📈 Evolução de Preços")

price_trend = (
    df_price
    .groupby(["Year", "Economic_Status"], as_index=False)
    .agg(Price_USD=("Price_USD", "mean"))
)

fig_price = px.line(
    price_trend,
    x="Year",
    y="Price_USD",
    color="Economic_Status",
    markers=True
)
st.plotly_chart(fig_price, use_container_width=True)


# ======================================================
# REPRESSION VS PRICE
# ======================================================
st.subheader("⚖️ Repressão vs Preço")

seiz_reg = (
    seizures[
        (seizures["Drug_Tag"].isin(drug_tags)) &
        (seizures["Economic_Status"].isin(econ_status))
    ]
    .groupby(["Year", "Economic_Status"], as_index=False)
    .agg(Kilograms=("Kilograms", "sum"))
)

rep_merge = price_trend.merge(
    seiz_reg,
    on=["Year", "Economic_Status"],
    how="inner"
)

fig_rep = px.scatter(
    rep_merge,
    x="Kilograms",
    y="Price_USD",
    color="Economic_Status",
    trendline="ols"
)
st.plotly_chart(fig_rep, use_container_width=True)


# ======================================================
# EVOLUÇÃO DAS APREENSÕES
# ======================================================
st.subheader("📦 Evolução das Apreensões")

fig_seiz_time = px.line(
    seiz_reg,
    x="Year",
    y="Kilograms",
    color="Economic_Status",
    markers=True
)
st.plotly_chart(fig_seiz_time, use_container_width=True)


# ======================================================
# RANKING DE PAÍSES — TRÁFICO (PROXY)
# ======================================================
st.subheader("🏆 Ranking de Países — Atividade de Tráfico (Proxy)")

seiz_country = seizures.groupby("Country", as_index=False)["Kilograms"].sum()
crime_country = crimes.groupby("Country", as_index=False)["Calculated_total"].sum()

traffic_rank = seiz_country.merge(crime_country, on="Country", how="inner")

traffic_rank["Traffic_Score"] = (
    traffic_rank["Kilograms"] / traffic_rank["Kilograms"].max() +
    traffic_rank["Calculated_total"] / traffic_rank["Calculated_total"].max()
)

top_traffic = traffic_rank.sort_values("Traffic_Score", ascending=False).head(15)

fig_rank = px.bar(
    top_traffic,
    x="Traffic_Score",
    y="Country",
    orientation="h"
)
st.plotly_chart(fig_rank, use_container_width=True)


# ======================================================
# MAPA MUNDIAL — TRÁFICO (PROXY)
# ======================================================
st.subheader("🌍 Mapa Mundial — Intensidade do Tráfico (Proxy)")

fig_map = px.choropleth(
    traffic_rank,
    locations="Country",
    locationmode="country names",
    color="Traffic_Score",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig_map, use_container_width=True)


# ======================================================
# SIMULAÇÃO DE RECEITA TRIBUTÁRIA
# ======================================================
st.subheader("💰 Simulação de Receita Tributária por País (Ranking Dinâmico)")

colA, colB, colC = st.columns(3)

tax_rate = colA.slider("Imposto (%)", 5, 50, 20)
price_reduction = colB.slider("Redução de Preço (%)", 0, 60, 30)
top_n = colC.slider("Top países", 5, 30, 15)

# 🔁 BASE REALMENTE FILTRADA (isso é a chave)
sim_base = df_price.copy()

# 🔢 AGREGAÇÃO POR PAÍS (VARIÁVEL)
country_sim = (
    sim_base
    .groupby("Country", as_index=False)
    .agg(
        Avg_Price=("Price_USD", "mean"),
        Transactions=("Price_USD", "count")
    )
)

# ❗ GARANTE VARIABILIDADE ENTRE PAÍSES
country_sim["Estimated_Volume"] = country_sim["Transactions"] * 800  # proxy realista

# 💰 SIMULAÇÃO ECONÔMICA
country_sim["Legal_Price"] = country_sim["Avg_Price"] * (1 - price_reduction / 100)
country_sim["Tax_Revenue"] = (
    country_sim["Legal_Price"] *
    (tax_rate / 100) *
    country_sim["Estimated_Volume"]
)

# 🔥 RANKING REFEITO A CADA INTERAÇÃO
country_sim = (
    country_sim
    .sort_values("Tax_Revenue", ascending=False)
    .head(top_n)
)

# 📊 GRÁFICO — AGORA MUDA PAÍS + ORDEM + VALOR
fig_tax = px.bar(
    country_sim,
    x="Tax_Revenue",
    y="Country",
    orientation="h",
    title="Ranking Dinâmico — Receita Tributária Estimada por País",
    labels={"Tax_Revenue": "Receita Estimada (USD)"}
)

fig_tax.update_layout(
    yaxis=dict(categoryorder="total ascending")
)

st.plotly_chart(fig_tax, use_container_width=True)

st.metric(
    "💵 Receita Total (Top Países)",
    f"${country_sim['Tax_Revenue'].sum():,.0f}"
)

# ======================================================
# TENDÊNCIA DE PREÇOS (RESPEITA TODOS OS FILTROS)
# ======================================================
st.subheader("📈 Tendência de Preços ao Longo do Tempo")

trend = (
    df_price
    .groupby(["Year", "Country"], as_index=False)
    .agg(Price_USD=("Price_USD", "mean"))
)

fig_trend = px.line(
    trend,
    x="Year",
    y="Price_USD",
    color="Country",
    markers=True
)

st.plotly_chart(fig_trend, use_container_width=True)


# ======================================================
# STORYTELLING FINAL
# ======================================================
st.subheader("📌 Conclusão — Vale a pena legalizar?")

st.markdown("""
- Repressão isolada **não reduz preços de forma consistente**
- Países desenvolvidos e em desenvolvimento apresentam **padrões semelhantes**
- O mercado ilegal captura integralmente a renda
- A legalização redireciona valor para o Estado
- Evidências sugerem que **regular é mais eficiente do que proibir**
""")

st.caption("Projeto de Portfólio | Data Science | UNODC | MATHEUS BRANDÃO | linkedin.com/in/matheussbrandao/")
