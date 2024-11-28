import joblib
import seaborn as sn
import matplotlib.pyplot as plt
import prophet as prophet
import pandas as pd
import streamlit as st

# Carregar o modelo treinado
model = joblib.load('prophet.joblib')

# Título da página
st.title("📊Previsão do Preço do Petróleo Brent")

# Número de dias para previsão
days = st.number_input("Quantos dias para prever?", min_value=1, max_value=10, value=5)

# Datafreme com previsãoes
future_dates = model.make_future_dataframe(periods=days)

forecast = model.predict(future_dates)

# Exibir os resultados da previsão
st.write(f"Previsão do preço em US$ para os próximos {days} dias:")

# Formatação Tabela
forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%d/%m/%Y')
forecast_table = forecast[['ds', 'yhat']].tail(days).reset_index(drop=True)

st.markdown("""
    <style>
        .table-container table {
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.write("### Data e Preço Previsto (US$)")
st.table(forecast_table.style.set_table_styles([{
    'selector': 'thead th',
    'props': [('text-align', 'center')]
}]))

# Gráfico com dados históricos e previsão
fig, ax = plt.subplots(figsize=(10, 6))

historical_data = forecast[['ds', 'yhat']].tail(15) #Intervalo de 15 dias passados
ax.plot(historical_data['ds'], historical_data['yhat'], label='Histórico (últimos 15 dias)', color='blue')

forecast_data = forecast[['ds', 'yhat']].tail(days) #Adicionando previsão
ax.plot(forecast_data['ds'], forecast_data['yhat'], label=f'Previsão para os próximos {days} dias', color='red', linestyle='--')

# Customizando o gráfico
ax.set_xlabel('Data')
ax.set_ylabel('Preço do Petróleo (US$)')
ax.set_title(f'Preço do Petróleo Brent - Histórico e Previsão')
ax.legend()
plt.xticks(rotation=45)

# Exibindo o gráfico
st.pyplot(fig)










