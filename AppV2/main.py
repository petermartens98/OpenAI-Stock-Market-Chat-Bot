import streamlit as st
import yfinance as yf
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from htmlTemplates import css, user_template, bot_template


def init_ses_states():
    if 'generated' not in st.session_state: 
        st.session_state['generated'] = []
    if 'past' not in st.session_state: 
        st.session_state['past'] = []


def relret(df):
    rel = df.pct_change()
    cumret = (1+rel).cumprod()-1
    cumret = cumret.fillna(0)
    return cumret


def main():
    st.set_page_config(page_title="Stock Price AI Bot", page_icon=":chart:")
    st.write(css, unsafe_allow_html=True)
    init_ses_states()
    st.title("Stock Price AI Bot")
    st.caption("Visualizations and OpenAI Chatbot for Multiple Stocks Over A Specified Period")

    with st.sidebar:
        asset_tickers = ['TSL', 'AAPL', 'MSFT', 'BTC-USD', 'ETH-USD']
        asset_dropdown = st.multiselect('Pick Assets:', asset_tickers)

        metric_tickers = ['Adj. Close', 'Relative Returns']
        metric_dropdown = st.selectbox("Metric", metric_tickers)

        viz_tickers = ['Line Chart', 'Area Chart']
        viz_dropdown = st.multiselect("Pick Charts:", viz_tickers)

        start = st.date_input('Start', value=pd.to_datetime('2023-01-01'))
        end = st.date_input('End', value=pd.to_datetime('today'))

        chatbot_temp = st.slider("Chat Bot Temperature",0.0,1.0,0.1)


    # Only when a stock is selected
    if len(asset_dropdown) > 0:
        df = yf.download(asset_dropdown,start,end)['Adj Close']
        if metric_dropdown == 'Relative Returns':
            df = relret(df)
        if len(viz_dropdown) > 0:
            with st.expander("Data Visualizations for {} of {}".format(metric_dropdown,asset_dropdown), expanded=True):
                if "Line Chart" in viz_dropdown:
                    st.subheader("Line Chart")
                    st.line_chart(df)
                if "Area Chart" in viz_dropdown:
                    st.subheader("Area Chart")
                    st.area_chart(df)
        st.header("Chat with your Data") 

        # Accept input from user
        query = st.text_input("Enter a query:") 

        # Execute pandas response logic
        if st.button("Execute") and query:
            with st.spinner('Generating response...'):
                try:
                    agent = create_pandas_dataframe_agent(OpenAI(temperature=chatbot_temp),pd.DataFrame(df),verbose=True,) 
                    answer = agent.run(query)

                    st.session_state.past.append(query)
                    st.session_state.generated.append(answer)

                    for i in range(len(st.session_state.past)-1,-1,-1):
                        st.write(bot_template.replace("{{MSG}}",st.session_state.generated[i] ), unsafe_allow_html=True)
                        st.write(user_template.replace("{{MSG}}",st.session_state.past[i] ), unsafe_allow_html=True)
                        st.write("")


                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    load_dotenv()
    main()
