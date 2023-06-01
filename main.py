# Imports
import streamlit as st
import yfinance as yf
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import os
from  apikey import apikey

# Define OpenAI API KEY
os.environ['OPENAI_API_KEY'] = apikey

# CSS styling
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 85%;
  padding: 0 1.25rem;
  color: #fff;
}
'''
st.write(css, unsafe_allow_html=True)

# Define user and bot templates
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://resizing.flixster.com/ocuc8yjm8Fu5UK5Ze8lbdp58m9Y=/300x300/v2/https://flxt.tmsimg.com/assets/p11759522_i_h9_aa.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.shutterstock.com/image-vector/nerd-robot-vector-illustration-version-260nw-2126944997.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# App Title
st.title("Stock Price Relative Returns AI Tool")
st.caption("Visualizations and OpenAI Chatbot Comparing Multiple Stocks Over A Specified Period")

# Define Stocke Drop Down Menu and Available Stock to choose from
tickers = ['TSL', 'AAPL', 'MSFT', 'BTC-USD', 'ETH-USD']
dropdown = st.multiselect('Pick Asset:', tickers)

start = st.date_input('Start', value=pd.to_datetime('2023-01-01'))
end = st.date_input('End', value=pd.to_datetime('today'))

# Define function to convert price to relative returns
def relret(df):
    rel = df.pct_change()
    cumret = (1+rel).cumprod()-1
    cumret = cumret.fillna(0)
    return cumret

# Only when a stock is selected
if len(dropdown) > 0:
    #df = yf.download(dropdown,start,end)['Adj Close']
    df = relret(yf.download(dropdown,start,end)['Adj Close'])
    st.header("Relative Returns of {}".format(dropdown))
    st.line_chart(df)
    st.header("Chat with your Data") 

     # Accept input from user
    query = st.text_input("Enter a query:") 

    # Execute pandas response logic
    if st.button("Execute") and query:
        with st.spinner('Generating response...'):
            try:

                 # Define pandas df agent - 0 ~ no creativity vs 1 ~ very creative
                agent = create_pandas_dataframe_agent(OpenAI(temperature=0.0),pd.DataFrame(df),verbose=True) 

                # Run agent and retrieve answer
                answer = agent.run(query)

                # Display user query and agents answer
                st.write(user_template.replace("{{MSG}}",query ), unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}", answer ), unsafe_allow_html=True)
                st.write("")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

