import streamlit as st
import yfinance as yf
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import re
from dotenv import load_dotenv
import sqlite3
from htmlTemplates import css, user_template, bot_template
 

def create_users_db():
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_user_to_db(email, password):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO Users (email, password)
        VALUES (?, ?)
    """
    cursor.execute(insert_query, (email, password))
    conn.commit()
    conn.close()


def authenticate_user(email, password):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    select_query = """
        SELECT * FROM Users WHERE email = ? AND password = ?
    """
    cursor.execute(select_query, (email, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return True
    else:
        return False


def init_ses_states():
    st.session_state.setdefault('chat_history', [])
    st.session_state.setdefault('user_authenticated', False)


def relative_returns(df):
    rel = df.pct_change()
    cumret = ((1 + rel).cumprod() - 1).fillna(0)
    return cumret


def display_convo():
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                 st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def approve_password(password):
    if len(password) >= 8 and re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[_@$#!?&*%])", password):
        return True
    return False
    

def approve_email(email):
    email_regex = '^[a-zA-Z0-9]+[\._]?[a-zA-Z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(email_regex, email):
        return True
    else:
        return False
    

def user_authentication_tab():
    with st.expander("User Authentication", expanded=True):
        login_tab, create_account_tab = st.tabs(["Login", "Create Account"])

        with login_tab:
            email = st.text_input("Email:") 
            password = st.text_input("Password:", type='password')
            if st.button("Login"):
                if authenticate_user(email=email,password=password):
                    st.session_state.user_authenticated = True
                else:
                    st.caption('Incorrect Username or Password.')

            if st.session_state.user_authenticated:
                st.caption("User Authenticated")

        with create_account_tab:
            new_email = st.text_input("New Email:")
            new_password = st.text_input("New Password:", type='password')
            confirm_password = st.text_input("Confirm Password:", type='password')
            if st.button("Create Account"):
                if not approve_email(new_email):
                    st.caption("Invalid Email")
                    return
                if not approve_password(new_password):
                    st.caption("Invalid Password")
                    return
                if new_password != confirm_password:
                    st.caption("Passwords do not match")
                    return
                add_user_to_db(email=new_email, password=new_password)
                st.caption(f"{new_email} Successfully Added")


def main():
    st.set_page_config(page_title="Stock Price AI Bot", page_icon=":chart:")
    st.write(css, unsafe_allow_html=True)
    create_users_db()
    init_ses_states()
    st.title("Stock Price AI Bot")
    st.caption("Visualizations and OpenAI Chatbot for Multiple Stocks Over A Specified Period")


    with st.sidebar:
        user_authentication_tab()
    
    
    if st.session_state.user_authenticated:
        with st.sidebar:
            with st.expander("Settings",expanded=True):
                asset_tickers = sorted(['DOW','NVDA','TSL','GOOGL','AMZN','AI','NIO','LCID','F','LYFY','AAPL', 'MSFT', 'BTC-USD', 'ETH-USD'])
                asset_dropdown = st.multiselect('Pick Assets:', asset_tickers)

                metric_tickers = ['Adj. Close', 'Relative Returns']
                metric_dropdown = st.selectbox("Metric", metric_tickers)

                viz_tickers = ['Line Chart', 'Area Chart']
                viz_dropdown = st.multiselect("Pick Charts:", viz_tickers)

                start = st.date_input('Start', value=pd.to_datetime('2023-01-01'))
                end = st.date_input('End', value=pd.to_datetime('today'))

                chatbot_temp = st.slider("Chat Bot Temperature",0.0,1.0,0.5)

        # Only when a stock is selected
        if len(asset_dropdown) > 0:
            df = yf.download(asset_dropdown,start,end)['Adj Close']
            if metric_dropdown == 'Relative Returns':
                df = relative_returns(df)
            if len(viz_dropdown) > 0:
                with st.expander("Data Visualizations for {} of {}".format(metric_dropdown,asset_dropdown), expanded=True):
                    if "Line Chart" in viz_dropdown:
                        st.subheader("Line Chart")
                        st.line_chart(df)
                    if "Area Chart" in viz_dropdown:
                        st.subheader("Area Chart")
                        st.area_chart(df)
            st.header("Chat with your Data") 

            query = st.text_input("Enter a query:")

            chat_prompt = f'''
                You are an AI ChatBot intended to help with user stock data.
                \nYou have access to a pandas dataframe with the following specifications 
                \nDATA MODE: {metric_dropdown}
                \nSTOCKS: {asset_dropdown} 
                \nTIME PERIOD: {start} to {end}
                \nCHAT HISTORY: {st.session_state.chat_history}
                \nUSER MESSAGE: {query}
                \nAI RESPONSE HERE:
            '''

            if st.button("Execute") and query:
                with st.spinner('Generating response...'):
                    try:
                        agent = create_pandas_dataframe_agent(
                            ChatOpenAI(temperature=chatbot_temp, model='gpt-4'),
                            pd.DataFrame(df),
                            verbose=True
                        )

                        answer = agent.run(chat_prompt)
                        st.session_state.chat_history.append(f"USER: {query}\n")
                        st.session_state.chat_history.append(f"AI: {answer}\n")
                        display_convo()

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    load_dotenv()
    main()

