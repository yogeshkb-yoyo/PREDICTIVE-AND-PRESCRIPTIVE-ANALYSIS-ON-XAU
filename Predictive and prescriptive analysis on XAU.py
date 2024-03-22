from keras.layers import Input, Dense, GRU, Conv1D, Concatenate
from tensorflow import keras
from keras.models import Model
from sklearn.model_selection import train_test_split
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import google.generativeai as genai
from gtts import gTTS
from keras.models import Sequential
from keras.layers import Conv1D, GRU, Dense
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import TimeSeriesSplit
import concurrent.futures
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from scipy.stats import ttest_ind
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

TICKER, SEQ_LENGTH, SHORT_WINDOW, LONG_WINDOW = 'GOLDBEES.BO', 10, 5, 20
end_date, start_date = pd.to_datetime(
    'today'), pd.to_datetime('today') - pd.DateOffset(years=2)


def get_historical_data(t, s, e):
    return yf.download(t, start=s, end=e)['Adj Close']


def collect_data(sym, s, e):
    return yf.download(sym, start=s, end=e)


def enhanced_eda(d):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
                        'Gold Price Over Time (USD)', 'Gold Price with Rolling Statistics'])
    trace1 = go.Scatter(
        x=d.index, y=d['Close'], mode='lines', name='Gold Price (USD)')
    fig.add_trace(trace1, row=1, col=1)
    trace2 = go.Scatter(x=d.index, y=d['Close'].rolling(window=30).mean(
    ), mode='lines', name='30-Day Rolling Mean', line=dict(dash='dash'))
    trace3 = go.Scatter(x=d.index, y=d['Close'].rolling(window=30).std(
    ), mode='lines', name='30-Day Rolling Std', line=dict(dash='dash'))
    fig.add_trace(trace2, row=2, col=1)
    fig.add_trace(trace3, row=2, col=1)
    fig.update_layout(title_text='Gold Price Over Time and Rolling Statistics',
                      xaxis_title='Date', yaxis_title='Gold Price (USD)', showlegend=True)
    fig.show()


def perform_statistical_analysis(d, sd):
    t_stat, p_value = ttest_ind(
        d[d.index < sd]['Close'], d[d.index >= sd]['Close'])
    print(f'\nT-Statistic: {t_stat}, P-Value: {p_value}')


def calculate_risk_metrics(d, sd, sym):
    print(d['Close'].agg({'mean', 'median', 'std'}).round(2))
    fig = px.histogram(d, x='Close', nbins=30, labels={
                       'Close': 'Gold Price (USD)'}, title='Distribution of Gold Prices (USD)')
    fig.update_layout(xaxis_title='Gold Price (USD)', yaxis_title='Frequency')
    fig.show()
    t_stat_r, p_value_r = ttest_ind(
        d[d.index < sd]['Close'], d[d.index >= sd]['Close'])
    print(
        f'\nT-Statistic for Daily Returns: {t_stat_r}, P-Value for Daily Returns: {p_value_r}')


def fetch_prices(ticker, start, end):
    return yf.download(ticker, start=start, end=end)[['Close', 'High', 'Low']]


def preprocess_and_create_sequences(data, seq_length):
    scalers, preprocessed_data = preprocess_data(data)
    X, y = create_sequences(
        preprocessed_data[['Close', 'High', 'Low']], seq_length)
    # Reshape input data for compatibility with Conv1D layer
    # Assuming 3 features (Close, High, Low)
    X = X.reshape(X.shape[0], X.shape[1], 3)
    return X, y, scalers


def preprocess_data(data):
    scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in data.columns}
    preprocessed_data = data.copy()
    for col in data.columns:
        if col in scalers:
            preprocessed_data[col] = scalers[col].fit_transform(
                np.array(data[col]).reshape(-1, 1)).flatten()
    return scalers, preprocessed_data


def create_sequences(data, seq_length):
    return np.array([data.iloc[i:i + seq_length].values for i in range(len(data) - seq_length)]), np.array([data.iloc[i + seq_length].values for i in range(len(data) - seq_length)])


def build_cnn_gru_model(seq_length, input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu',
               input_shape=(seq_length, 3)),
        GRU(50, return_sequences=True), 
        GRU(50),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_evaluate_cnn_gru_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=1)
    y_pred = model.predict(X_test)
    mse, mae = mean_squared_error(
        y_test, y_pred), mean_absolute_error(y_test, y_pred)
    acc_percentage = (
        np.sum(np.abs((y_test - y_pred) / y_test) < 0.0125   ) / len(y_test)) * 100
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, Accuracy: {acc_percentage:.2f}%')
    return model, y_pred


def inverse_transform_predictions(predictions, scalers, columns):
    return pd.DataFrame({col: scalers[col].inverse_transform(predictions[:, i].reshape(-1, 1)).flatten() for i, col in enumerate(columns)})


def text_to_speech(text, language='en', filename='news.mp3'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filename)
    return filename


symbol, split_date = 'GOLD', '2022-06-01'
gold_data = collect_data(symbol, start_date, end_date)
enhanced_eda(gold_data)
perform_statistical_analysis(gold_data, split_date)
calculate_risk_metrics(gold_data, split_date, symbol)

gold_prices = fetch_prices(TICKER, start_date, end_date)
X, y, _ = preprocess_and_create_sequences(gold_prices, SEQ_LENGTH)

genai.configure(api_key="_API_")


model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(
    "what are the current news about gold market? and suggest sell or buy for gold based real world news")
print(response.text)
text_to_speech(response.text)

split_index = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Example usage:
with concurrent.futures.ThreadPoolExecutor() as executor:
    cnn_gru_model = build_cnn_gru_model(SEQ_LENGTH, gold_prices.shape[1])
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 3)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 3)
    trained_model, cnn_gru_predictions = train_evaluate_cnn_gru_model(
        cnn_gru_model, X_train_reshaped, y_train, X_test_reshaped, y_test)

# Use the model for predictions
predictions_2 = cnn_gru_model.predict(X_test_reshaped)


cnn_gru_predictions_df = inverse_transform_predictions(
    cnn_gru_predictions, _, gold_prices.columns)


model_2 = cnn_gru_model
trained_model_2, gru_predictions_2 = train_evaluate_cnn_gru_model(
    model_2, X_train, y_train, X_test, y_test)

ensemble_predictions = (cnn_gru_predictions + gru_predictions_2) / 2

ensemble_predictions_df = inverse_transform_predictions(
    ensemble_predictions, _, gold_prices.columns)

last_close_price_ensemble = ensemble_predictions_df['Close'].iloc[-1]
print(f"Last Close Price (Ensemble): {last_close_price_ensemble:.2f}")

next_day_prediction_ensemble = ensemble_predictions_df.iloc[-1]
next_day_high_ensemble, next_day_low_ensemble, next_day_close_ensemble = next_day_prediction_ensemble[
    'High'], next_day_prediction_ensemble['Low'], next_day_prediction_ensemble['Close']

print(f"Predicted Next Day High (Ensemble): {next_day_high_ensemble:.2f}")
print(f"Predicted Next Day Low (Ensemble): {next_day_low_ensemble:.2f}")
print(f"Predicted Next Day Close (Ensemble): {next_day_close_ensemble:.2f}")

short_mavg, long_mavg = gold_prices['Close'].rolling(window=SHORT_WINDOW, min_periods=1).mean(
), gold_prices['Close'].rolling(window=LONG_WINDOW, min_periods=1).mean()

buy_signal, sell_signal = short_mavg.iloc[-1] > long_mavg.iloc[-1] and ensemble_predictions_df['Close'].iloc[
    0] > short_mavg.iloc[-1], short_mavg.iloc[-1] < long_mavg.iloc[-1] and ensemble_predictions_df['Close'].iloc[0] < short_mavg.iloc[-1]
print(
    f"Suggestion: {'Buy' if buy_signal else 'Sell' if sell_signal else 'Hold'}")

prediction = ensemble_predictions_df['Close'].iloc[-1] if short_mavg.iloc[-1] > long_mavg.iloc[-1] else ensemble_predictions_df['Close'].iloc[-1]
buy_signal, sell_signal = gold_prices['Close'].shift(1) < long_mavg.shift(
    1), gold_prices['Close'].shift(1) > long_mavg.shift(1)

buy_signals, sell_signals = gold_prices['Close'][buy_signal], gold_prices['Close'][sell_signal]

fig = go.Figure()
fig.add_trace(go.Scatter(x=gold_prices.index,
              y=gold_prices['Close'], mode='lines', name='Actual Prices', line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=short_mavg.index, y=short_mavg,
              mode='lines', name=f'Short-term MA ({SHORT_WINDOW} days)'))
fig.add_trace(go.Scatter(x=long_mavg.index, y=long_mavg,
              mode='lines', name=f'Long-term MA ({LONG_WINDOW} days)'))
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals,
              mode='markers', name='Buy Signal', marker=dict(color='green')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals,
              mode='markers', name='Sell Signal', marker=dict(color='red')))
fig.add_trace(go.Scatter(x=[gold_prices.index[-1], gold_prices.index[-1] + pd.Timedelta(days=1)], y=[
              gold_prices['Close'].iloc[-1], prediction], mode='lines', name='Next Day\'s Prediction', line=dict(color='blue', width=2)))

fig.update_layout(title='Gold Prices with Moving Averages and Buy/Sell Signals', xaxis_title='Date', yaxis_title='Gold Price', showlegend=True,
                  legend=dict(x=0.02, y=0.98), xaxis=dict(showgrid=True), yaxis=dict(range=[gold_prices['Close'].min() - 5, gold_prices['Close'].max() + 5]))
fig.show()

ticker_names = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones', 'USDJPY=X': 'USD to JPY',
                'USDEUR=X': 'USD to EUR', 'CL=F': 'Crude Oil', 'BTC-USD': 'Bitcoin', 'GLD': 'Gold'}
data = {ticker_names[t]: get_historical_data(
    t, start_date, end_date) for t in ticker_names}
df = pd.DataFrame(data)
corr_matrix = df.corr()

heatmap_plotly = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns,
                           y=corr_matrix.index, colorscale='Viridis', colorbar=dict(title='Correlation')))
heatmap_plotly.update_layout(
    title='Correlation Heatmap - Gold and Affecting Factors', xaxis_title='Factors', yaxis_title='Factors')
heatmap_plotly.show()

buy_signals, sell_signals = gold_prices['Close'][buy_signal], gold_prices['Close'][sell_signal]

all_signals = pd.Series(0, index=gold_prices.index)
all_signals[buy_signals.index] = 1
all_signals[sell_signals.index] = -1


def backtest_strategy_plotly(prices, signals):
    df = pd.DataFrame({'Price': prices, 'Signal': signals})
    df['Returns'] = df['Price'].pct_change()
    df['StrategyReturns'] = df['Returns'] * df['Signal'].shift(1)
    df['CumulativeReturns'] = (1 + df['StrategyReturns']).cumprod()
    annualized_return = (df['CumulativeReturns'][-1]) ** (252 / len(df)) - 1
    sharpe_ratio = (df['StrategyReturns'].mean() * 252) / \
        (df['StrategyReturns'].std() * np.sqrt(252))

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        subplot_titles=["Trading Signals"])
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Price'], mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Signal'] == 1].index, y=df[df['Signal'] == 1]['Price'],
                  mode='markers', marker=dict(color='green', size=8), name='Buy Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Signal'] == -1].index, y=df[df['Signal'] == -1]['Price'],
                  mode='markers', marker=dict(color='red', size=8), name='Sell Signal'), row=1, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)

    fig.update_layout(title_text='Backtesting Results',
                      xaxis_title='Date', height=800, showlegend=True)
    fig.show()
    return df


backtest_results = backtest_strategy_plotly(gold_prices['Close'], all_signals)


# Assuming 'gold_prices' is your DataFrame containing gold prices

# Create Antecedent/Consequent objects representing input/output variables
price_change = ctrl.Antecedent(np.arange(-5, 5, 0.1), 'price_change')
buy_signal_strength = ctrl.Consequent(
    np.arange(0, 101, 1), 'buy_signal_strength')

# Define fuzzy sets and membership functions
price_change['negative'] = fuzz.trimf(price_change.universe, [-5, -5, 0])
price_change['zero'] = fuzz.trimf(price_change.universe, [-1, 0, 1])
price_change['positive'] = fuzz.trimf(price_change.universe, [0, 5, 5])

buy_signal_strength['weak'] = fuzz.trimf(
    buy_signal_strength.universe, [0, 25, 50])
buy_signal_strength['moderate'] = fuzz.trimf(
    buy_signal_strength.universe, [25, 50, 75])
buy_signal_strength['strong'] = fuzz.trimf(
    buy_signal_strength.universe, [50, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(price_change['negative'], buy_signal_strength['strong'])
rule2 = ctrl.Rule(price_change['zero'], buy_signal_strength['moderate'])
rule3 = ctrl.Rule(price_change['positive'], buy_signal_strength['weak'])

# Create control system
buy_signal_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
buy_signal_simulation = ctrl.ControlSystemSimulation(buy_signal_ctrl)

# Simulate fuzzy logic based on the change in gold prices
# assuming 'Close' column represents prices
price_change_value = gold_prices['Close'].pct_change().iloc[-1] * 100
buy_signal_simulation.input['price_change'] = price_change_value

# Compute the result
buy_signal_simulation.compute()

# Access the result
buy_strength = buy_signal_simulation.output['buy_signal_strength']

# Print the fuzzy logic result
print(f'Fuzzy Logic Buy Signal Strength: {buy_strength:.2f}%')
