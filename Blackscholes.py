# === LIBRARIES ===
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
# import plotly.graph_objects as go # Optional, not currently used

# === BLACK-SCHOLES FUNCTION ===
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility (annualized standard deviation of log returns)
        option_type (str): 'call' or 'put'

    Returns:
        float: Option price, or 0.0 if inputs are invalid or result is negative/NaN.
    """
    # Input validation: T and sigma must be positive
    epsilon = 1e-10 # Use a small epsilon to avoid division by zero or log(0) issues
    T = max(T, epsilon)
    sigma = max(sigma, epsilon)

    # Prevent calculation errors for non-positive stock/strike prices
    if S <= 0 or K <= 0:
        return 0.0 # Option price is effectively 0

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = np.nan # Initialize price as NaN
    try:
        if option_type.lower() == 'call':
            price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type.lower() == 'put':
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    except OverflowError:
        price = np.inf # Handle potential overflow
    except Exception:
         price = np.nan # Catch other potential math errors

    # Ensure price is not negative and handle NaN/inf cases
    if np.isnan(price) or np.isinf(price) or price < 0:
        return 0.0
    else:
        return price

# === STREAMLIT APP LAYOUT ===
st.set_page_config(layout="wide") # Use wide layout for better display

# === MAIN PANEL DISPLAY ===

st.title("Black-Scholes Pricing Model")

# --- Attribution Section --- Added as requested ---
st.markdown(f"""
<div style="margin-bottom: 15px; font-size: 14px;">
    <span style="font-weight: bold;">Created by:</span> Aniket Vasaikar<br>
    <a href="https://www.linkedin.com/in/aniketvasaikar/" target="_blank">Connect on LinkedIn</a>
</div>
""", unsafe_allow_html=True)
st.markdown("---") # Visual separator

# === SIDEBAR FOR INPUTS ===
st.sidebar.header("Model Parameters")

# --- Main Model Parameters ---
S = st.sidebar.number_input("Current Stock Price (S)", min_value=0.1, value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.01, max_value=5.0, value=0.25, step=0.01, format="%.2f")
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.20, step=0.01, format="%.2f")
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.00, max_value=0.5, value=0.05, step=0.005, format="%.3f")

# --- Heatmap Configuration ---
st.sidebar.header("Heatmap Parameters")
st.sidebar.write("Heatmap shows price variation based on Volatility and Spot Price.")

# Volatility Range for Heatmap
vol_min = st.sidebar.number_input("Min Volatility", min_value=0.01, max_value=1.0, value=0.05, step=0.01, format="%.2f")
vol_max = st.sidebar.number_input("Max Volatility", min_value=vol_min + 0.01, max_value=2.0, value=0.50, step=0.01, format="%.2f")
# Step slider removed - using fixed value below
vol_steps = 15  # <-- FIXED VALUE for volatility steps

# Spot Price Range for Heatmap
default_spot_min = max(0.1, S * 0.8)
default_spot_max = S * 1.2
spot_min = st.sidebar.number_input("Min Spot Price ($)", min_value=0.1, value=default_spot_min, step=1.0, format="%.2f")
spot_max = st.sidebar.number_input("Max Spot Price ($)", min_value=spot_min + 1.0, value=default_spot_max, step=1.0, format="%.2f")
# Step slider removed - using fixed value below
spot_steps = 15 # <-- FIXED VALUE for spot price steps

# === CALCULATIONS AND DISPLAY (MAIN PANEL CONTINUED) ===

# --- Calculate Single Option Prices ---
call_price = black_scholes(S, K, T, r, sigma, option_type='call')
put_price = black_scholes(S, K, T, r, sigma, option_type='put')

# --- Display Single Prices in Colored Boxes ---
col1, col2 = st.columns(2)

call_price_str = f"${call_price:.2f}"
put_price_str = f"${put_price:.2f}"

call_html = f"""
<div style="background-color: #5EB08A; padding: 15px; border-radius: 8px; text-align: center; color: white; line-height: 1.4;">
    <span style="font-size: 14px; opacity: 0.9;">Call Value</span><br>
    <strong style="font-size: 26px; font-weight: bold;">{call_price_str}</strong>
</div>
"""
put_html = f"""
<div style="background-color: #AC5B64; padding: 15px; border-radius: 8px; text-align: center; color: white; line-height: 1.4;">
    <span style="font-size: 14px; opacity: 0.9;">Put Value</span><br>
    <strong style="font-size: 26px; font-weight: bold;">{put_price_str}</strong>
</div>
"""
with col1:
    st.markdown(call_html, unsafe_allow_html=True)
with col2:
    st.markdown(put_html, unsafe_allow_html=True)

st.markdown("---") # Visual separator

# --- Generate Heatmap Data (Volatility vs Spot Price) ---
st.header("Options Price - Interactive Heatmap")

vol_range = np.linspace(vol_min, vol_max, vol_steps)
spot_range = np.linspace(spot_min, spot_max, spot_steps)

call_heatmap_data = np.zeros((vol_steps, spot_steps))
put_heatmap_data = np.zeros((vol_steps, spot_steps))

for i, v in enumerate(vol_range):
    for j, s_val in enumerate(spot_range):
        call_heatmap_data[i, j] = black_scholes(s_val, K, T, r, v, option_type='call')
        put_heatmap_data[i, j] = black_scholes(s_val, K, T, r, v, option_type='put')

vol_labels = [f"{v:.2f}" for v in vol_range]
spot_labels = [f"{s_val:.2f}" for s_val in spot_range]

call_heatmap_df = pd.DataFrame(call_heatmap_data, index=vol_labels, columns=spot_labels)
put_heatmap_df = pd.DataFrame(put_heatmap_data, index=vol_labels, columns=spot_labels)

call_heatmap_df.index.name = "Volatility"
call_heatmap_df.columns.name = "Spot Price ($)"
put_heatmap_df.index.name = "Volatility"
put_heatmap_df.columns.name = "Spot Price ($)"

# --- Display Heatmaps ---
col_hm1, col_hm2 = st.columns(2)

with col_hm1:
    st.subheader("Call Price Heatmap")
    # WARNING: 'blugrn' might not be a valid Plotly color scale name.
    # If errors occur, try standard names like 'BuGn', 'Greens', 'YlGnBu'.
    fig_call = px.imshow(
        call_heatmap_df,
        aspect="auto",
        color_continuous_scale='blugrn', # User specified color scale
        labels=dict(color="Call Price"),
        text_auto=".2f"
    )
    fig_call.update_layout(
        xaxis_title="Spot Price ($)",
        yaxis_title="Volatility (σ)",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_call.update_yaxes(tickangle=0)
    st.plotly_chart(fig_call, use_container_width=True)

with col_hm2:
    st.subheader("Put Price Heatmap")
    # WARNING: 'brwnyl' might not be a valid Plotly color scale name.
    # If errors occur, try standard names like 'YlOrRd', 'OrRd', 'Reds', 'Hot'.
    fig_put = px.imshow(
        put_heatmap_df,
        aspect="auto",
        color_continuous_scale='brwnyl', # User specified color scale
        labels=dict(color="Put Price"),
        text_auto=".2f"
    )
    fig_put.update_layout(
        xaxis_title="Spot Price ($)",
        yaxis_title="Volatility (σ)",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_put.update_yaxes(tickangle=0)
    st.plotly_chart(fig_put, use_container_width=True)

# --- Optional: Display Raw DataFrames ---
# Uncomment below to add an expander showing the data tables
# with st.expander("View Raw Heatmap Data"):
#     st.write("Call Heatmap Data (Volatility vs Spot Price):")
#     st.dataframe(call_heatmap_df.style.format("{:.2f}"))
#     st.write("Put Heatmap Data (Volatility vs Spot Price):")
#     st.dataframe(put_heatmap_df.style.format("{:.2f}"))