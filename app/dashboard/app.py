"""Streamlit dashboard for trading bot monitoring."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import httpx
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.data.repos import (
    get_equity_curve, get_recent_orders, get_recent_signals,
    get_open_positions
)
from app.trading.broker_alpaca import Broker
from app.core.settings import config
from app.core.utils import format_currency, format_percentage

# Page config
st.set_page_config(
    page_title="Algorithmic Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #00ff00; }
    .negative { color: #ff0000; }
    </style>
""", unsafe_allow_html=True)


def fetch_api_status():
    """Fetch status from API."""
    try:
        api_port = config.get('api', {}).get('port', 8000)
        response = httpx.get(f"http://localhost:{api_port}/api/status", timeout=5)
        return response.json()
    except:
        return None


def control_trading(action: str):
    """Send control command to API."""
    try:
        api_port = config.get('api', {}).get('port', 8000)
        response = httpx.post(
            f"http://localhost:{api_port}/api/control",
            json={"action": action},
            timeout=5
        )
        return response.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# Sidebar
st.sidebar.title("📈 Algo Trading Bot")
st.sidebar.markdown("---")

# Control buttons
st.sidebar.subheader("Trading Controls")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Start", use_container_width=True):
        result = control_trading("start")
        if result:
            st.success("Trading started!")

with col2:
    if st.button("⏸️ Pause", use_container_width=True):
        result = control_trading("pause")
        if result:
            st.warning("Trading paused!")

col3, col4 = st.sidebar.columns(2)

with col3:
    if st.button("▶️ Resume", use_container_width=True):
        result = control_trading("resume")
        if result:
            st.success("Trading resumed!")

with col4:
    if st.button("⏹️ Stop", use_container_width=True):
        result = control_trading("stop")
        if result:
            st.error("Trading stopped!")

st.sidebar.markdown("---")

# Settings
st.sidebar.subheader("Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Main content
st.title("📊 Trading Bot Dashboard")

# Status indicators
status = fetch_api_status()

if status:
    account = status.get('account', {})
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        equity = account.get('equity', 0)
        st.metric("Total Equity", format_currency(equity))
    
    with col2:
        # Calculate today's P&L (simplified)
        st.metric("Today's P&L", "$0.00", "0.00%")
    
    with col3:
        positions = account.get('open_positions', 0)
        st.metric("Open Positions", positions)
    
    with col4:
        buying_power = account.get('buying_power', 0)
        st.metric("Buying Power", format_currency(buying_power))
    
    with col5:
        status_emoji = "🟢" if status.get('running') and not status.get('paused') else "🔴"
        status_text = "Active" if status.get('running') and not status.get('paused') else "Inactive"
        st.metric("Status", f"{status_emoji} {status_text}")

else:
    st.warning("⚠️ Cannot connect to API server. Make sure it's running.")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Equity Curve",
    "💼 Positions",
    "📋 Orders",
    "🎯 Signals",
    "📊 Performance"
])

# Tab 1: Equity Curve
with tab1:
    st.subheader("Equity Curve")
    
    days = st.slider("Days to show", 1, 90, 30)
    
    try:
        df_equity = get_equity_curve(days=days)
        
        if not df_equity.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_equity['ts'],
                y=df_equity['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#00ff00', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            
            fig.update_layout(
                title="Account Equity Over Time",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity data available yet. Start trading to see your equity curve.")
    
    except Exception as e:
        st.error(f"Error loading equity curve: {e}")

# Tab 2: Positions
with tab2:
    st.subheader("Open Positions")
    
    try:
        df_positions = get_open_positions()
        
        if not df_positions.empty:
            # Format for display
            display_df = df_positions.copy()
            display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"${x:.2f}")
            display_df['unrealized_pl'] = display_df['unrealized_pl'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00"
            )
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # P&L chart
            if 'unrealized_pl' in df_positions.columns:
                fig = px.bar(
                    df_positions,
                    x='symbol',
                    y='unrealized_pl',
                    title="Unrealized P&L by Symbol",
                    color='unrealized_pl',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions")
    
    except Exception as e:
        st.error(f"Error loading positions: {e}")

# Tab 3: Orders
with tab3:
    st.subheader("Recent Orders")
    
    order_limit = st.selectbox("Show orders", [50, 100, 200], index=1)
    
    try:
        df_orders = get_recent_orders(limit=order_limit)
        
        if not df_orders.empty:
            # Format for display
            display_df = df_orders.copy()
            
            # Format prices
            if 'fill_price' in display_df.columns:
                display_df['fill_price'] = display_df['fill_price'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "-"
                )
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No orders yet")
    
    except Exception as e:
        st.error(f"Error loading orders: {e}")

# Tab 4: Signals
with tab4:
    st.subheader("Recent Trading Signals")
    
    signal_limit = st.selectbox("Show signals", [20, 50, 100], index=1)
    
    try:
        df_signals = get_recent_signals(limit=signal_limit)
        
        if not df_signals.empty:
            # Color code by side
            def highlight_side(val):
                if val == 'LONG':
                    return 'background-color: #90ee90'
                elif val == 'SHORT':
                    return 'background-color: #ffcccb'
                else:
                    return ''
            
            styled_df = df_signals.style.applymap(
                highlight_side,
                subset=['side'] if 'side' in df_signals.columns else []
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Signal distribution
            if 'side' in df_signals.columns:
                signal_counts = df_signals['side'].value_counts()
                
                fig = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title="Signal Distribution",
                    color=signal_counts.index,
                    color_discrete_map={'LONG': '#90ee90', 'SHORT': '#ffcccb', 'FLAT': '#d3d3d3'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signals yet")
    
    except Exception as e:
        st.error(f"Error loading signals: {e}")

# Tab 5: Performance
with tab5:
    st.subheader("Performance Metrics")
    
    try:
        df_orders = get_recent_orders(limit=1000)
        
        if not df_orders.empty and 'fill_price' in df_orders.columns:
            # Calculate some basic metrics
            filled_orders = df_orders[df_orders['status'] == 'filled']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Orders", len(df_orders))
                st.metric("Filled Orders", len(filled_orders))
            
            with col2:
                if 'symbol' in df_orders.columns:
                    unique_symbols = df_orders['symbol'].nunique()
                    st.metric("Symbols Traded", unique_symbols)
            
            with col3:
                # Most traded symbol
                if 'symbol' in df_orders.columns:
                    top_symbol = df_orders['symbol'].mode()[0] if len(df_orders) > 0 else "N/A"
                    st.metric("Most Traded", top_symbol)
            
            # Order volume by symbol
            if 'symbol' in df_orders.columns:
                symbol_counts = df_orders['symbol'].value_counts().head(10)
                
                fig = px.bar(
                    x=symbol_counts.index,
                    y=symbol_counts.values,
                    title="Order Volume by Symbol (Top 10)",
                    labels={'x': 'Symbol', 'y': 'Number of Orders'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start trading to see performance metrics")
    
    except Exception as e:
        st.error(f"Error loading performance data: {e}")

# Debug info
if show_debug:
    st.markdown("---")
    st.subheader("🔧 Debug Information")
    
    with st.expander("API Status"):
        st.json(status)
    
    with st.expander("Configuration"):
        st.json(config)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

