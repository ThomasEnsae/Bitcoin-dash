import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
import datetime
import pandas as pd
import time
from typing import List, Dict, Any, Tuple, Optional


class CoinbaseDataFetcher:
    """Class to fetch and process historical cryptocurrency data from Coinbase Exchange API."""

    BASE_URL = "https://api.exchange.coinbase.com/products"

    def __init__(self, product_id: str, granularity: int, max_candles: int = 300):
        """
        Initialize the data fetcher with specific parameters.

        Args:
            product_id: The asset pair (e.g., 'BTC-USD')
            granularity: The candle duration in seconds
            max_candles: Maximum number of candles per API request
        """
        self.product_id = product_id
        self.granularity = granularity
        self.max_candles = max_candles
        self.time_delta = datetime.timedelta(seconds=granularity * max_candles)

    def fetch_candles(self, start: datetime.datetime, end: datetime.datetime) -> List[List]:
        """
        Fetch historical candles from Coinbase Exchange for a specific time period.

        Args:
            start: Datetime object representing the start time
            end: Datetime object representing the end time

        Returns:
            List of candle data

        Raises:
            Exception: If the API returns an error message
        """
        url = f"{self.BASE_URL}/{self.product_id}/candles"
        params = {
            'start': start.isoformat(),
            'end': end.isoformat(),
            'granularity': self.granularity
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Handle error messages from the API
        if isinstance(data, dict) and data.get("message"):
            raise Exception(f"Error fetching data: {data['message']}")

        return data

    def fetch_historical_data(self, start_date: datetime.datetime,
                              end_date: Optional[datetime.datetime] = None) -> pd.DataFrame:
        """
        Fetch all historical data between start and end dates.

        Args:
            start_date: Starting date for data collection
            end_date: Ending date for data collection (defaults to current time)

        Returns:
            DataFrame containing processed historical data
        """
        if end_date is None:
            end_date = datetime.datetime.now()

        all_data = []
        current_start = start_date

        with st.spinner(f"Fetching {self.product_id} data from {start_date.date()} to {end_date.date()}"):
            progress_bar = st.progress(0)
            total_days = (end_date - start_date).days
            days_processed = 0

            while current_start < end_date:
                current_end = min(current_start + self.time_delta, end_date)

                try:
                    candles = self.fetch_candles(current_start, current_end)
                    all_data.extend(candles)
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    break

                # Move the window forward
                current_start = current_end

                # Update progress
                days_processed = (current_start - start_date).days
                progress = min(days_processed / max(1, total_days), 1.0)
                progress_bar.progress(progress)

                # Respect API rate limits
                time.sleep(0.35)

        # Convert to DataFrame and process
        return self._process_candle_data(all_data)

    def _process_candle_data(self, candle_data: List[List]) -> pd.DataFrame:
        """
        Process raw candle data into a structured DataFrame.

        Args:
            candle_data: Raw candle data from API

        Returns:
            Processed DataFrame
        """
        if not candle_data:
            return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])

        # API returns: [time, low, high, open, close, volume]
        df = pd.DataFrame(candle_data, columns=["time", "low", "high", "open", "close", "volume"])

        # Convert timestamp to datetime
        df["time"] = pd.to_datetime(df["time"], unit="s")

        # Sort chronologically and reset index
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_cleaned = df[~df['time'].duplicated(keep='first')]

        return df_cleaned


class BitcoinAnalyzer:
    """Class to analyze Bitcoin data and create visualizations."""

    def __init__(self, df_historical: pd.DataFrame, interval_minutes: int = 30):
        """
        Initialize the analyzer with historical data.

        Args:
            df_historical: DataFrame containing historical price data
            interval_minutes: Minutes interval for analysis (default 30)
        """
        self.df_historical = df_historical
        self.interval_minutes = interval_minutes
        self.days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.data_by_cell = {}  # Store time series data for each cell

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for analysis, including filtering by interval and calculating returns.

        Returns:
            Processed DataFrame with returns
        """
        # Filter for desired interval
        filtered_df = self._filter_by_interval()

        # Calculate returns in basis points
        filtered_df['returns'] = filtered_df['close'].pct_change() * 10000  # Convert to basis points

        # Create main dataframe with time features
        df = filtered_df.copy()
        df['timestamp'] = pd.to_datetime(df.index)
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['hour_of_day'] = df['timestamp'].dt.strftime('%Hh%M')
        df.dropna(inplace=True)

        return df

    def _filter_by_interval(self) -> pd.DataFrame:
        """
        Filter historical data to include only specified minute intervals.

        Returns:
            Filtered DataFrame
        """
        if self.df_historical.empty:
            return self.df_historical

        if self.interval_minutes == 30:
            # Keep only 0 and 30 minute marks
            filtered_df = self.df_historical[
                (self.df_historical['time'].dt.minute == 0) |
                (self.df_historical['time'].dt.minute == 30)
                ]
        else:
            # Apply different filtering if needed
            filtered_df = self.df_historical[
                self.df_historical['time'].dt.minute % self.interval_minutes == 0
                ]

        filtered_df['time'] = pd.to_datetime(filtered_df['time'])
        filtered_df.set_index(filtered_df['time'], inplace=True)
        return filtered_df

    def create_visualization(self, df: pd.DataFrame, title_suffix: str = "") -> go.Figure:
        """
        Create a visualization with mean returns heatmap and time series data in hovertext.

        Args:
            df: Processed data frame
            title_suffix: Optional suffix for the title

        Returns:
            Plotly figure object
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No data available for the selected date range",
                height=600
            )
            return fig

        # Create pivot table for mean returns
        pivot_mean = df.pivot_table(
            index='hour_of_day',
            columns='day_of_week',
            values='returns',
            aggfunc='mean'
        )

        # Reorder columns to standard day order
        day_cols = [day for day in self.days_order if day in pivot_mean.columns]
        pivot_mean = pivot_mean[day_cols]

        # Store time series data for each cell
        self.data_by_cell = {}
        hover_texts = []

        for i, time_slot in enumerate(pivot_mean.index):
            hover_row = []
            for j, day in enumerate(pivot_mean.columns):
                # Get all data points for this day and time
                cell_data = df[(df['day_of_week'] == day) & (df['hour_of_day'] == time_slot)]

                # Store data for reference
                cell_key = f"{day}_{time_slot}"
                self.data_by_cell[cell_key] = cell_data

                # Calculate statistics
                sample_count = len(cell_data)
                mean_value = cell_data['returns'].mean() if sample_count > 0 else 0
                std_value = cell_data['returns'].std() if sample_count > 0 else 0

                # Format dates and returns for hover text
                if sample_count > 0:
                    dates_str = "<br>".join([f"{d.strftime('%Y-%m-%d')}: {r:.2f} bp"
                                             for d, r in zip(cell_data.index, cell_data['returns'])])
                    hover_text = f"Day: {day}<br>Time: {time_slot}<br>Mean: {mean_value:.2f} bp<br>Std: {std_value:.2f}<br>n={sample_count}<br><br>Time Series:<br>{dates_str}"
                else:
                    hover_text = f"Day: {day}<br>Time: {time_slot}<br>No data available"

                hover_row.append(hover_text)
            hover_texts.append(hover_row)

        # Create figure
        fig = go.Figure()

        # Add mean returns heatmap with customized hover text
        fig.add_trace(
            go.Heatmap(
                z=pivot_mean.values,
                x=pivot_mean.columns.tolist(),
                y=pivot_mean.index.tolist(),
                colorscale='RdYlGn',
                text=np.round(pivot_mean.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverinfo="text",
                hovertext=hover_texts,
                colorbar=dict(title="Returns<br>(basis points)")
            )
        )

        # Calculate total samples
        total_samples = len(df)

        # Update layout
        fig.update_layout(
            height=900,
            width=200,
            title=f"Bitcoin Returns Analysis by 30-Minute Interval & Day<br>{title_suffix}<br>Total Samples: {total_samples:,}",
            font=dict(family="Arial", size=12),
            hovermode="closest"
        )

        # Update axes
        fig.update_yaxes(
            title="Hour of Day",
            autorange="reversed"
        )

        fig.update_xaxes(
            title="Day of Week"
        )

        return fig


def create_streamlit_app():
    """Create the Streamlit dashboard."""

    st.set_page_config(
        page_title="Bitcoin Returns Heatmap",
        page_icon="📊",
        layout="wide"
    )

    st.title("Bitcoin Returns Heatmap Analysis")
    st.markdown("This dashboard shows a heatmap of Bitcoin returns by day of week and time of day.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Date range selection
    min_date = datetime.date(2024, 1, 1)  # Setting a reasonable minimum date
    max_date = datetime.date.today()

    # Default to Trump's inauguration date if it's within the allowed range
    inauguration_date = datetime.date(2025, 1, 20)
    default_start = inauguration_date if inauguration_date <= max_date else min_date

    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=start_date,
        max_value=max_date
    )

    # Product and interval selection
    product_id = st.sidebar.selectbox(
        "Cryptocurrency Pair",
        options=["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        index=0
    )

    granularity_options = {
        "1 minute": 60,
        "5 minutes": 60 * 5,
        "15 minutes": 60 * 15,
        "1 hour": 60 * 60,
        "6 hours": 60 * 60 * 6,
        "1 day": 60 * 60 * 24
    }

    granularity_selection = st.sidebar.selectbox(
        "Candle Interval",
        options=list(granularity_options.keys()),
        index=2  # Default to 15 minutes
    )

    granularity = granularity_options[granularity_selection]

    interval_options = {
        "30 minutes": 30,
        "1 hour": 60,
        "2 hours": 120,
        "4 hours": 240,
        "6 hours": 360,
        "12 hours": 720,
        "24 hours": 1440
    }

    interval_selection = st.sidebar.selectbox(
        "Analysis Interval",
        options=list(interval_options.keys()),
        index=0  # Default to 30 minutes
    )

    interval_minutes = interval_options[interval_selection]

    if st.sidebar.button("Generate Heatmap"):
        # Convert dates to datetime objects
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)

        # Initialize data fetcher
        fetcher = CoinbaseDataFetcher(product_id, granularity, max_candles=300)

        # Fetch historical data
        df_historical = fetcher.fetch_historical_data(start_datetime, end_datetime)

        if not df_historical.empty:
            # Initialize analyzer
            analyzer = BitcoinAnalyzer(df_historical, interval_minutes=interval_minutes)

            # Process dataset
            processed_data = analyzer.prepare_data()

            # Create visualization
            title_suffix = f"Period: {start_date} to {end_date}"
            fig = analyzer.create_visualization(processed_data, title_suffix)

            # Display the figure
            st.plotly_chart(fig, use_container_width=True)

            # Show data stats
            st.subheader("Data Summary")
            stats_col1, stats_col2 = st.columns(2)

            with stats_col1:
                st.metric("Total Data Points", len(df_historical), None)

                # Format the first date as string instead of using date object
                first_date = "N/A"
                if not df_historical.empty:
                    first_date = df_historical['time'].min().strftime('%Y-%m-%d')
                st.write(f"**First Date:** {first_date}")

            with stats_col2:
                st.metric("Analysis Intervals", len(processed_data) if not processed_data.empty else 0, None)

                # Format the last date as string instead of using date object
                last_date = "N/A"
                if not df_historical.empty:
                    last_date = df_historical['time'].max().strftime('%Y-%m-%d')
                st.write(f"**Last Date:** {last_date}")

            # Show raw data (expandable)
            with st.expander("View Raw Data"):
                st.dataframe(df_historical)

        else:
            st.error("No data available for the selected time period. Please try a different date range.")

    else:
        # Initial instructions
        st.info("Select your parameters in the sidebar and click 'Generate Heatmap' to view the analysis.")


if __name__ == "__main__":
    create_streamlit_app()
