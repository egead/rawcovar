import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_earthquake_catalog(filepath='earthquake_events.csv'):
    """
    Load the earthquake catalog CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded earthquake catalog
    """
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    return df

def load_station_arrivals(filepath='station_arrivals.csv'):
    """
    Load the station arrivals CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded station arrivals
    """
    df = pd.read_csv(filepath, parse_dates=['arrival_time', 'datetime'])
    return df

def analyze_event_frequency(df, min_magnitude=2.0, period='M'):
    """
    Analyze event frequency by time periods and magnitude.
    
    Args:
        df (pandas.DataFrame): Earthquake catalog DataFrame
        min_magnitude (float): Minimum magnitude to consider
        period (str): Pandas period frequency code 
                      ('M' for month, 'W' for week, 'Q' for quarter)
    
    Returns:
        tuple: 
        - DataFrame of periods with event counts and magnitude statistics
        - DataFrame of top periods sorted by event count
    """
    df_filtered = df[df['magnitude'] >= min_magnitude]
    
    # Create and group by stats
    df_filtered['period_start'] = df_filtered['datetime'].dt.to_period(period).apply(lambda r: r.start_time)
    
    # Modified aggregation to work without event_id
    period_stats = df_filtered.groupby('period_start').agg({
        'datetime': 'count',  # Count events by datetime instead of event_id
        'magnitude': ['mean', 'max']  
    }).reset_index()
    
    period_stats.columns = ['period_start', 'event_count', 'mean_magnitude', 'max_magnitude']
    
    top_periods = period_stats.sort_values('event_count', ascending=False)
    
    return period_stats, top_periods

def analyze_station_arrivals_in_period(station_df, event_df, start_date, end_date):
    """
    Analyze station arrivals for a specific time period.
    Modified to work without event_id, using datetime to link events and arrivals.
    
    Args:
        station_df (pandas.DataFrame): Station arrivals DataFrame
        event_df (pandas.DataFrame): Earthquake events DataFrame
        start_date (pd.Timestamp): Start of the time period
        end_date (pd.Timestamp): End of the time period
    
    Returns:
        dict: Station arrival analysis for the specified period
    """
    # Filter events and stations in the chosen time period
    period_events = event_df[
        (event_df['datetime'] >= start_date) & 
        (event_df['datetime'] <= end_date)
    ]
    
    period_station_arrivals = station_df[
        (station_df['arrival_time'] >= start_date) & 
        (station_df['arrival_time'] <= end_date)
    ]
    
    station_occurrence = period_station_arrivals['station_name'].value_counts()
    
    # Merge station arrivals with events using datetime instead of event_id
    # This assumes the datetime in station_df matches exactly with datetime in event_df
    merged_data = pd.merge(period_station_arrivals, period_events, on='datetime', how='left')
    
    # Station coverage by magnitude in this period
    station_magnitude_coverage = merged_data.groupby('station_name')['magnitude'].agg(['count','min','max','mean'])
    
    return {
        'station_occurrence': station_occurrence,
        'station_magnitude_coverage': station_magnitude_coverage
    }

def plot_events_on_map(df, title='Events in the Catalogue'):
    """
    Create a cartographic plot of earthquake events.
    
    Args:
        df (pandas.DataFrame): DataFrame containing earthquake events
        title (str): Title of the plot
    """
    fig = plt.figure(figsize=(12, 10))

    # Extract coordinates
    lats = df['latitude'].tolist()
    lons = df['longitude'].tolist()

    # Create local projection
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    projection = ccrs.NearsidePerspective(
        central_longitude=center_lon,
        central_latitude=center_lat,
        satellite_height=3000000.0  # Reduce for more zoom
    )

    # Create map axes with the projection
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    depths = df['depth'].tolist()
    marker_sizes = [max(10, d * 2 + 10) for d in depths]  # Deeper = larger marker

    scatter = ax.scatter(
        lons, lats,
        transform=ccrs.PlateCarree(),  # input: lat/lon
        c=df['magnitude'].tolist(),
        s=marker_sizes,
        cmap='hot_r',  # white: low, red/yellow: high
        alpha=0.8,
        zorder=5,
        edgecolor='k',
        linewidth=0.3
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.05)
    cbar.set_label('Magnitude')

    # Set map extent with padding
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    padding = max(lat_range, lon_range) * 0.2  # Reduce padding for zoom
    ax.set_extent(
        [min(lons) - padding, max(lons) + padding, 
         min(lats) - padding, max(lats) + padding],
        crs=ccrs.PlateCarree()
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False  

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

import pandas as pd

def filtered_cat_summary(event_df, station_df, min_mag=1.0, prd="D"):
    """
    Generate a summary of earthquake catalog filtered by magnitude and analyze the most active period.
    
    Parameters:
    -----------
    event_df : DataFrame
        DataFrame containing earthquake events with datetime and magnitude columns
    station_df : DataFrame
        DataFrame containing station arrival information with arrival_time column
    min_mag : float
        Minimum magnitude threshold for events
    prd : str
        Time period for grouping ('D' for day, 'W' for week, 'M' for month)
    """
    # Analyze event frequency
    period_stats, top_periods = analyze_event_frequency(event_df, min_magnitude=min_mag, period=prd)
    
    print(f"Event Frequency Analysis (Magnitude >= {min_mag}, {prd} Period):")
    print("\nTop 10 Most Active Periods:")
    print(top_periods.head(10))
    
    # Get the top period
    top_period = top_periods.iloc[0]
    print("\nTop Active Period:")
    print(top_period)
    
    # Define the period range based on the period parameter
    top_period_start = top_period['period_start']
    
    # Set the end date based on the period parameter
    if prd == "D":
        # For daily, use 1 day
        top_period_end = top_period_start + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)
    elif prd == "W":
        # For weekly, use 1 week
        top_period_end = top_period_start + pd.DateOffset(weeks=1) - pd.Timedelta(seconds=1)
    elif prd == "M":
        # For monthly, use 1 month
        top_period_end = top_period_start + pd.DateOffset(months=1) - pd.Timedelta(seconds=1)
    else:
        # Default fallback to the original 2 weeks
        top_period_end = top_period_start + pd.DateOffset(weeks=2) - pd.Timedelta(days=1)
    
    # Filter events within the top period
    top_period_events = event_df[
        (event_df['datetime'] >= top_period_start) &
        (event_df['datetime'] <= top_period_end)
    ]
    
    # Filter station data to only include arrivals within the top period timeframe
    # Make sure arrival_time is datetime
    if not pd.api.types.is_datetime64_dtype(station_df['arrival_time']):
        station_df = station_df.copy()
        station_df['arrival_time'] = pd.to_datetime(station_df['arrival_time'])
        
    top_period_station_df = station_df[
        (station_df['arrival_time'] >= top_period_start) &
        (station_df['arrival_time'] <= top_period_end)
    ]
    
    # Plot events on map (using the existing function without changes)
    plot_events_on_map(top_period_events,
                    title=f'Events from {top_period_start.date()} to {top_period_end.date()}')
    
    print("\nStation Arrival Analysis for Top Period:")
    
    # Analyze station data directly for the period
    station_occurrence = top_period_station_df['station_name'].value_counts().reset_index()
    station_occurrence.columns = ['station_name', 'arrival_count']
    
    # For magnitude coverage, we need to match station arrivals with events
    # Since we don't have event_id, we'll match based on closest time
    
    # Create a function to find the closest event for each station arrival
    def find_closest_event(arrival_time, events_df):
        time_diffs = abs(events_df['datetime'] - arrival_time)
        closest_idx = time_diffs.idxmin()
        closest_event = events_df.loc[closest_idx]
        return closest_event['magnitude']
    
    # Apply the function to each station arrival (this might be slow for large datasets)
    if len(top_period_events) > 0 and len(top_period_station_df) > 0:
        top_period_station_df = top_period_station_df.copy()
        top_period_station_df['magnitude'] = top_period_station_df['arrival_time'].apply(
            lambda x: find_closest_event(x, top_period_events)
        )
        
        # Get magnitude statistics for each station
        station_mag_stats = top_period_station_df.groupby('station_name')['magnitude'].agg([
            'count', 'min', 'max', 'mean'
        ]).reset_index()
        
        # Sort by count
        station_mag_stats = station_mag_stats.sort_values('count', ascending=False)
    else:
        station_mag_stats = pd.DataFrame(columns=['station_name', 'count', 'min', 'max', 'mean'])
    
    print("\nTop 10 Stations by Number of Arrivals:")
    print(station_occurrence.head(10))
    
    print("\nTop 10 Stations Magnitude Coverage:")
    print(station_mag_stats.head(10))
""" def filtered_cat_summary(event_df, station_df, min_mag=1.0, prd="D"):

 period_stats, top_periods = analyze_event_frequency(event_df, min_magnitude=min_mag, period=prd)
 print(f"Event Frequency Analysis (Magnitude >= {min_mag}, {prd} Period):")
 print("\nTop 10 Most Active Periods:")
 print(top_periods.head(10))

 top_period = top_periods.iloc[0]
 print("\nTop Active Period:")
 print(top_period)

 top_period_start = top_period['period_start']
 top_period_end = top_period_start + pd.DateOffset(weeks=2) - pd.Timedelta(days=1) # work on this part, its incomplete as is.

 # Filter events within the top period
 top_period_events = event_df[
    (event_df['datetime'] >= top_period_start) & 
    (event_df['datetime'] <= top_period_end)
 ]

 # Plot events on map
 plot_events_on_map(top_period_events, 
                 title=f'Events from {top_period_start.date()} to {top_period_end.date()}')

 print("\nStation Arrival Analysis for Top Period:")
 station_analysis = analyze_station_arrivals_in_period(station_df, event_df,start_date=top_period_start,end_date=top_period_end)

 print("\nTop 10 Stations by Number of Arrivals:")
 print(station_analysis['station_occurrence'].head(10))

 print("\nTop 10 Stations Magnitude Coverage:")
 print(station_analysis['station_magnitude_coverage'].head(10))

 """