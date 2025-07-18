import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Configure altair
alt.data_transformers.enable('json')

# Configure page
st.set_page_config(
    page_title="College Basketball Analytics Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the college basketball dataset"""
    try:
        # Load the comprehensive dataset
        df = pd.read_csv('archive/cbb.csv')
        
        # Load individual year files and combine
        year_files = [f'archive/cbb{year}.csv' for year in range(13, 26)]
        yearly_dfs = []
        
        for year, file in enumerate(year_files, start=2013):
            try:
                temp_df = pd.read_csv(file)
                if 'YEAR' not in temp_df.columns:
                    temp_df['YEAR'] = year
                yearly_dfs.append(temp_df)
            except FileNotFoundError:
                continue
        
        if yearly_dfs:
            yearly_combined = pd.concat(yearly_dfs, ignore_index=True)
            # Combine with main dataset if different structure
            if set(df.columns) != set(yearly_combined.columns):
                # Align columns
                common_cols = list(set(df.columns) & set(yearly_combined.columns))
                df = df[common_cols]
                yearly_combined = yearly_combined[common_cols]
            
            df = pd.concat([df, yearly_combined], ignore_index=True).drop_duplicates()
        
        # Clean and prepare data
        df['WIN_PCT'] = df['W'] / df['G']
        df['EFFICIENCY_DIFF'] = df['ADJOE'] - df['ADJDE']
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    # Title and header
    st.title("🏀 College Basketball Analytics Dashboard")
    st.markdown("### Explore team performance, efficiency metrics, and tournament success patterns")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("Could not load data. Please check file paths.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎯 Dashboard Controls")
    
    # Year selection
    years = sorted(df['YEAR'].unique()) if 'YEAR' in df.columns else [2024]
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=years,
        default=years[-3:] if len(years) >= 3 else years,
        help="Choose which years to include in analysis"
    )
    
    # Conference selection
    conferences = sorted(df['CONF'].unique()) if 'CONF' in df.columns else ['All']
    selected_conferences = st.sidebar.multiselect(
        "Select Conferences",
        options=conferences,
        default=conferences[:5] if len(conferences) > 5 else conferences,
        help="Filter by conference"
    )
    
    # Team search
    team_search = st.sidebar.text_input(
        "Search Teams",
        placeholder="Enter team name...",
        help="Search for specific teams"
    )
    
    # Performance metric selector
    metric_options = {
        'Offensive Efficiency': 'ADJOE',
        'Defensive Efficiency': 'ADJDE', 
        'Win Percentage': 'WIN_PCT',
        'Efficiency Differential': 'EFFICIENCY_DIFF'
    }
    
    selected_metric = st.sidebar.selectbox(
        "Primary Metric",
        options=list(metric_options.keys()),
        help="Choose the main metric for analysis"
    )
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_years and 'YEAR' in df.columns:
        filtered_df = filtered_df[filtered_df['YEAR'].isin(selected_years)]
    
    if selected_conferences:
        filtered_df = filtered_df[filtered_df['CONF'].isin(selected_conferences)]
    
    if team_search:
        team_col = 'TEAM' if 'TEAM' in filtered_df.columns else 'Team'
        filtered_df = filtered_df[filtered_df[team_col].str.contains(team_search, case=False, na=False)]
    
    # Main dashboard layout
    if not filtered_df.empty:
        
        # Key metrics row
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_wins = filtered_df['W'].mean()
            st.metric("Average Wins", f"{avg_wins:.1f}")
        
        with col2:
            avg_efficiency = filtered_df['ADJOE'].mean() if 'ADJOE' in filtered_df.columns else 0
            st.metric("Avg Offensive Efficiency", f"{avg_efficiency:.1f}")
        
        with col3:
            avg_def_efficiency = filtered_df['ADJDE'].mean() if 'ADJDE' in filtered_df.columns else 0
            st.metric("Avg Defensive Efficiency", f"{avg_def_efficiency:.1f}")
        
        with col4:
            tournament_teams = len(filtered_df[filtered_df['POSTSEASON'].notna()]) if 'POSTSEASON' in filtered_df.columns else 0
            st.metric("Tournament Teams", tournament_teams)
        
        # Main visualization section
        st.subheader("🎯 Interactive Performance Analysis")
        
        # Create two columns for coordinated visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Offensive vs Defensive Efficiency")
            
            # Scatter plot with selection capability
            team_col = 'TEAM' if 'TEAM' in filtered_df.columns else 'Team'
            
            # Calculate quadrant lines
            x_median = filtered_df['ADJDE'].median()
            y_median = filtered_df['ADJOE'].median()
            
            # Create base scatter plot
            click_selection = alt.selection_point()
            scatter = alt.Chart(filtered_df).mark_circle(size=100).add_params(
                click_selection
            ).encode(
                x=alt.X('ADJDE:Q', title='Defensive Efficiency (lower is better)'),
                y=alt.Y('ADJOE:Q', title='Offensive Efficiency'),
                size=alt.Size('W:Q', scale=alt.Scale(range=[50, 400]), title='Wins'),
                color=alt.Color('CONF:N', title='Conference'),
                tooltip=[team_col + ':N', 'W:Q', 'G:Q', 'WIN_PCT:Q', 'CONF:N', 'ADJOE:Q', 'ADJDE:Q']
            ).properties(
                width=450,
                height=350,
                title="Team Efficiency Comparison"
            )
            
            # Add quadrant lines
            v_line = alt.Chart(pd.DataFrame({'x': [x_median]})).mark_rule(
                strokeDash=[5, 5], color='gray', opacity=0.5
            ).encode(x='x:Q')
            
            h_line = alt.Chart(pd.DataFrame({'y': [y_median]})).mark_rule(
                strokeDash=[5, 5], color='gray', opacity=0.5
            ).encode(y='y:Q')
            
            # Combine charts
            chart1 = scatter + v_line + h_line
            
            st.altair_chart(chart1, use_container_width=True)
        
        with col2:
            st.markdown("#### Tournament Success by Efficiency")
            
            # Tournament success visualization
            if 'POSTSEASON' in filtered_df.columns:
                tournament_df = filtered_df[filtered_df['POSTSEASON'].notna()].copy()
                
                if not tournament_df.empty:
                    y_field = 'SEED' if 'SEED' in tournament_df.columns else 'W'
                    y_title = 'Tournament Seed (lower is better)' if 'SEED' in tournament_df.columns else 'Wins'
                    
                    chart2 = alt.Chart(tournament_df).mark_circle(size=100).encode(
                        x=alt.X('EFFICIENCY_DIFF:Q', title='Efficiency Differential'),
                        y=alt.Y(f'{y_field}:Q', title=y_title, sort='descending' if y_field == 'SEED' else 'ascending'),
                        color=alt.Color('POSTSEASON:N', title='Tournament Result'),
                        size=alt.Size('W:Q', scale=alt.Scale(range=[50, 300]), title='Wins'),
                        tooltip=[team_col + ':N', 'ADJOE:Q', 'ADJDE:Q', 'W:Q', 'CONF:N', 'POSTSEASON:N']
                    ).properties(
                        width=450,
                        height=350,
                        title="Tournament Performance vs Efficiency"
                    )
                    
                    st.altair_chart(chart2, use_container_width=True)
                else:
                    st.info("No tournament data available for selected filters")
            else:
                # Alternative visualization - Win percentage distribution
                chart2 = alt.Chart(filtered_df).mark_bar().encode(
                    x=alt.X('WIN_PCT:Q', bin=alt.Bin(maxbins=20), title='Win Percentage'),
                    y=alt.Y('count():Q', title='Number of Teams'),
                    color=alt.Color('CONF:N', title='Conference'),
                    tooltip=['count():Q', 'CONF:N']
                ).properties(
                    width=450,
                    height=350,
                    title="Win Percentage Distribution"
                )
                st.altair_chart(chart2, use_container_width=True)
        
        # Additional coordinated visualizations
        st.subheader("🔍 Detailed Team Analysis")
        
        # Performance trends (if multiple years selected)
        if len(selected_years) > 1 and 'YEAR' in filtered_df.columns:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Conference Performance Overview")
                
                # Conference average performance with distribution
                conf_stats = filtered_df.groupby('CONF').agg({
                    'ADJOE': 'mean',
                    'ADJDE': 'mean',
                    'WIN_PCT': 'mean',
                    'W': 'mean',
                    'EFFICIENCY_DIFF': 'mean'
                }).reset_index()
                
                # Sort by selected metric
                conf_stats = conf_stats.sort_values(metric_options[selected_metric], ascending=False)
                
                chart3 = alt.Chart(conf_stats).mark_bar().encode(
                    x=alt.X('CONF:N', title='Conference', sort='-y'),
                    y=alt.Y(f'{metric_options[selected_metric]}:Q', title=f'Average {selected_metric}'),
                    color=alt.Color(f'{metric_options[selected_metric]}:Q', 
                                  scale=alt.Scale(scheme='viridis'), 
                                  title=f'Avg {selected_metric}'),
                    tooltip=['CONF:N', f'{metric_options[selected_metric]}:Q', 'WIN_PCT:Q', 'W:Q']
                ).properties(
                    width=450,
                    height=350,
                    title=f"Conference Rankings by {selected_metric}"
                )
                
                st.altair_chart(chart3, use_container_width=True)
            
            with col2:
                st.markdown("#### Elite Teams Leaderboard")
                
                # Get top 15 teams overall by selected metric
                top_teams = filtered_df.nlargest(15, metric_options[selected_metric]).copy()
                top_teams['Rank'] = range(1, len(top_teams) + 1)
                
                chart4 = alt.Chart(top_teams).mark_circle(size=150).encode(
                    x=alt.X('ADJOE:Q', title='Offensive Efficiency'),
                    y=alt.Y('ADJDE:Q', title='Defensive Efficiency (lower is better)', sort='descending'),
                    color=alt.Color('CONF:N', title='Conference'),
                    size=alt.Size('W:Q', scale=alt.Scale(range=[100, 400]), title='Wins'),
                    tooltip=[f'{team_col}:N', 'Rank:O', 'CONF:N', 'W:Q', 'WIN_PCT:Q', 
                            'ADJOE:Q', 'ADJDE:Q', f'{metric_options[selected_metric]}:Q']
                ).properties(
                    width=450,
                    height=350,
                    title=f"Top 15 Teams by {selected_metric}"
                )
                
                st.altair_chart(chart4, use_container_width=True)
                
                # Show ranking table
                st.markdown("**Top 10 Rankings:**")
                ranking_display = top_teams.head(10)[[team_col, 'CONF', 'W', 'WIN_PCT', metric_options[selected_metric]]].copy()
                ranking_display.columns = ['Team', 'Conference', 'Wins', 'Win %', selected_metric]
                ranking_display['Rank'] = range(1, len(ranking_display) + 1)
                ranking_display = ranking_display[['Rank', 'Team', 'Conference', 'Wins', 'Win %', selected_metric]]
                
                st.dataframe(ranking_display, hide_index=True, use_container_width=True)
        
        # Team comparison section
        st.subheader("⚖️ Team Comparison Tool")
        
        # Team selector for comparison
        available_teams = sorted(filtered_df[team_col].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_teams_compare = st.multiselect(
                "Select Teams to Compare",
                options=available_teams,
                default=available_teams[:3] if len(available_teams) >= 3 else available_teams,
                help="Choose teams for detailed comparison"
            )
        
        with col2:
            comparison_metrics = st.multiselect(
                "Select Metrics for Comparison",
                options=['ADJOE', 'ADJDE', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB'],
                default=['ADJOE', 'ADJDE', 'EFG_O', 'EFG_D'],
                help="Choose which metrics to compare"
            )
        
        if selected_teams_compare and comparison_metrics:
            comparison_df = filtered_df[filtered_df[team_col].isin(selected_teams_compare)]
            
            # Create normalized comparison data for better visualization
            comparison_data = []
            for team in selected_teams_compare:
                team_data = comparison_df[comparison_df[team_col] == team]
                if not team_data.empty:
                    for metric in comparison_metrics:
                        comparison_data.append({
                            'Team': team,
                            'Metric': metric,
                            'Value': team_data[metric].iloc[0],
                            'Normalized_Value': (team_data[metric].iloc[0] - comparison_df[metric].min()) / 
                                              (comparison_df[metric].max() - comparison_df[metric].min())
                        })
            
            comparison_chart_df = pd.DataFrame(comparison_data)
            
            # Create grouped bar chart as alternative to radar chart
            chart5 = alt.Chart(comparison_chart_df).mark_bar().encode(
                x=alt.X('Metric:N', title='Performance Metrics'),
                y=alt.Y('Value:Q', title='Metric Value'),
                color=alt.Color('Team:N', title='Team'),
                column=alt.Column('Team:N', title='Team Comparison'),
                tooltip=['Team:N', 'Metric:N', 'Value:Q']
            ).properties(
                width=120,
                height=300,
                title="Team Performance Comparison"
            ).resolve_scale(
                y='independent'
            )
            
            st.altair_chart(chart5, use_container_width=True)
        
        # Data table with sorting and filtering
        st.subheader("📋 Detailed Team Statistics")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_all_columns = st.checkbox("Show All Columns", value=False)
        
        with col2:
            sort_column = st.selectbox(
                "Sort by Column",
                options=filtered_df.columns.tolist(),
                index=list(filtered_df.columns).index('W') if 'W' in filtered_df.columns else 0
            )
        
        with col3:
            sort_order = st.selectbox("Sort Order", options=['Descending', 'Ascending'])
        
        # Prepare display dataframe
        display_df = filtered_df.copy()
        
        if not show_all_columns:
            key_columns = [team_col, 'CONF', 'W', 'G', 'WIN_PCT', 'ADJOE', 'ADJDE', 'EFFICIENCY_DIFF']
            if 'YEAR' in display_df.columns:
                key_columns.insert(-1, 'YEAR')
            display_df = display_df[[col for col in key_columns if col in display_df.columns]]
        
        # Sort dataframe
        ascending = sort_order == 'Ascending'
        display_df = display_df.sort_values(by=sort_column, ascending=ascending)
        
        # Display with highlighting
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"college_basketball_filtered_{len(selected_years)}years.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No data available for the selected filters. Please adjust your selections.")
    


if __name__ == "__main__":
    main() 