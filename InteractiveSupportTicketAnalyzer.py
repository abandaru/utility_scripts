import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from plotly.colors import qualitative
import plotly.io as pio

warnings.filterwarnings('ignore')

class InteractiveSupportTicketAnalyzer:
    def __init__(self, excel_file_path):
        """
        Initialize the analyzer with interactive visualizations using Plotly
        """
        self.excel_file_path = excel_file_path
        self.df = None
        self.sheet_name = 'product_issues'
        
        # Set up Plotly theme
        pio.templates.default = "plotly_white"
        
    def read_excel_data(self):
        """
        Read the Excel file and load the data
        """
        try:
            # Read the Excel file
            self.df = pd.read_excel(self.excel_file_path, sheet_name=self.sheet_name)
            print(f"✅ Successfully loaded {len(self.df)} records from {self.sheet_name} sheet")
            
            # Display basic info about the dataset
            print("\n📊 Dataset Overview:")
            print(f"Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Clean column names (remove extra spaces, standardize)
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#', '')
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File '{self.excel_file_path}' not found")
            return False
        except Exception as e:
            print(f"❌ Error reading Excel file: {str(e)}")
            return False
    
    def data_cleaning_and_preparation(self):
        """
        Clean and prepare the data for analysis including trend analysis
        """
        print("\n🧹 Cleaning and preparing data...")
        
        # Handle missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  - {col}: {count} missing values")
        
        # Fill missing values with appropriate defaults
        categorical_columns = ['type', 'category', 'source', 'status', 'priority', 't-shirt_size']
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
        
        # Convert ticket# to string to handle mixed types
        if 'ticket' in self.df.columns:
            self.df['ticket'] = self.df['ticket'].astype(str)
        
        # Convert date columns
        date_columns = ['date_reported', 'actual_closed_date']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Calculate resolution time for closed tickets
        if 'date_reported' in self.df.columns and 'actual_closed_date' in self.df.columns:
            self.df['resolution_time_days'] = (self.df['actual_closed_date'] - self.df['date_reported']).dt.days
            self.df['resolution_time_days'] = self.df['resolution_time_days'].where(self.df['resolution_time_days'] >= 0)
        
        # Calculate aging for open tickets
        current_date = datetime.now()
        if 'date_reported' in self.df.columns:
            self.df['aging_days'] = (current_date - self.df['date_reported']).dt.days
        
        # Add derived columns for better analysis
        if 'date_reported' in self.df.columns:
            self.df['year_month'] = self.df['date_reported'].dt.to_period('M').astype(str)
            self.df['day_of_week'] = self.df['date_reported'].dt.day_name()
            self.df['month_name'] = self.df['date_reported'].dt.month_name()
        
        print("✅ Data cleaning completed")
    
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics including trend metrics
        """
        print("\n📈 Generating Summary Statistics...")
        
        summary_report = {
            'total_tickets': len(self.df),
            'unique_tickets': self.df['ticket'].nunique() if 'ticket' in self.df.columns else len(self.df),
            'date_range': {
                'start': self.df['date_reported'].min() if 'date_reported' in self.df.columns else None,
                'end': self.df['date_reported'].max() if 'date_reported' in self.df.columns else None
            }
        }
        
        # Calculate trend metrics with error handling
        try:
            if 'resolution_time_days' in self.df.columns:
                closed_tickets = self.df[self.df['resolution_time_days'].notna()]
                if not closed_tickets.empty:
                    avg_resolution = closed_tickets['resolution_time_days'].mean()
                    median_resolution = closed_tickets['resolution_time_days'].median()
                    
                    # Ensure values are float and handle NaN
                    avg_resolution = float(avg_resolution) if not pd.isna(avg_resolution) else 0.0
                    median_resolution = float(median_resolution) if not pd.isna(median_resolution) else 0.0
                    
                    resolution_by_priority = closed_tickets.groupby('priority')['resolution_time_days'].mean()
                    resolution_by_type = closed_tickets.groupby('type')['resolution_time_days'].mean()
                    
                    # Convert to regular dict with float values
                    resolution_by_priority_dict = {}
                    for priority, value in resolution_by_priority.items():
                        resolution_by_priority_dict[priority] = float(value) if not pd.isna(value) else 0.0
                    
                    resolution_by_type_dict = {}
                    for ticket_type, value in resolution_by_type.items():
                        resolution_by_type_dict[ticket_type] = float(value) if not pd.isna(value) else 0.0
                    
                    summary_report['resolution_metrics'] = {
                        'avg_resolution_time': avg_resolution,
                        'median_resolution_time': median_resolution,
                        'resolution_by_priority': resolution_by_priority_dict,
                        'resolution_by_type': resolution_by_type_dict
                    }
                else:
                    print("Warning: No closed tickets found for resolution analysis")
        except Exception as e:
            print(f"Warning: Could not calculate resolution metrics: {e}")
        
        # Calculate aging metrics for open tickets with error handling
        try:
            if 'aging_days' in self.df.columns:
                open_tickets = self.df[self.df['status'].isin(['Open', 'In Progress', 'Pending'])]
                if not open_tickets.empty:
                    avg_aging = open_tickets['aging_days'].mean()
                    max_aging = open_tickets['aging_days'].max()
                    tickets_over_30 = (open_tickets['aging_days'] > 30).sum()
                    tickets_over_90 = (open_tickets['aging_days'] > 90).sum()
                    
                    # Ensure values are proper types and handle NaN
                    avg_aging = float(avg_aging) if not pd.isna(avg_aging) else 0.0
                    max_aging = float(max_aging) if not pd.isna(max_aging) else 0.0
                    tickets_over_30 = int(tickets_over_30)
                    tickets_over_90 = int(tickets_over_90)
                    
                    summary_report['aging_metrics'] = {
                        'avg_aging_days': avg_aging,
                        'max_aging_days': max_aging,
                        'tickets_over_30_days': tickets_over_30,
                        'tickets_over_90_days': tickets_over_90
                    }
                else:
                    print("Warning: No open tickets found for aging analysis")
        except Exception as e:
            print(f"Warning: Could not calculate aging metrics: {e}")
        
        # Analyze each categorical column
        categorical_columns = ['type', 'category', 'source', 'status', 'priority', 't-shirt_size']
        
        for col in categorical_columns:
            try:
                if col in self.df.columns:
                    value_counts = self.df[col].value_counts()
                    # Convert to regular dict with int values
                    summary_report[f'{col}_distribution'] = {k: int(v) for k, v in value_counts.items()}
            except Exception as e:
                print(f"Warning: Could not calculate distribution for {col}: {e}")
        
        return summary_report
    
    def create_interactive_overview_dashboard(self):
        """
        Create an interactive overview dashboard with key metrics
        """
        print("\n🎨 Creating interactive overview dashboard...")
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Status Distribution', 'Priority vs Resolution Time', 
                          'Category Distribution', 'Monthly Trend'),
            specs=[[{"type": "pie"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Status Distribution (Pie Chart)
        if 'status' in self.df.columns:
            status_counts = self.df['status'].value_counts()
            fig.add_trace(go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name="Status",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                textinfo='label+percent'
            ), row=1, col=1)
        
        # 2. Priority vs Resolution Time (Box Plot)
        if 'priority' in self.df.columns and 'resolution_time_days' in self.df.columns:
            for i, priority in enumerate(['Low', 'Medium', 'High', 'Critical']):
                if priority in self.df['priority'].values:
                    data = self.df[self.df['priority'] == priority]['resolution_time_days'].dropna()
                    if not data.empty:
                        fig.add_trace(go.Box(
                            y=data,
                            name=priority,
                            boxmean=True,
                            hovertemplate=f"<b>{priority} Priority</b><br>Resolution Time: %{{y}} days<extra></extra>"
                        ), row=1, col=2)
        
        # 3. Category Distribution (Bar Chart)
        if 'category' in self.df.columns:
            category_counts = self.df['category'].value_counts()
            fig.add_trace(go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                name="Categories",
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                marker_color=px.colors.qualitative.Set3
            ), row=2, col=1)
        
        # 4. Monthly Trend (Line Chart)
        if 'year_month' in self.df.columns:
            monthly_counts = self.df['year_month'].value_counts().sort_index()
            fig.add_trace(go.Scatter(
                x=monthly_counts.index,
                y=monthly_counts.values,
                mode='lines+markers',
                name="Monthly Tickets",
                line=dict(width=3),
                hovertemplate="<b>%{x}</b><br>Tickets: %{y}<extra></extra>"
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Support Tickets Interactive Overview Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # Save and show
        fig.write_html("interactive_overview_dashboard.html")
        fig.show()
        print("📊 Interactive overview dashboard saved as 'interactive_overview_dashboard.html'")
    
    def create_trend_analysis_dashboard(self):
        """
        Create comprehensive trend analysis with interactive charts
        """
        print("\n📈 Creating interactive trend analysis dashboard...")
        
        # Create a multi-tab dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Monthly Creation Trend', 'Day of Week Pattern',
                          'Resolution Time by Priority', 'Aging Distribution',
                          'Type vs Priority Heatmap', 'Source Channel Analysis'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "violin"}, {"type": "histogram"}],
                   [{"type": "heatmap"}, {"type": "pie"}]]
        )
        
        # 1. Monthly Creation Trend with Resolution Rate
        if 'year_month' in self.df.columns:
            monthly_created = self.df.groupby('year_month').size()
            if 'actual_closed_date' in self.df.columns:
                self.df['closed_year_month'] = self.df['actual_closed_date'].dt.to_period('M').astype(str)
                monthly_resolved = self.df[self.df['closed_year_month'].notna()].groupby('closed_year_month').size()
                
                fig.add_trace(go.Scatter(
                    x=monthly_created.index,
                    y=monthly_created.values,
                    mode='lines+markers',
                    name='Created',
                    line=dict(color='blue', width=3),
                    hovertemplate="<b>%{x}</b><br>Created: %{y}<extra></extra>"
                ), row=1, col=1)
                
                if not monthly_resolved.empty:
                    fig.add_trace(go.Scatter(
                        x=monthly_resolved.index,
                        y=monthly_resolved.values,
                        mode='lines+markers',
                        name='Resolved',
                        line=dict(color='green', width=3),
                        hovertemplate="<b>%{x}</b><br>Resolved: %{y}<extra></extra>"
                    ), row=1, col=1)
        
        # 2. Day of Week Pattern
        if 'day_of_week' in self.df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = self.df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
            
            colors = ['darkblue' if day in ['Saturday', 'Sunday'] else 'lightblue' for day in day_order]
            fig.add_trace(go.Bar(
                x=day_counts.index,
                y=day_counts.values,
                name="Day Pattern",
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Tickets: %{y}<extra></extra>"
            ), row=1, col=2)
        
        # 3. Resolution Time Distribution by Priority (Violin Plot)
        if 'priority' in self.df.columns and 'resolution_time_days' in self.df.columns:
            for priority in self.df['priority'].unique():
                if priority and priority != 'Unknown':
                    data = self.df[self.df['priority'] == priority]['resolution_time_days'].dropna()
                    if not data.empty:
                        fig.add_trace(go.Violin(
                            y=data,
                            name=f'{priority}',
                            box_visible=True,
                            meanline_visible=True,
                            hovertemplate=f"<b>{priority}</b><br>Days: %{{y}}<extra></extra>"
                        ), row=2, col=1)
        
        # 4. Aging Distribution (Histogram)
        if 'aging_days' in self.df.columns:
            open_tickets = self.df[self.df['status'].isin(['Open', 'In Progress', 'Pending'])]
            if not open_tickets.empty:
                fig.add_trace(go.Histogram(
                    x=open_tickets['aging_days'],
                    nbinsx=20,
                    name="Aging Distribution",
                    hovertemplate="<b>Age Range</b><br>Days: %{x}<br>Count: %{y}<extra></extra>",
                    marker_color='orange',
                    opacity=0.7
                ), row=2, col=2)
        
        # 5. Type vs Priority Heatmap
        if 'type' in self.df.columns and 'priority' in self.df.columns:
            heatmap_data = pd.crosstab(self.df['type'], self.df['priority'])
            fig.add_trace(go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Blues',
                hovertemplate="<b>%{y} - %{x}</b><br>Count: %{z}<extra></extra>"
            ), row=3, col=1)
        
        # 6. Source Channel Analysis
        if 'source' in self.df.columns:
            source_counts = self.df['source'].value_counts()
            fig.add_trace(go.Pie(
                labels=source_counts.index,
                values=source_counts.values,
                name="Sources",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
            ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Interactive Trend Analysis Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1200,
            showlegend=True
        )
        
        # Save and show
        fig.write_html("interactive_trend_dashboard.html")
        fig.show()
        print("📈 Interactive trend dashboard saved as 'interactive_trend_dashboard.html'")
    
    def create_performance_analytics_dashboard(self):
        """
        Create performance analytics with advanced interactive features
        """
        print("\n⚡ Creating performance analytics dashboard...")
        
        # Create advanced dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Resolution Time vs T-shirt Size', 'Priority Distribution Over Time',
                          'Category Performance Matrix', 'SLA Compliance Analysis'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # 1. Resolution Time vs T-shirt Size (Bubble Chart)
        if all(col in self.df.columns for col in ['t-shirt_size', 'resolution_time_days', 'priority']):
            closed_tickets = self.df[self.df['resolution_time_days'].notna()]
            if not closed_tickets.empty:
                size_resolution = closed_tickets.groupby(['t-shirt_size', 'priority']).agg({
                    'resolution_time_days': ['mean', 'count']
                }).round(1)
                size_resolution.columns = ['avg_resolution', 'count']
                size_resolution = size_resolution.reset_index()
                
                colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red'}
                
                for priority in size_resolution['priority'].unique():
                    priority_data = size_resolution[size_resolution['priority'] == priority]
                    fig.add_trace(go.Scatter(
                        x=priority_data['t-shirt_size'],
                        y=priority_data['avg_resolution'],
                        mode='markers',
                        marker=dict(
                            size=priority_data['count'] * 2,
                            color=colors.get(priority, 'blue'),
                            opacity=0.7
                        ),
                        name=f'{priority} Priority',
                        hovertemplate="<b>%{x} - %{fullData.name}</b><br>Avg Resolution: %{y} days<br>Count: %{marker.size}<extra></extra>"
                    ), row=1, col=1)
        
        # 2. Priority Distribution Over Time (Stacked Bar)
        if 'year_month' in self.df.columns and 'priority' in self.df.columns:
            priority_time = pd.crosstab(self.df['year_month'], self.df['priority'])
            colors_priority = ['green', 'yellow', 'orange', 'red']
            
            for i, priority in enumerate(priority_time.columns):
                fig.add_trace(go.Bar(
                    x=priority_time.index,
                    y=priority_time[priority],
                    name=priority,
                    marker_color=colors_priority[i % len(colors_priority)],
                    hovertemplate=f"<b>%{{x}} - {priority}</b><br>Count: %{{y}}<extra></extra>"
                ), row=1, col=2)
        
        # 3. Category Performance Matrix (Scatter)
        if all(col in self.df.columns for col in ['category', 'resolution_time_days', 'aging_days']):
            category_performance = self.df.groupby('category').agg({
                'resolution_time_days': 'mean',
                'aging_days': 'mean',
                'ticket': 'count'
            }).dropna()
            
            if not category_performance.empty:
                fig.add_trace(go.Scatter(
                    x=category_performance['resolution_time_days'],
                    y=category_performance['aging_days'],
                    mode='markers+text',
                    marker=dict(
                        size=category_performance['ticket'],
                        sizemode='area',
                        sizeref=2.*max(category_performance['ticket'])/(40.**2),
                        color='purple',
                        opacity=0.6
                    ),
                    text=category_performance.index,
                    textposition="top center",
                    hovertemplate="<b>%{text}</b><br>Avg Resolution: %{x} days<br>Avg Aging: %{y} days<br>Tickets: %{marker.size}<extra></extra>"
                ), row=2, col=1)
        
        # 4. SLA Compliance Gauge
        if 'resolution_time_days' in self.df.columns:
            try:
                closed_tickets = self.df[self.df['resolution_time_days'].notna()]
                if not closed_tickets.empty:
                    sla_compliance = (closed_tickets['resolution_time_days'] <= 7).mean() * 100
                    sla_compliance = float(sla_compliance)  # Ensure it's a float
                    
                    fig.add_trace(go.Indicator(
                        mode="gauge+number+delta",
                        value=sla_compliance,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "7-Day SLA Compliance %"},
                        delta={'reference': 80},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ), row=2, col=2)
                else:
                    # No closed tickets, show default gauge
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=0,
                        title={'text': "7-Day SLA Compliance %"},
                        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "gray"}}
                    ), row=2, col=2)
            except Exception as e:
                print(f"Warning: Could not create SLA gauge: {e}")
                # Add a simple indicator instead
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=0,
                    title={'text': "SLA Data Unavailable"}
                ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Performance Analytics Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=800,
            showlegend=True
        )
        
        # Save and show
        fig.write_html("performance_analytics_dashboard.html")
        fig.show()
        print("⚡ Performance analytics dashboard saved as 'performance_analytics_dashboard.html'")
    
    def create_executive_summary_report(self, summary_stats):
        """
        Create an executive summary with key KPIs and interactive elements
        """
        print("\n📋 Creating executive summary report...")
        
        # Create KPI dashboard
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=('Total Tickets', 'Resolution Time', 'Open Tickets', 'SLA Performance',
                          'Priority Breakdown', 'Monthly Trend', 'Top Categories', 'Channel Mix'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}, {"type": "pie"}]]
        )
        
        # Calculate KPI values (ensuring they are numeric)
        total_tickets = int(summary_stats.get('total_tickets', 0))
        
        # Get average resolution time as a float
        avg_resolution_time = summary_stats.get('resolution_metrics', {}).get('avg_resolution_time', 0)
        if avg_resolution_time is None:
            avg_resolution_time = 0
        avg_resolution_time = float(avg_resolution_time)
        
        # Calculate open tickets
        open_tickets = int(len(self.df[self.df['status'].isin(['Open', 'In Progress', 'Pending'])]))
        
        # Calculate SLA performance (example calculation)
        sla_performance = 85  # Default value
        if 'resolution_time_days' in self.df.columns:
            closed_tickets = self.df[self.df['resolution_time_days'].notna()]
            if not closed_tickets.empty:
                sla_compliance = (closed_tickets['resolution_time_days'] <= 7).mean() * 100
                sla_performance = float(sla_compliance)
        
        # KPI Indicators with numeric values
        kpis = [
            (total_tickets, "Total Tickets", "blue"),
            (avg_resolution_time, "Avg Resolution (Days)", "green"),
            (open_tickets, "Open Tickets", "orange"),
            (sla_performance, "SLA Performance (%)", "red")
        ]
        
        for i, (value, title, color) in enumerate(kpis):
            # Format the number display based on the title
            if "Resolution" in title:
                number_format = ".1f"
            elif "SLA" in title:
                number_format = ".0f"
            else:
                number_format = ","
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=value,
                title={"text": title},
                number={
                    'font': {'size': 40, 'color': color},
                    'valueformat': number_format
                }
            ), row=1, col=i+1)
        
        # Priority Breakdown
        if 'priority_distribution' in summary_stats:
            priority_data = summary_stats['priority_distribution']
            fig.add_trace(go.Pie(
                labels=list(priority_data.keys()),
                values=list(priority_data.values()),
                name="Priority"
            ), row=2, col=1)
        
        # Monthly Trend
        if 'year_month' in self.df.columns:
            monthly_counts = self.df['year_month'].value_counts().sort_index()
            fig.add_trace(go.Scatter(
                x=monthly_counts.index,
                y=monthly_counts.values,
                mode='lines+markers',
                name="Trend",
                line=dict(width=3)
            ), row=2, col=2)
        
        # Top Categories
        if 'category_distribution' in summary_stats:
            category_data = summary_stats['category_distribution']
            top_categories = dict(list(category_data.items())[:5])
            fig.add_trace(go.Bar(
                x=list(top_categories.keys()),
                y=list(top_categories.values()),
                name="Categories"
            ), row=2, col=3)
        
        # Channel Mix
        if 'source_distribution' in summary_stats:
            source_data = summary_stats['source_distribution']
            fig.add_trace(go.Pie(
                labels=list(source_data.keys()),
                values=list(source_data.values()),
                name="Channels"
            ), row=2, col=4)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Executive Summary Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 28}
            },
            height=800
        )
        
        # Save and show
        fig.write_html("executive_summary_dashboard.html")
        fig.show()
        print("📋 Executive summary saved as 'executive_summary_dashboard.html'")
    
    def generate_interactive_insights_report(self, summary_stats):
        """
        Generate detailed insights with interactive elements
        """
        print("\n💡 Generating interactive insights report...")
        
        # Create insights visualization
        insights_data = []
        
        # Workload Analysis
        try:
            open_tickets = len(self.df[self.df['status'].isin(['Open', 'In Progress', 'Pending'])])
            total_tickets = summary_stats.get('total_tickets', 1)  # Avoid division by zero
            workload_percentage = (open_tickets / total_tickets) * 100 if total_tickets > 0 else 0
            
            insights_data.append({
                'metric': 'Workload %',
                'value': float(workload_percentage),
                'status': 'High' if workload_percentage > 40 else 'Normal',
                'recommendation': 'Consider resource allocation review' if workload_percentage > 40 else 'Workload manageable'
            })
        except Exception as e:
            print(f"Warning: Could not calculate workload metrics: {e}")
        
        # Resolution Time Analysis
        try:
            if 'resolution_metrics' in summary_stats and summary_stats['resolution_metrics']:
                avg_resolution = summary_stats['resolution_metrics'].get('avg_resolution_time', 0)
                if avg_resolution is not None:
                    avg_resolution = float(avg_resolution)
                    insights_data.append({
                        'metric': 'Avg Resolution Days',
                        'value': avg_resolution,
                        'status': 'Slow' if avg_resolution > 7 else 'Good',
                        'recommendation': 'Focus on process optimization' if avg_resolution > 7 else 'Maintain current performance'
                    })
        except Exception as e:
            print(f"Warning: Could not calculate resolution metrics: {e}")
        
        # Priority Analysis
        try:
            if 'priority_distribution' in summary_stats:
                priority_dist = summary_stats['priority_distribution']
                high_priority_count = priority_dist.get('High', 0) + priority_dist.get('Critical', 0)
                total_tickets = summary_stats.get('total_tickets', 1)
                high_priority_pct = (high_priority_count / total_tickets) * 100 if total_tickets > 0 else 0
                
                insights_data.append({
                    'metric': 'High Priority %',
                    'value': float(high_priority_pct),
                    'status': 'High' if high_priority_pct > 25 else 'Normal',
                    'recommendation': 'Review priority assignment' if high_priority_pct > 25 else 'Priority distribution healthy'
                })
        except Exception as e:
            print(f"Warning: Could not calculate priority metrics: {e}")
        
        # Aging Analysis
        try:
            if 'aging_metrics' in summary_stats and summary_stats['aging_metrics']:
                aging_metrics = summary_stats['aging_metrics']
                old_tickets = aging_metrics.get('tickets_over_30_days', 0)
                if old_tickets is not None:
                    old_tickets = int(old_tickets)
                    insights_data.append({
                        'metric': 'Tickets > 30 Days',
                        'value': float(old_tickets),
                        'status': 'High' if old_tickets > 5 else 'Normal',
                        'recommendation': 'Focus on aging tickets' if old_tickets > 5 else 'Aging under control'
                    })
        except Exception as e:
            print(f"Warning: Could not calculate aging metrics: {e}")
        
        # If no insights data could be generated, create a default message
        if not insights_data:
            insights_data.append({
                'metric': 'Data Available',
                'value': float(len(self.df)),
                'status': 'Normal',
                'recommendation': 'Basic analysis completed'
            })
        
        # Create insights dashboard
        fig = go.Figure()
        
        metrics = [d['metric'] for d in insights_data]
        values = [d['value'] for d in insights_data]
        statuses = [d['status'] for d in insights_data]
        recommendations = [d['recommendation'] for d in insights_data]
        
        colors = ['red' if s in ['High', 'Slow'] else 'green' for s in statuses]
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Value: %{y}<br>Status: %{customdata[0]}<br>Recommendation: %{customdata[1]}<extra></extra>",
            customdata=list(zip(statuses, recommendations))
        ))
        
        fig.update_layout(
            title={
                'text': "Key Performance Insights",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Metrics",
            yaxis_title="Values",
            height=600,
            showlegend=False
        )
        
        fig.write_html("insights_dashboard.html")
        fig.show()
        print("💡 Insights dashboard saved as 'insights_dashboard.html'")
    
    def run_interactive_analysis(self):
        """
        Run the complete interactive analysis pipeline
        """
        print("🚀 Starting Interactive Support Ticket Analysis...")
        print("=" * 60)
        
        # Step 1: Read data
        if not self.read_excel_data():
            return
        
        # Step 2: Clean data
        self.data_cleaning_and_preparation()
        
        # Step 3: Generate statistics
        summary_stats = self.generate_summary_statistics()
        
        # Step 4: Create interactive visualizations
        self.create_interactive_overview_dashboard()
        self.create_trend_analysis_dashboard()
        self.create_performance_analytics_dashboard()
        self.create_executive_summary_report(summary_stats)
        self.generate_interactive_insights_report(summary_stats)
        
        # Step 5: Save sample data to Excel with new columns
        try:
            self.df.to_excel('analyzed_tickets_with_metrics.xlsx', index=False)
            print("💾 Analyzed data saved to 'analyzed_tickets_with_metrics.xlsx'")
        except Exception as e:
            print(f"⚠️  Could not save Excel file: {e}")
        
        print("\n🎉 Interactive Analysis Complete!")
        print("\n📊 Generated Interactive Dashboards:")
        print("1. 📊 interactive_overview_dashboard.html - Main overview with key metrics")
        print("2. 📈 interactive_trend_dashboard.html - Comprehensive trend analysis")
        print("3. ⚡ performance_analytics_dashboard.html - Advanced performance analytics")
        print("4. 📋 executive_summary_dashboard.html - Executive KPI dashboard")
        print("5. 💡 insights_dashboard.html - Key insights and recommendations")
        
        print("\n✨ Interactive Features Include:")
        print("• Hover tooltips with detailed information")
        print("• Zoom and pan capabilities")
        print("• Interactive legends (click to show/hide)")
        print("• Responsive design for different screen sizes")
        print("• Export options (PNG, PDF, HTML)")

def create_sample_data_with_dates():
    """
    Create realistic sample data for support tickets with date columns
    """
    print("🔨 Creating enhanced sample support ticket data...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Define realistic data for each column
    ticket_types = ['Bug', 'Feature Request', 'Support', 'Enhancement', 'Documentation', 'Training']
    categories = ['UI/UX', 'Performance', 'Security', 'Integration', 'Database', 'API', 
                 'Authentication', 'Reporting', 'Mobile', 'Configuration']
    sources = ['Email', 'Web Portal', 'Phone', 'Chat', 'Slack', 'Internal', 'Customer Portal']
    statuses = ['Open', 'In Progress', 'Resolved', 'Closed', 'Pending', 'Waiting for Customer']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    tshirt_sizes = ['XS', 'S', 'M', 'L', 'XL']
    
    # Create sample data with realistic distributions
    n_records = 500
    
    # Generate date_reported spanning the last 12 months
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now() - timedelta(days=1)
    date_range = (end_date - start_date).days
    
    # Create dates with some clustering (more tickets on weekdays)
    reported_dates = []
    for i in range(n_records):
        random_days = np.random.randint(0, date_range)
        date = start_date + timedelta(days=random_days)
        # Add weekday bias (less tickets on weekends)
        if date.weekday() >= 5:  # Weekend
            if np.random.random() > 0.3:  # 70% chance to move to weekday
                days_to_add = np.random.randint(1, 3)
                date += timedelta(days=days_to_add)
        reported_dates.append(date)
    
    # Generate sample data with weighted probabilities
    sample_data = {
        'Type': np.random.choice(ticket_types, n_records, 
                               p=[0.35, 0.20, 0.25, 0.10, 0.05, 0.05]),
        'Category': np.random.choice(categories, n_records),
        'Source': np.random.choice(sources, n_records, 
                                 p=[0.25, 0.30, 0.15, 0.15, 0.05, 0.05, 0.05]),
        'Ticket#': [f'TKT-{1000 + i}' for i in range(n_records)],
        'Status': np.random.choice(statuses, n_records, 
                                 p=[0.15, 0.20, 0.35, 0.20, 0.05, 0.05]),
        'Priority': np.random.choice(priorities, n_records, 
                                   p=[0.40, 0.35, 0.20, 0.05]),
        'T-shirt Size': np.random.choice(tshirt_sizes, n_records, 
                                       p=[0.15, 0.25, 0.35, 0.20, 0.05]),
        'Date_Reported': reported_dates
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Generate actual_closed_Date based on status and priority
    closed_dates = []
    for idx, row in df.iterrows():
        if row['Status'] in ['Resolved', 'Closed']:
            # Calculate realistic resolution time based on priority and size
            priority_multiplier = {'Critical': 0.5, 'High': 1.0, 'Medium': 2.0, 'Low': 4.0}
            size_multiplier = {'XS': 0.5, 'S': 1.0, 'M': 2.0, 'L': 4.0, 'XL': 8.0}
            
            base_days = priority_multiplier.get(row['Priority'], 2.0) * size_multiplier.get(row['T-shirt Size'], 2.0)
            resolution_days = max(1, int(np.random.normal(base_days, base_days * 0.3)))
            
            closed_date = row['Date_Reported'] + timedelta(days=resolution_days)
            # Ensure closed date is not in the future
            closed_date = min(closed_date, datetime.now())
            closed_dates.append(closed_date)
        else:
            closed_dates.append(None)  # Open tickets have no close date
    
    df['Actual_Closed_Date'] = closed_dates
    
    # Add some missing values to make it realistic (3% missing data)
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices[:10], 'Category'] = np.nan
    df.loc[missing_indices[10:15], 'Priority'] = np.nan
    df.loc[missing_indices[15:20], 'T-shirt Size'] = np.nan
    
    return df

def main():
    """
    Main execution function with interactive analysis
    """
    print("🚀 Interactive Support Tickets Analysis")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data_with_dates()
    
    # Save sample data to Excel
    try:
        df.to_excel('product_issues_interactive_sample.xlsx', sheet_name='product_issues', index=False)
        print("💾 Sample data saved to 'product_issues_interactive_sample.xlsx'")
    except Exception as e:
        print(f"⚠️  Could not save Excel file: {e}")
        return
    
    print(f"\n✅ Sample data created successfully!")
    print(f"📊 Generated {len(df)} support tickets with interactive analysis")
    
    # Create analyzer instance
    analyzer = InteractiveSupportTicketAnalyzer('product_issues_interactive_sample.xlsx')
    
    # Run the complete interactive analysis
    analyzer.run_interactive_analysis()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Open the generated HTML files in your web browser")
    print("2. Explore the interactive features (hover, zoom, click)")
    print("3. Replace the sample file with your actual Excel data")
    print("4. Customize visualizations as needed")
    print("5. Share dashboards with stakeholders")

if __name__ == "__main__":
    main()
