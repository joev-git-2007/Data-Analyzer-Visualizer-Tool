# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO           #To convert csv contect to a format where pandas can read

#streamlit refreshes every moment after user interacts and starts executing from start. So inorder not to recreate everything, we use session_state.
if 'data' not in st.session_state:      
    st.session_state.data = None
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'charts_selected' not in st.session_state:
    st.session_state.charts_selected = []

st.set_page_config( page_title="DVizual", page_icon="📊", layout="wide")

def csv(data):
    try:
        str_content = data.decode('utf-8')  # Decode bytes to string
        df = pd.read_csv(StringIO(str_content))  # pandas reads csv which is converted to readable format (StringIO)
        return df
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        return None

def analyze_data(df):
    if df is None:
        return {}
    
    analysis = {
        'shape': df.shape,  # rows, cols
        'columns': df.columns.tolist(),  # cols stored to "columns" list
        'dtypes': df.dtypes.to_dict(),  # datatypes to each col
        'missing': df.isnull().sum().to_dict(),  # if null, stores sum
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),  # selects numeric cols only
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(), # selects text cols
        'preview': df.head(10),  # first 10 rows
        'describe': df.describe(include='all')  #full statistical summary
    }
    return analysis

def summary(analysis):
    
    rows, cols = analysis['shape'] # assigns rows and cols to variables
    missing_total = sum(analysis['missing'].values())  #total sum of missing values
    if (rows * cols) > 0:
        completeness = ((rows * cols - missing_total) / (rows * cols) * 100) 
    else:
        completeness = 0
    
    s1 = [] #list which stores what to display in summary part
    
    # Executive Summary
    s1.append("AI REPORT SUMMARY")
    s1.append("=" * 20)
    s1.append(f"This dataset contains {rows:,} rows and {cols} columns with {completeness:.1f}% data completeness.")
    
    # Data Structure
    s1.append("\nDATA STRUCTURE")
    s1.append("=" * 15)
    num_numeric = len(analysis['numeric_cols'])
    num_categorical = len(analysis['categorical_cols'])
    s1.append(f"- Numeric columns: {num_numeric}")
    s1.append(f"- Categorical columns: {num_categorical}")
    s1.append(f"- Total missing values: {missing_total:,}")
    
    # Column Insights
    s1.append("\nCOLUMN INSIGHTS")
    s1.append("=" * 15)
    
    # Find interesting columns
    single_value_cols = []   #finds cols that contain only one non-null value
    for col in analysis['columns']:
        if analysis['describe'].loc['count', col] == 1:
            single_value_cols.append(col) #if a particular col has one null value, append it
    
    unique_cols = []
    for col in analysis['columns']:
        unique_value_count = analysis['describe'].loc['unique', col]
        total_non_null_count = analysis['describe'].loc['count', col]
        
        if unique_value_count == total_non_null_count:
            unique_cols.append(col)
    
    if single_value_cols:
        s1.append(f"Constant columns (single value): {', '.join(single_value_cols)}")
    if unique_cols:
        s1.append(f"Unique identifier columns: {', '.join(unique_cols)}")
    
    # Data Quality
    s1.append("\nDATA QUALITY ASSESSMENT")
    s1.append("=" * 23)
    if missing_total == 0:
        s1.append("✓ Excellent data completeness - no missing values detected")
    else:
        s1.append(f"⚠ {missing_total:,} missing values found across {len([k for k, v in analysis['missing'].items() if v > 0])} columns")
    
    # Recommendations
    s1.append("\nRECOMMENDATIONS")
    s1.append("=" * 15)
    if missing_total > 0:
        s1.append("- Investigate missing value patterns")
    if single_value_cols:
        s1.append("- Consider removing constant columns")
    if num_numeric > 0:
        s1.append("- Explore correlations between numeric variables")
    if num_categorical > 0:
        s1.append("- Analyze categorical distributions for insights")
    
    return "\n".join(s1)

def create_distribution_chart(df, column):
    """Create distribution chart for numeric columns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    df[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')   #histogram
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    return fig

def create_categorical_chart(df, column):
    """Create bar chart for categorical columns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts = df[column].value_counts().head(15)  # Top 15 values
    value_counts.plot(kind='bar', ax=ax, color='lightcoral')  #bar graph
    ax.set_title(f'Top Values in {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, numeric_cols):
    """Create correlation heatmap"""
    if len(numeric_cols) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    ax.set_title('Correlation Matrix')
    
    # Set ticks and labels
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    return fig

def create_scatter_plot(df, x_col, y_col):
    """Create scatter plot for two numeric columns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x=x_col, y=y_col, kind='scatter', ax=ax, alpha=0.7, color='purple')
    ax.set_title(f'{y_col} vs {x_col}')
    plt.tight_layout()
    return fig

def main():
    st.title("📊 Advanced Data Visualization & Analysis Tool")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'], help="Upload a CSV file for comprehensive analysis" )
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            file_content = uploaded_file.getvalue()  #extracts the data in bytes from uploaded file
            df = csv(file_content)  # earlier fn. 
            if df is not None:
                st.session_state.data = df
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to parse the file")
                return
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Main content
    if st.session_state.data is not None:
        df = st.session_state.data
        analysis = analyze_data(df) # earlier fn which extracts each data values from dataset.
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Full Dataset", "🤖 AI Summary", "📊 Visualizations", "📋 Statistics"])
        
        # Tab 1: Full Dataset
        with tab1:
            st.subheader("Complete Dataset View")
            st.dataframe(df, use_container_width=True, height=600)   # Display a dataframe as an interactive table
            
            # Dataset info
            rows, cols = df.shape
            st.info(f"Showing {rows:,} rows × {cols} columns")  #rows:, => 1,000 instead of 1000
            
            # Download button
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False) #converts df to csv in string. Removes pandas indexing
            csv_str = csv_buffer.getvalue() #extracts string from memory
            
            st.download_button( label="📥 Download Full Dataset", data=csv_str, file_name="analyzed_dataset.csv", mime="text/csv" )
        
        # Tab 2: AI Summary
        with tab2:
            st.subheader("Intelligent Data Analysis")
            
            if st.button("🔄 Regenerate Analysis") or not st.session_state.summary:
                with st.spinner("Performing analysis..."):
                    st.session_state.summary = summary(analysis)
            
            # Display summary
            st.markdown("### 📋 Analysis Report")
            st.text_area("", st.session_state.summary, height=500)
            
            # Quick stats
            st.subheader("Quick Stats")
            col1, col2, col3, col4 = st.columns(4) #creates 4 cols
            col1.metric("Rows", f"{analysis['shape'][0]:,}")
            col2.metric("Columns", analysis['shape'][1])
            col3.metric("Numeric", len(analysis['numeric_cols']))
            col4.metric("Categorical", len(analysis['categorical_cols']))
   
        # Tab 3: Visualizations
        with tab3:
            st.subheader("Interactive Visualization Studio")
            
            # Chart selection
            st.markdown("#### Select Visualization Types:")
            chart_options = {
                "Distribution Charts": "Show distributions for numeric columns",
                "Categorical Charts": "Show value counts for text columns",
                "Correlation Heatmap": "Show relationships between numeric variables",
                "Scatter Plots": "Show relationships between two numeric variables"
            }
            
            # Initialize selected_charts
            selected_charts = []
            for chart, description in chart_options.items():
                if st.checkbox(chart, value=True, help=description):
                    selected_charts.append(chart)
            
            # Stores selected charts in session state
            st.session_state.charts_selected = selected_charts
            
            # Generate visualizations button
            if st.button("Generate Selected Charts"):
                if not selected_charts:
                    st.warning("Please select at least one chart type")
                else:
                    # Distribution charts
                    if "Distribution Charts" in selected_charts and analysis['numeric_cols']:
                        st.markdown("### Distribution Charts")
                        cols_per_row = 2
                        numeric_cols = analysis['numeric_cols']
                        
                        for i in range(0, len(numeric_cols), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                                with cols[j]:
                                    try:
                                        fig = create_distribution_chart(df, col_name)
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    except Exception as e:
                                        st.error(f"Could not create chart for {col_name}: {str(e)}")
                    
                    # Categorical charts
                    if "Categorical Charts" in selected_charts and analysis['categorical_cols']:
                        st.markdown("### Categorical Value Charts")
                        cols_per_row = 2
                        cat_cols = analysis['categorical_cols']
                        
                        for i in range(0, len(cat_cols), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col_name in enumerate(cat_cols[i:i+cols_per_row]):
                                with cols[j]:
                                    try:
                                        fig = create_categorical_chart(df, col_name)
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    except Exception as e:
                                        st.error(f"Could not create chart for {col_name}: {str(e)}")
                    
                    # Correlation heatmap
                    if "Correlation Heatmap" in selected_charts and len(analysis['numeric_cols']) > 1:
                        st.markdown("### Correlation Analysis")
                        try:
                            fig = create_correlation_heatmap(df, analysis['numeric_cols'])
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                        except Exception as e:
                            st.error(f"Could not create correlation heatmap: {str(e)}")
                    
                    # Scatter plots
                    if "Scatter Plots" in selected_charts and len(analysis['numeric_cols']) >= 2:
                        st.markdown("### Scatter Plot Generator")
                        numeric_cols = analysis['numeric_cols']
                        
                        # Uses session state to store scatter plot selections
                        if 'x_col' not in st.session_state:
                            st.session_state.x_col = numeric_cols[0] if numeric_cols else None
                        if 'y_col' not in st.session_state:
                            st.session_state.y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
                        
                        col1, col2 = st.columns(2)
                        x_col = col1.selectbox("X-axis variable", numeric_cols, key="x_axis", 
                                            index=numeric_cols.index(st.session_state.x_col) if st.session_state.x_col in numeric_cols else 0)
                        y_col = col2.selectbox("Y-axis variable", numeric_cols, key="y_axis",
                                            index=numeric_cols.index(st.session_state.y_col) if st.session_state.y_col in numeric_cols else 0)
                        
                        # Update session state
                        st.session_state.x_col = x_col
                        st.session_state.y_col = y_col
                        
                        if x_col and y_col and x_col != y_col:
                            try:
                                fig = create_scatter_plot(df, x_col, y_col)
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Could not create scatter plot: {str(e)}")
                        elif x_col == y_col:
                            st.warning("Please select different variables for X and Y axes")
            else:
                st.info("Select chart types above and click 'Generate Selected Charts'")

        
        # Tab 4: Statistics
        with tab4:
            st.subheader("Comprehensive Statistics")
            
            # Data types
            st.markdown("### Data Types")
            dtypes_df = pd.DataFrame(analysis['dtypes'].items(), columns=['Column', 'Data Type'])
            st.dataframe(dtypes_df, use_container_width=True)
            
            # Missing values
            st.markdown("### Missing Values")
            missing_df = pd.DataFrame(analysis['missing'].items(), columns=['Column', 'Missing Count'])
            missing_df['Percentage'] = (missing_df['Missing Count'] / df.shape[0] * 100).round(2) 
            st.dataframe(missing_df, use_container_width=True)
            
            # Descriptive statistics
            st.markdown("### Descriptive Statistics")
            st.dataframe(analysis['describe'], use_container_width=True) #automatically generates stats
            
            # Column details
            st.markdown("### Column Details")
            for col in df.columns:
                with st.expander(f"Details for: {col}"):  #expander is used just to prevent cluttering
                    col_data = df[col]
                    st.write(f"**Data Type:** {col_data.dtype}")
                    st.write(f"**Non-null Count:** {col_data.count()}")
                    st.write(f"**Missing Values:** {col_data.isnull().sum()}")
                    if col_data.dtype in ['int64', 'float64']:
                        st.write(f"**Mean:** {col_data.mean():.2f}")
                        st.write(f"**Std Dev:** {col_data.std():.2f}")
                        st.write(f"**Min:** {col_data.min()}")
                        st.write(f"**Max:** {col_data.max()}")
                    else:
                        st.write(f"**Unique Values:** {col_data.nunique()}")
                        st.write("**Top Values:**")
                        top_values = col_data.value_counts().head(5)
                        for val, count in top_values.items():
                            st.write(f"  - {val}: {count}")

if __name__ == "__main__":
    main()
#__name__ is a builtin var which stores name of modeule being executed.
#When done streamlit run app.py, name stores main, becoz, this file is the main entry of execution
#If app.py is called in another call, name stores the value app

