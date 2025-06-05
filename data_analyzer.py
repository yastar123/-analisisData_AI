import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.file_type = None
        
    def read_file(self, file_path):
        """Read data from various file types"""
        file_ext = file_path.split('.')[-1].lower()
        self.file_type = file_ext
        
        try:
            if file_ext in ['xlsx', 'xls']:
                self.df = pd.read_excel(file_path)
            elif file_ext == 'csv':
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
            return self.get_basic_info()
            
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
    
    def get_basic_info(self):
        """Get basic information about the dataset"""
        if self.df is None:
            raise Exception("No data loaded")
            
        info = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df.describe().to_dict() if self.df.select_dtypes(include=[np.number]).columns.any() else None,
            "categorical_summary": self.df.describe(include=['object']).to_dict() if self.df.select_dtypes(include=['object']).columns.any() else None
        }
        
        return info
    
    def analyze_data(self):
        """Perform comprehensive data analysis"""
        if self.df is None:
            raise Exception("No data loaded")
            
        analysis = {
            "correlations": self._get_correlations(),
            "distributions": self._get_distributions(),
            "patterns": self._find_patterns(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _get_correlations(self):
        """Calculate correlations between numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
            
        corr_matrix = numeric_df.corr()
        return corr_matrix.to_dict()
    
    def _get_distributions(self):
        """Analyze distributions of numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
            
        distributions = {}
        for col in numeric_df.columns:
            distributions[col] = {
                "mean": float(numeric_df[col].mean()),
                "median": float(numeric_df[col].median()),
                "std": float(numeric_df[col].std()),
                "skew": float(numeric_df[col].skew()),
                "kurtosis": float(numeric_df[col].kurtosis())
            }
        return distributions
    
    def _find_patterns(self):
        """Find interesting patterns in the data"""
        patterns = []
        
        # Check for correlations
        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.7)
            high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                        for x, y in zip(*high_corr) if x != y and x < y]
            if high_corr:
                patterns.append(f"Ditemukan korelasi kuat antara: {', '.join([f'{x} dan {y} ({v:.2f})' for x, y, v in high_corr])}")
        
        # Check for outliers
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = numeric_df[(numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))][col]
            if not outliers.empty:
                patterns.append(f"Outliers detected in {col}: {len(outliers)} points")
        
        return patterns
    
    def _generate_recommendations(self):
        """Generate recommendations for analysis and visualization in a structured format"""
        recommendations = []
        
        # Check data types and suggest appropriate visualizations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) > 0:
            recommendations.append({'type': 'info', 'text': 'Kolom numerik terdeteksi. Pertimbangkan untuk membuat:'})
            # Histogram recommendations for each numeric column
            for col in numeric_cols:
                recommendations.append({'type': 'histogram', 'text': f'Histogram untuk {col}', 'params': {'column': col}})
            
            # Box plot recommendations for each numeric column
            for col in numeric_cols:
                 recommendations.append({'type': 'box', 'text': f'Box plot untuk {col}', 'params': {'column': col}})

            if len(numeric_cols) > 1:
                # Scatter plot recommendations (suggest pairs? Or just a generic one?)
                # For simplicity, let's suggest a scatter matrix or specific pairs if few columns
                if len(numeric_cols) <= 5: # Limit pairs to avoid too many recommendations
                     for i in range(len(numeric_cols)):
                         for j in range(i + 1, len(numeric_cols)):
                             col1 = numeric_cols[i]
                             col2 = numeric_cols[j]
                             recommendations.append({'type': 'scatter', 'text': f'Scatter plot: {col1} vs {col2}', 'params': {'x_column': col1, 'y_column': col2}})

                recommendations.append({'type': 'heatmap', 'text': 'Heatmap Korelasi (untuk kolom numerik)', 'params': {}})
        
        if len(categorical_cols) > 0:
            recommendations.append({'type': 'info', 'text': 'Kolom kategorikal terdeteksi. Pertimbangkan untuk membuat:'})
            # Bar plot recommendations for each categorical column
            for col in categorical_cols:
                recommendations.append({'type': 'bar', 'text': f'Bar plot untuk {col}', 'params': {'column': col}})
            
            # Pie chart recommendations for each categorical column (if not too many unique values)
            for col in categorical_cols:
                if self.df[col].nunique() <= 10: # Limit pie charts to columns with <= 10 unique values
                    recommendations.append({'type': 'pie', 'text': f'Pie chart untuk {col}', 'params': {'column': col}})

            # Box plots grouped by categories (if both numeric and categorical exist)
            if len(numeric_cols) > 0:
                recommendations.append({'type': 'info', 'text': 'Consider visualizing numeric data by categories:'})
                for num_col in numeric_cols:
                    for cat_col in categorical_cols:
                        recommendations.append({'type': 'box', 'text': f'Box plot of {num_col} by {cat_col}', 'params': {'column': num_col, 'group_by': cat_col}})

        return recommendations
    
    def create_visualization(self, viz_type, **kwargs):
        """Create various types of visualizations"""
        if self.df is None:
            raise Exception("No data loaded")
            
        try:
            if viz_type == "histogram":
                return self._create_histogram(**kwargs)
            elif viz_type == "scatter":
                return self._create_scatter(**kwargs)
            elif viz_type == "box":
                return self._create_boxplot(**kwargs)
            elif viz_type == "bar":
                return self._create_barplot(**kwargs)
            elif viz_type == "heatmap":
                return self._create_heatmap(**kwargs)
            elif viz_type == "pie":
                return self._create_piechart(**kwargs)
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
                
        except Exception as e:
            raise Exception(f"Error creating visualization: {str(e)}")
    
    def _create_histogram(self, column, bins=30):
        """Create histogram plot"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, bins=bins)
        plt.title(f'Distribusi {column}')
        return self._fig_to_base64()
    
    def _create_scatter(self, x_column, y_column, color_column=None):
        """Create scatter plot"""
        fig = px.scatter(self.df, x=x_column, y=y_column, color=color_column,
                        title=f'Scatter Plot: {x_column} vs {y_column}')
        return self._fig_to_base64(fig)
    
    def _create_boxplot(self, column, group_by=None):
        """Create box plot"""
        plt.figure(figsize=(10, 6))
        if group_by:
            sns.boxplot(data=self.df, x=group_by, y=column)
        else:
            sns.boxplot(data=self.df, y=column)
        plt.title(f'Box Plot dari {column}')
        return self._fig_to_base64()
    
    def _create_barplot(self, column, group_by=None):
        """Create bar plot"""
        plt.figure(figsize=(10, 6))
        if group_by:
            sns.barplot(data=self.df, x=group_by, y=column)
        else:
            self.df[column].value_counts().plot(kind='bar')
        plt.title(f'Bar Plot dari {column}')
        return self._fig_to_base64()
    
    def _create_heatmap(self):
        """Create correlation heatmap"""
        plt.figure(figsize=(10, 8))
        numeric_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Heatmap Korelasi')
        return self._fig_to_base64()
    
    def _create_piechart(self, column):
        """Create pie chart"""
        plt.figure(figsize=(10, 8))
        self.df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Pie Chart dari {column}')
        return self._fig_to_base64()
    
    def _fig_to_base64(self, fig=None):
        """Convert matplotlib/plotly figure to base64 string"""
        if fig is None:
            fig = plt.gcf()
            
        if isinstance(fig, (plt.Figure, plt.Axes)):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            return img_str
        else:  # Plotly figure
            return fig.to_image(format="png") 