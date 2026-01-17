"""Visualization utilities for multi-omics data."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

logger = logging.getLogger(__name__)


class VisualizationUtils:
    """Utility functions for creating visualizations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualization utilities.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_figsize = self.config.get('figsize', (12, 8))
        self.default_dpi = self.config.get('dpi', 300)
        self.color_palette = self.config.get('color_palette', 'Set2')
        self.font_size = self.config.get('font_size', 12)
        
        # Set default styles
        self._setup_default_styles()
        
        logger.info("Initialized VisualizationUtils")
    
    def _setup_default_styles(self):
        """Setup default matplotlib and seaborn styles."""
        # Seaborn style
        sns.set_style("whitegrid")
        sns.set_palette(self.color_palette)
        
        # Matplotlib defaults
        plt.rcParams['figure.figsize'] = self.default_figsize
        plt.rcParams['figure.dpi'] = self.default_dpi
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['axes.titlesize'] = self.font_size + 2
        plt.rcParams['axes.labelsize'] = self.font_size
        plt.rcParams['xtick.labelsize'] = self.font_size - 2
        plt.rcParams['ytick.labelsize'] = self.font_size - 2
        plt.rcParams['legend.fontsize'] = self.font_size - 2
    
    def create_multi_omics_color_palette(self, omics_types: List[str]) -> Dict[str, str]:
        """Create a consistent color palette for different omics types."""
        # Define standard colors for common omics types
        omics_colors = {
            'gene_expression': '#1f77b4',  # Blue
            'proteomics': '#ff7f0e',       # Orange
            'metabolomics': '#2ca02c',     # Green
            'clinical': '#d62728',         # Red
            'cnv': '#9467bd',              # Purple
            'mutation': '#8c564b',         # Brown
            'methylation': '#e377c2',      # Pink
            'mirna': '#7f7f7f',            # Gray
            'transcriptomics': '#17becf',  # Cyan
            'lipidomics': '#bcbd22'        # Olive
        }
        
        # Create palette
        palette = {}
        available_colors = list(plt.cm.Set3.colors) + list(plt.cm.tab20.colors)
        
        for i, omics_type in enumerate(omics_types):
            # Use predefined color if available
            if omics_type.lower() in omics_colors:
                palette[omics_type] = omics_colors[omics_type.lower()]
            else:
                # Use color from palette
                palette[omics_type] = available_colors[i % len(available_colors)]
        
        return palette
    
    def create_heatmap_with_dendrogram(self, data: pd.DataFrame, 
                                     method: str = 'ward',
                                     metric: str = 'euclidean',
                                     title: str = "Heatmap with Dendrogram",
                                     **kwargs) -> plt.Figure:
        """Create a heatmap with hierarchical clustering dendrogram."""
        fig = plt.figure(figsize=kwargs.get('figsize', self.default_figsize))
        
        # Create grid spec for dendrogram and heatmap
        gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05,
                             height_ratios=[0.3, 1], width_ratios=[1, 0.3])
        
        # Main heatmap axis
        ax_heatmap = fig.add_subplot(gs[1, 0])
        
        # Dendrogram axes
        ax_dendro_row = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
        ax_dendro_col = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
        
        # Hide dendrogram tick labels
        ax_dendro_row.axis('off')
        ax_dendro_col.axis('off')
        
        # Perform clustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Row clustering
        row_linkage = linkage(pdist(data.values), method=method, metric=metric)
        row_dendro = dendrogram(row_linkage, ax=ax_dendro_row, orientation='top')
        
        # Column clustering
        col_linkage = linkage(pdist(data.values.T), method=method, metric=metric)
        col_dendro = dendrogram(col_linkage, ax=ax_dendro_col, orientation='left')
        
        # Get cluster indices
        row_indices = row_dendro['leaves']
        col_indices = col_dendro['leaves']
        
        # Reorder data
        data_clustered = data.iloc[row_indices, col_indices]
        
        # Create heatmap
        sns.heatmap(data_clustered, ax=ax_heatmap, cmap='viridis',
                   cbar_kws={'label': 'Value'})
        
        ax_heatmap.set_title(title)
        
        return fig
    
    def create_interactive_scatter_plot(self, data: pd.DataFrame, 
                                      x_col: str, y_col: str,
                                      color_col: Optional[str] = None,
                                      size_col: Optional[str] = None,
                                      hover_cols: Optional[List[str]] = None,
                                      title: str = "Interactive Scatter Plot") -> go.Figure:
        """Create an interactive scatter plot using Plotly."""
        fig = go.Figure()
        
        # Prepare hover data
        hover_data = hover_cols if hover_cols else []
        if color_col and color_col not in hover_data:
            hover_data.append(color_col)
        if size_col and size_col not in hover_data:
            hover_data.append(size_col)
        
        # Create scatter plot
        scatter_fig = px.scatter(
            data, x=x_col, y=y_col, color=color_col, size=size_col,
            hover_data=hover_data, title=title
        )
        
        return scatter_fig
    
    def create_interactive_heatmap(self, data: pd.DataFrame, 
                                 title: str = "Interactive Heatmap") -> go.Figure:
        """Create an interactive heatmap using Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='viridis',
            hovertemplate='Sample: %{y}<br>Feature: %{x}<br>Value: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Samples",
            height=600
        )
        
        return fig
    
    def save_plot(self, fig: Union[plt.Figure, go.Figure], 
                  output_path: str, format: str = 'png', **kwargs) -> str:
        """Save plot in various formats."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, plt.Figure):
            # Matplotlib figure
            fig.savefig(output_path, format=format, dpi=kwargs.get('dpi', self.default_dpi),
                       bbox_inches='tight', **kwargs)
        elif isinstance(fig, go.Figure):
            # Plotly figure
            if format == 'html':
                fig.write_html(output_path)
            elif format == 'png':
                fig.write_image(output_path, width=kwargs.get('width', 1200),
                              height=kwargs.get('height', 800))
            else:
                fig.write_image(output_path, format=format)
        
        return str(output_path)
    
    def create_subplot_grid(self, plots: List[plt.Figure], 
                          n_cols: int = 2, title: str = "Subplot Grid") -> plt.Figure:
        """Create a grid of subplots from individual plots."""
        n_plots = len(plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create main figure
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 6))
        fig.suptitle(title, fontsize=16)
        
        # Create grid
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
        
        for i, plot_fig in enumerate(plots):
            row = i // n_cols
            col = i % n_cols
            
            # Create subplot axis
            ax = fig.add_subplot(gs[row, col])
            
            # Copy content from original plot
            # This is a simplified approach - in practice, you'd recreate the plot
            ax.text(0.5, 0.5, f"Plot {i+1}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f"Subplot {i+1}")
        
        return fig
    
    def create_quality_report_layout(self, qc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a layout for quality control report."""
        layout = {
            'title': 'Multi-Omics Quality Control Report',
            'sections': [
                {
                    'name': 'overview',
                    'title': 'Overview',
                    'plots': ['overall_quality', 'summary_statistics']
                },
                {
                    'name': 'missing_data',
                    'title': 'Missing Data Analysis',
                    'plots': ['missing_data_heatmap', 'missing_data_by_column', 'completeness_overview']
                },
                {
                    'name': 'distributions',
                    'title': 'Data Distributions',
                    'plots': ['distribution_plots', 'normality_tests', 'outlier_detection']
                },
                {
                    'name': 'correlations',
                    'title': 'Feature Correlations',
                    'plots': ['correlation_heatmap', 'high_correlations', 'condition_number']
                },
                {
                    'name': 'integration',
                    'title': 'Data Integration',
                    'plots': ['sample_alignment', 'feature_overlap', 'integration_quality']
                }
            ],
            'summary_stats': {
                'total_data_types': len(qc_results.get('metrics', {})),
                'overall_quality': qc_results.get('overall_quality', 'UNKNOWN'),
                'failed_tests': len(qc_results.get('failed_tests', [])),
                'warnings': len(qc_results.get('warnings', []))
            }
        }
        
        return layout
    
    def export_plotly_to_static(self, fig: go.Figure, output_path: str, 
                               format: str = 'png', width: int = 1200, height: int = 800) -> str:
        """Export Plotly figure to static format."""
        try:
            # Set default renderer
            pio.kaleido.scope.default_format = format
            pio.kaleido.scope.default_width = width
            pio.kaleido.scope.default_height = height
            
            # Save figure
            fig.write_image(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export Plotly figure: {e}")
            raise
    
    def create_publication_ready_figure(self, fig: plt.Figure, 
                                      title: str = "",
                                      xlabel: str = "", ylabel: str = "",
                                      font_size: int = 12) -> plt.Figure:
        """Make figure publication-ready with consistent styling."""
        # Update font sizes
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.titlesize'] = font_size + 2
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['xtick.labelsize'] = font_size - 2
        plt.rcParams['ytick.labelsize'] = font_size - 2
        plt.rcParams['legend.fontsize'] = font_size - 2
        
        # Apply to current figure
        fig.suptitle(title, fontsize=font_size + 4)
        
        # Update all axes
        for ax in fig.get_axes():
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=font_size)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=font_size)
            
            # Make lines thicker
            for line in ax.get_lines():
                line.set_linewidth(2)
            
            # Make spines thicker
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
        # Tight layout
        fig.tight_layout()
        
        return fig
    
    def add_statistical_annotations(self, ax: plt.Axes, data: pd.DataFrame, 
                                  x_col: str, y_col: str, 
                                  test_type: str = 'ttest') -> None:
        """Add statistical significance annotations to plots."""
        try:
            from scipy import stats
            
            groups = data[x_col].unique()
            
            if len(groups) == 2:
                # Two-group comparison
                group1 = data[data[x_col] == groups[0]][y_col]
                group2 = data[data[x_col] == groups[1]][y_col]
                
                if test_type == 'ttest':
                    stat, p_value = stats.ttest_ind(group1, group2)
                elif test_type == 'mann-whitney':
                    stat, p_value = stats.mannwhitneyu(group1, group2)
                else:
                    return
                
                # Add annotation
                y_max = data[y_col].max()
                annotation = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
                
                ax.annotate(annotation, 
                           xy=(0.5, y_max * 1.1), 
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
                
                # Add significance bars
                ax.plot([0, 1], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1)
                
        except ImportError:
            logger.warning("scipy not available for statistical annotations")
        except Exception as e:
            logger.warning(f"Failed to add statistical annotations: {e}")
    
    def create_color_gradient(self, n_colors: int, 
                            start_color: str = '#1f77b4', 
                            end_color: str = '#ff7f0e') -> List[str]:
        """Create a color gradient."""
        cmap = LinearSegmentedColormap.from_custom(
            'custom_gradient', [start_color, end_color], N=n_colors
        )
        
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        hex_colors = ['#%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) 
                     for c in colors]
        
        return hex_colors
    
    def optimize_figure_size(self, n_subplots: int, 
                           subplot_aspect: Tuple[int, int] = (4, 3),
                           max_width: int = 20, max_height: int = 12) -> Tuple[float, float]:
        """Calculate optimal figure size for multiple subplots."""
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_subplots)))
        n_rows = int(np.ceil(n_subplots / n_cols))
        
        # Calculate figure size
        fig_width = min(n_cols * subplot_aspect[0], max_width)
        fig_height = min(n_rows * subplot_aspect[1], max_height)
        
        return (fig_width, fig_height)
    
    def save_plot_collection(self, plots: Dict[str, Union[plt.Figure, go.Figure]], 
                           output_dir: str, prefix: str = "plot", **kwargs) -> Dict[str, str]:
        """Save a collection of plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for plot_name, plot_fig in plots.items():
            try:
                # Generate filename
                filename = f"{prefix}_{plot_name}.{kwargs.get('format', 'png')}"
                file_path = output_path / filename
                
                # Save plot
                saved_path = self.save_plot(plot_fig, str(file_path), **kwargs)
                saved_files[plot_name] = saved_path
                
            except Exception as e:
                logger.error(f"Failed to save plot {plot_name}: {e}")
                continue
        
        return saved_files
    
    def create_animation(self, data_frames: List[pd.DataFrame], 
                        x_col: str, y_col: str,
                        animation_col: str = 'time_point',
                        title: str = "Animated Plot") -> go.Figure:
        """Create an animated plot using Plotly."""
        # Create initial frame
        fig = go.Figure()
        
        # Add traces for each time point
        for i, df in enumerate(data_frames):
            frame_data = df[df[animation_col] == df[animation_col].unique()[i]]
            
            fig.add_trace(
                go.Scatter(
                    x=frame_data[x_col],
                    y=frame_data[y_col],
                    mode='markers',
                    name=f'Frame {i}',
                    visible=(i == 0)  # Only first frame visible initially
                )
            )
        
        # Create animation frames
        frames = []
        for i in range(len(data_frames)):
            frame_traces = []
            for j in range(len(data_frames)):
                frame_traces.append(go.Frame(
                    data=[go.Scatter(visible=(j <= i))],
                    name=f'Frame {i}'
                ))
            frames.append(go.Frame(data=frame_traces, name=f'Frame {i}'))
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title=title,
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                       'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                         'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig