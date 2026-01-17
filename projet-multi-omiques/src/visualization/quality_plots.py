"""Quality control visualization for multi-omics data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging
from ..exceptions import VisualizationError

logger = logging.getLogger(__name__)


class QualityControlPlots:
    """Create quality control visualizations for multi-omics data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quality control plots.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.figsize = self.config.get('figsize', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        self.color_palette = self.config.get('color_palette', 'Set2')
        self.save_format = self.config.get('save_format', 'png')
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette(self.color_palette)
        
        logger.info("Initialized QualityControlPlots")
    
    def create_qc_report_plots(self, qc_results: Dict[str, Any], 
                              output_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Create comprehensive QC report plots.
        
        Args:
            qc_results: Quality control results
            output_dir: Output directory for plots
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing plot file paths
        """
        logger.info("Creating QC report plots")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        # Create plots for each data type
        for data_type, metrics in qc_results.get('metrics', {}).items():
            if isinstance(metrics, dict) and 'quality_status' in metrics:
                logger.info(f"Creating plots for {data_type}")
                
                try:
                    # Missing data plot
                    if 'missing_data' in metrics:
                        missing_plot = self.plot_missing_data(
                            metrics['missing_data'], 
                            title=f"Missing Data - {data_type}"
                        )
                        missing_file = output_path / f"missing_data_{data_type}.{self.save_format}"
                        missing_plot.savefig(missing_file, dpi=self.dpi, bbox_inches='tight')
                        plt.close(missing_plot)
                        
                        plot_files[f"{data_type}_missing_data"] = str(missing_file)
                    
                    # Distribution plots
                    if 'distributions' in metrics:
                        dist_plot = self.plot_distributions(
                            metrics['distributions'],
                            title=f"Data Distributions - {data_type}"
                        )
                        dist_file = output_path / f"distributions_{data_type}.{self.save_format}"
                        dist_plot.savefig(dist_file, dpi=self.dpi, bbox_inches='tight')
                        plt.close(dist_plot)
                        
                        plot_files[f"{data_type}_distributions"] = str(dist_file)
                    
                    # Correlation plot
                    if 'correlations' in metrics and 'high_correlation_pairs' in metrics['correlations']:
                        corr_plot = self.plot_correlation_heatmap(
                            metrics['correlations'],
                            title=f"Feature Correlations - {data_type}"
                        )
                        corr_file = output_path / f"correlations_{data_type}.{self.save_format}"
                        corr_plot.savefig(corr_file, dpi=self.dpi, bbox_inches='tight')
                        plt.close(corr_plot)
                        
                        plot_files[f"{data_type}_correlations"] = str(corr_file)
                    
                    # Outlier plot
                    if 'outliers' in metrics:
                        outlier_plot = self.plot_outliers(
                            metrics['outliers'],
                            title=f"Outlier Detection - {data_type}"
                        )
                        outlier_file = output_path / f"outliers_{data_type}.{self.save_format}"
                        outlier_plot.savefig(outlier_file, dpi=self.dpi, bbox_inches='tight')
                        plt.close(outlier_plot)
                        
                        plot_files[f"{data_type}_outliers"] = str(outlier_file)
                    
                except Exception as e:
                    logger.error(f"Failed to create plots for {data_type}: {e}")
                    continue
        
        # Create summary dashboard
        try:
            summary_plot = self.create_qc_summary_dashboard(qc_results)
            summary_file = output_path / f"qc_summary_dashboard.{self.save_format}"
            summary_plot.savefig(summary_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(summary_plot)
            
            plot_files['qc_summary_dashboard'] = str(summary_file)
            
        except Exception as e:
            logger.error(f"Failed to create QC summary dashboard: {e}")
        
        logger.info(f"QC plots created: {len(plot_files)} plots")
        
        return {
            'plot_files': plot_files,
            'output_directory': str(output_path),
            'n_plots': len(plot_files)
        }
    
    def plot_missing_data(self, missing_data_metrics: Dict[str, Any], 
                         title: str = "Missing Data Analysis") -> plt.Figure:
        """Plot missing data patterns."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 1. Missing data percentage by column
        if 'columns_missing_percentage' in missing_data_metrics:
            missing_pct = pd.Series(missing_data_metrics['columns_missing_percentage'])
            
            if not missing_pct.empty:
                ax1 = axes[0, 0]
                missing_pct.plot(kind='bar', ax=ax1, color='coral')
                ax1.set_title('Missing Data % by Column')
                ax1.set_xlabel('Columns')
                ax1.set_ylabel('Missing %')
                ax1.tick_params(axis='x', rotation=45)
        
        # 2. Missing data heatmap (sample vs feature)
        ax2 = axes[0, 1]
        if 'missing_patterns' in missing_data_metrics:
            patterns = missing_data_metrics['missing_patterns']
            if patterns:
                pattern_df = pd.DataFrame(list(patterns.items()), columns=['n_missing', 'n_samples'])
                pattern_df.plot(x='n_missing', y='n_samples', kind='bar', ax=ax2, color='lightblue')
                ax2.set_title('Missing Data Patterns')
                ax2.set_xlabel('Number of Missing Values per Sample')
                ax2.set_ylabel('Number of Samples')
        
        # 3. Overall missing data summary
        ax3 = axes[1, 0]
        summary_data = {
            'Complete Cases': missing_data_metrics.get('complete_cases', 0),
            'Cases with Missing': missing_data_metrics.get('rows_with_missing', 0)
        }
        
        colors = ['lightgreen', 'lightcoral']
        ax3.pie(summary_data.values(), labels=summary_data.keys(), colors=colors, autopct='%1.1f%%')
        ax3.set_title('Data Completeness Overview')
        
        # 4. Missing data correlation matrix
        ax4 = axes[1, 1]
        # This would require the actual data to create a meaningful correlation matrix
        # For now, show a text summary
        ax4.text(0.1, 0.5, 
                f"Total Missing: {missing_data_metrics.get('missing_percentage', 0):.1f}%\n"
                f"Complete Cases: {missing_data_metrics.get('complete_cases_percentage', 0):.1f}%",
                transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Missing Data Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_distributions(self, distribution_metrics: Dict[str, Any], 
                          title: str = "Data Distributions") -> plt.Figure:
        """Plot data distributions."""
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        if 'message' in distribution_metrics:
            # No numeric data available
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, distribution_metrics['message'], 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Numeric Data Available')
            return fig
        
        # Get distribution data for first few features
        feature_data = []
        feature_names = []
        
        for feature, stats in distribution_metrics.items():
            if feature == 'overall' or not isinstance(stats, dict):
                continue
            
            if len(feature_names) >= 6:  # Limit to 6 features for readability
                break
            
            feature_names.append(feature)
            feature_data.append({
                'mean': stats.get('mean', 0),
                'std': stats.get('std', 0),
                'skewness': stats.get('skewness', 0),
                'kurtosis': stats.get('kurtosis', 0)
            })
        
        if not feature_data:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No feature distribution data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create distribution plots
        dist_df = pd.DataFrame(feature_data, index=feature_names)
        
        # 1. Mean and Std
        ax1 = plt.subplot(2, 2, 1)
        x_pos = range(len(feature_names))
        ax1.bar(x_pos, dist_df['mean'], yerr=dist_df['std'], capsize=5, color='skyblue', alpha=0.7)
        ax1.set_title('Mean ± Std')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        
        # 2. Skewness
        ax2 = plt.subplot(2, 2, 2)
        colors = ['red' if abs(x) > 2 else 'orange' if abs(x) > 1 else 'green' for x in dist_df['skewness']]
        ax2.bar(x_pos, dist_df['skewness'], color=colors, alpha=0.7)
        ax2.set_title('Skewness (normality)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        
        # 3. Kurtosis
        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(x_pos, dist_df['kurtosis'], color='lightgreen', alpha=0.7)
        ax3.set_title('Kurtosis (tail heaviness)')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(feature_names, rotation=45, ha='right')
        
        # 4. Distribution summary
        ax4 = plt.subplot(2, 2, 4)
        if 'overall' in distribution_metrics:
            overall = distribution_metrics['overall']
            summary_text = (
                f"Global Mean: {overall.get('global_mean', 'N/A'):.2f}\n"
                f"Global Std: {overall.get('global_std', 'N/A'):.2f}\n"
                f"Global Skewness: {overall.get('global_skewness', 'N/A'):.2f}\n"
                f"Global Kurtosis: {overall.get('global_kurtosis', 'N/A'):.2f}"
            )
            ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax4.set_title('Global Distribution Stats')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, correlation_metrics: Dict[str, Any], 
                               title: str = "Feature Correlations") -> plt.Figure:
        """Plot correlation heatmap."""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 1. Correlation summary
        ax1 = axes[0]
        
        if 'high_correlation_pairs' in correlation_metrics:
            high_corr = correlation_metrics['high_correlation_pairs']
            
            if high_corr:
                # Extract correlation values
                corr_values = [abs(pair['correlation']) for pair in high_corr[:20]]  # Top 20
                
                ax1.barh(range(len(corr_values)), corr_values, color='coral')
                ax1.set_title(f'High Correlations (|r| > 0.9)\nTotal: {len(high_corr)}')
                ax1.set_xlabel('Absolute Correlation')
                ax1.set_ylabel('Feature Pairs')
                
                # Add feature names as y-tick labels
                feature_labels = [f"{pair['feature1'][:15]}... vs {pair['feature2'][:15]}..." 
                                for pair in high_corr[:20]]
                ax1.set_yticks(range(len(corr_values)))
                ax1.set_yticklabels(feature_labels, fontsize=8)
            else:
                ax1.text(0.5, 0.5, "No high correlations found", 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('High Correlations')
        
        # 2. Correlation statistics
        ax2 = axes[1]
        
        stats_text = (
            f"High Correlations: {correlation_metrics.get('num_high_correlations', 0)}\n"
            f"Max Correlation: {correlation_metrics.get('max_correlation', 0):.3f}\n"
            f"Mean Correlation: {correlation_metrics.get('mean_correlation', 0):.3f}"
        )
        
        if 'condition_number' in correlation_metrics:
            stats_text += f"\nCondition Number: {correlation_metrics['condition_number']:.1f}"
        
        if 'eigenvalue_ratio' in correlation_metrics:
            stats_text += f"\nEigenvalue Ratio: {correlation_metrics['eigenvalue_ratio']:.1f}"
        
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_title('Correlation Statistics')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_outliers(self, outlier_metrics: Dict[str, Any], 
                     title: str = "Outlier Detection") -> plt.Figure:
        """Plot outlier detection results."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 1. Outlier percentage
        ax1 = axes[0, 0]
        outlier_pct = outlier_metrics.get('outlier_percentage', 0)
        method = outlier_metrics.get('method', 'unknown')
        
        # Create pie chart
        labels = ['Normal', 'Outliers']
        sizes = [100 - outlier_pct, outlier_pct]
        colors = ['lightblue', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Outlier Distribution\n(Method: {method})')
        
        # 2. Outlier detection method info
        ax2 = axes[0, 1]
        
        method_info = f"Method: {method}\n"
        if method == 'isolation_forest':
            method_info += f"Contamination: {outlier_metrics.get('contamination', 'unknown')}\n"
            method_info += f"Estimators: {outlier_metrics.get('n_estimators', 'unknown')}"
        elif method == 'iqr':
            method_info += "IQR Multiplier: 1.5"
        elif method == 'zscore':
            method_info += f"Threshold: {outlier_metrics.get('zscore_threshold', 'unknown')}"
        
        ax2.text(0.1, 0.5, method_info, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax2.set_title('Outlier Detection Method')
        ax2.axis('off')
        
        # 3. Outlier indices (if available)
        ax3 = axes[1, 0]
        outliers = outlier_metrics.get('outliers', [])
        
        if len(outliers) > 0:
            # Show distribution of outlier indices
            ax3.hist(outliers, bins=min(20, len(outliers)), color='coral', alpha=0.7)
            ax3.set_title(f'Outlier Index Distribution\n({len(outliers)} outliers)')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Frequency')
        else:
            ax3.text(0.5, 0.5, "No outliers detected", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Outlier Distribution')
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        
        summary_stats = (
            f"Total Outliers: {len(outliers)}\n"
            f"Outlier Percentage: {outlier_pct:.2f}%\n"
            f"Detection Method: {method}\n"
            f"Normal Samples: {100 - outlier_pct:.1f}%"
        )
        
        ax4.text(0.1, 0.5, summary_stats, transform=ax4.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax4.set_title('Outlier Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_qc_summary_dashboard(self, qc_results: Dict[str, Any]) -> plt.Figure:
        """Create comprehensive QC summary dashboard."""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Multi-Omics Quality Control Dashboard", fontsize=18, fontweight='bold')
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Overall quality status
        ax1 = fig.add_subplot(gs[0, :2])
        overall_quality = qc_results.get('overall_quality', 'UNKNOWN')
        
        quality_colors = {'PASS': 'green', 'FAIL': 'red', 'WARNING': 'orange'}
        color = quality_colors.get(overall_quality, 'gray')
        
        ax1.text(0.5, 0.5, f"Overall Quality: {overall_quality}", 
                ha='center', va='center', transform=ax1.transAxes, 
                fontsize=24, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))
        ax1.set_title('Overall Quality Status', fontsize=14)
        ax1.axis('off')
        
        # Failed tests
        ax2 = fig.add_subplot(gs[0, 2:])
        failed_tests = qc_results.get('failed_tests', [])
        
        if failed_tests:
            failed_text = "Failed Tests:\n" + "\n".join(f"• {test}" for test in failed_tests[:5])
            if len(failed_tests) > 5:
                failed_text += f"\n... and {len(failed_tests) - 5} more"
            ax2.text(0.1, 0.5, failed_text, transform=ax2.transAxes, fontsize=10,
                    color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.3))
        else:
            ax2.text(0.5, 0.5, "All tests passed", ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, color='green')
        
        ax2.set_title('Quality Issues', fontsize=14)
        ax2.axis('off')
        
        # Data type summary
        ax3 = fig.add_subplot(gs[1, :2])
        
        data_types = []
        sample_counts = []
        feature_counts = []
        
        for data_type, metrics in qc_results.get('metrics', {}).items():
            if isinstance(metrics, dict):
                data_types.append(data_type)
                
                # Extract sample and feature counts
                basic_metrics = metrics.get('basic', {})
                shape = basic_metrics.get('shape', (0, 0))
                sample_counts.append(shape[0])
                feature_counts.append(shape[1])
        
        if data_types:
            x = np.arange(len(data_types))
            width = 0.35
            
            ax3.bar(x - width/2, sample_counts, width, label='Samples', color='skyblue')
            ax3.bar(x + width/2, feature_counts, width, label='Features', color='lightcoral')
            
            ax3.set_xlabel('Data Type')
            ax3.set_ylabel('Count')
            ax3.set_title('Data Dimensions by Type')
            ax3.set_xticks(x)
            ax3.set_xticklabels(data_types, rotation=45, ha='right')
            ax3.legend()
        
        # Missing data summary
        ax4 = fig.add_subplot(gs[1, 2:])
        
        missing_percentages = []
        data_type_names = []
        
        for data_type, metrics in qc_results.get('metrics', {}).items():
            if isinstance(metrics, dict) and 'missing_data' in metrics:
                missing_pct = metrics['missing_data'].get('missing_percentage', 0)
                missing_percentages.append(missing_pct)
                data_type_names.append(data_type)
        
        if missing_percentages:
            colors = ['red' if pct > 20 else 'orange' if pct > 10 else 'green' 
                     for pct in missing_percentages]
            
            ax4.bar(data_type_names, missing_percentages, color=colors)
            ax4.set_ylabel('Missing Data %')
            ax4.set_title('Missing Data by Data Type')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add threshold lines
            ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
            ax4.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% threshold')
            ax4.legend()
        
        # Quality metrics heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Create quality matrix
        quality_data = []
        quality_index = []
        
        for data_type, metrics in qc_results.get('metrics', {}).items():
            if isinstance(metrics, dict):
                quality_row = []
                quality_index.append(data_type)
                
                # Extract various quality metrics
                if 'missing_data' in metrics:
                    quality_row.append(metrics['missing_data'].get('missing_percentage', 0))
                else:
                    quality_row.append(0)
                
                if 'outliers' in metrics and isinstance(metrics['outliers'], dict):
                    quality_row.append(metrics['outliers'].get('outlier_percentage', 0))
                else:
                    quality_row.append(0)
                
                if 'basic' in metrics:
                    quality_row.append(1 if metrics['basic'].get('duplicate_rows', 0) > 0 else 0)
                else:
                    quality_row.append(0)
                
                quality_data.append(quality_row)
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data, 
                                    index=quality_index,
                                    columns=['Missing %', 'Outlier %', 'Duplicates'])
            
            sns.heatmap(quality_df, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                       ax=ax5, cbar_kws={'label': 'Quality Issue Severity'})
            ax5.set_title('Quality Metrics Heatmap')
            ax5.set_ylabel('Data Type')
        
        # Summary statistics
        ax6 = fig.add_subplot(gs[2, 2:])
        
        total_tests = 0
        passed_tests = 0
        warnings = len(qc_results.get('warnings', []))
        
        for data_type, metrics in qc_results.get('metrics', {}).items():
            if isinstance(metrics, dict) and 'quality_status' in metrics:
                total_tests += 1
                if metrics['quality_status'] == 'PASS':
                    passed_tests += 1
        
        summary_text = (
            f"Quality Control Summary:\n\n"
            f"Total Data Types: {total_tests}\n"
            f"Passed: {passed_tests}\n"
            f"Failed: {total_tests - passed_tests}\n"
            f"Warnings: {warnings}\n\n"
            f"Pass Rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "Pass Rate: N/A"
        )
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax6.set_title('QC Summary Statistics')
        ax6.axis('off')
        
        return fig
    
    def plot_sample_alignment(self, alignment_results: Dict[str, Any], 
                            title: str = "Sample Alignment Results") -> plt.Figure:
        """Plot sample alignment results."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Extract alignment statistics
        stats = alignment_results.get('alignment_statistics', {})
        
        # 1. Alignment percentage by omics type
        ax1 = axes[0, 0]
        
        if 'alignment_percentage' in stats:
            omics_types = list(stats['alignment_percentage'].keys())
            percentages = list(stats['alignment_percentage'].values())
            
            colors = ['green' if pct > 80 else 'orange' if pct > 50 else 'red' 
                     for pct in percentages]
            
            bars = ax1.bar(omics_types, percentages, color=colors)
            ax1.set_ylabel('Alignment Percentage (%)')
            ax1.set_title('Sample Alignment by Data Type')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, pct in zip(bars, percentages):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        # 2. Sample counts comparison
        ax2 = axes[0, 1]
        
        if 'total_samples_per_omics' in stats and 'matched_samples' in stats:
            omics_types = list(stats['total_samples_per_omics'].keys())
            total_counts = list(stats['total_samples_per_omics'].values())
            matched_count = stats['matched_samples']
            
            x = np.arange(len(omics_types))
            width = 0.35
            
            ax2.bar(x - width/2, total_counts, width, label='Total Samples', color='lightblue')
            ax2.bar(x + width/2, [matched_count] * len(omics_types), width, 
                   label='Matched Samples', color='darkblue')
            
            ax2.set_xlabel('Data Type')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('Total vs Matched Samples')
            ax2.set_xticks(x)
            ax2.set_xticklabels(omics_types, rotation=45, ha='right')
            ax2.legend()
        
        # 3. Overlap matrix
        ax3 = axes[1, 0]
        
        if 'overlap_matrix' in stats:
            overlap_data = stats['overlap_matrix']
            
            if overlap_data:
                # Create overlap matrix
                omics_types = list(set([key.split('_vs_')[0] for key in overlap_data.keys()] + 
                                     [key.split('_vs_')[1] for key in overlap_data.keys()]))
                
                overlap_matrix = np.zeros((len(omics_types), len(omics_types)))
                
                for pair, overlap in overlap_data.items():
                    source, target = pair.split('_vs_')
                    i, j = omics_types.index(source), omics_types.index(target)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap  # Symmetric
                
                # Plot heatmap
                im = ax3.imshow(overlap_matrix, cmap='Blues', aspect='auto')
                ax3.set_xticks(range(len(omics_types)))
                ax3.set_yticks(range(len(omics_types)))
                ax3.set_xticklabels(omics_types, rotation=45, ha='right')
                ax3.set_yticklabels(omics_types)
                ax3.set_title('Sample Overlap Matrix')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label('Number of Overlapping Samples')
                
                # Add text annotations
                for i in range(len(omics_types)):
                    for j in range(len(omics_types)):
                        if overlap_matrix[i, j] > 0:
                            ax3.text(j, i, int(overlap_matrix[i, j]), 
                                   ha="center", va="center", color="black")
        
        # 4. Alignment method summary
        ax4 = axes[1, 1]
        
        method = alignment_results.get('method_used', 'unknown')
        fuzzy_matching = alignment_results.get('fuzzy_matching', False)
        tolerance = alignment_results.get('fuzzy_matching_config', {}).get('tolerance', 'N/A')
        
        method_info = (
            f"Alignment Method: {method}\n"
            f"Fuzzy Matching: {fuzzy_matching}\n"
            f"Tolerance: {tolerance}\n\n"
            f"Total Matched: {matched_count if 'matched_count' in locals() else 'N/A'}\n"
            f"Method Status: {'Success' if stats else 'Failed'}"
        )
        
        ax4.text(0.1, 0.5, method_info, transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax4.set_title('Alignment Method Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, plot_results: Dict[str, Any], output_dir: str, 
                   format: str = 'png') -> Dict[str, Any]:
        """Save plots to specified directory."""
        logger.info(f"Saving plots to: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for plot_name, plot_path in plot_results.get('plot_files', {}).items():
            try:
                # Copy or move plot files to output directory
                source_path = Path(plot_path)
                if source_path.exists():
                    dest_path = output_path / f"{plot_name}.{format}"
                    
                    # Convert format if needed
                    if format != self.save_format:
                        # Read and save in new format
                        import matplotlib.image as mpimg
                        img = mpimg.imread(source_path)
                        plt.imsave(dest_path, img)
                    else:
                        # Just copy
                        import shutil
                        shutil.copy2(source_path, dest_path)
                    
                    saved_files[plot_name] = str(dest_path)
            
            except Exception as e:
                logger.error(f"Failed to save plot {plot_name}: {e}")
                continue
        
        return {
            'saved_files': saved_files,
            'output_directory': str(output_path),
            'format': format
        }