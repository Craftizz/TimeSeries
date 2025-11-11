import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.ticker import FuncFormatter


def millions(x, pos):

    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.1f}K"
    else:
        return f"{int(x)}"
    

def plot(decomposed_per_category: pd.DataFrame,
         save_directory: Path, 
         dpi: int = 300
    ) -> None:

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    groups = decomposed_per_category['Group'].unique()

    for group in groups:

        group_data = decomposed_per_category[
            decomposed_per_category['Group'] == group
        ].copy()

        mean_sales = group_data['Mean Sales Volume'].iloc[0]
        std_sales = group_data['Std Deviation'].iloc[0]
        cv_sales = group_data['Coeff of Variation'].iloc[0]
        num_low_days = group_data['Is Unusually Low'].sum()
        total_days = len(group_data)

        # Create figure with space for statistics
        fig = plt.figure(figsize=(20, 12), facecolor='#FAFAFA')
        fig.subplots_adjust(bottom=0.08)
        
        # Create grid: 4 rows for plots, 1 row for statistics
        gs = fig.add_gridspec(
            5, 1, 
            height_ratios=[1.2, 1, 1, 1, 0.35], 
            hspace=0.3,
            top=0.94,
            bottom=0.06,
            left=0.08,
            right=0.96
        )
        
        # Create axes for plots
        axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
        
        # Modern color palette
        colors = {
            'observed': '#0EA5E9',  
            'trend': '#8B5CF6', 
            'seasonal': '#F59E0B', 
            'residual': '#EF4444',
            'low_sales': '#DC2626', 
            'accent': '#10B981'
        }
        
        # Overall title with modern styling
        fig.suptitle(
            f"{group}" + (f" of Station {group_data['Station'].iloc[0]}" if group_data['Station'].any() else ""), 
            fontsize=18, 
            weight="bold",
            y=0.98,
            fontfamily='sans-serif'
        )
        
        # Plot Observed with unusually low days highlighted
        axes[0].plot(
            group_data.index, 
            group_data['Observed'], 
            color=colors['observed'], 
            linewidth=2.5,
            alpha=0.9,
            label='Observed'
        )

        axes[0].fill_between(
            group_data.index, 
            group_data['Observed'], 
            alpha=0.15, 
            color=colors['observed']
        )
        
        # Highlight unusually low days
        low_days = group_data[group_data['Is Unusually Low']]
        if not low_days.empty:
            axes[0].scatter(
                low_days['Date'],
                low_days['Observed'],
                color=colors['low_sales'],
                s=80,
                zorder=5,
                alpha=0.7,
                edgecolors='white',
                linewidth=1.5,
                label='Unusually Low',
                marker='.'
            )
        
        axes[0].set_ylabel("Sales Volume", fontsize=11, weight="bold")
        axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[0].yaxis.set_major_formatter(FuncFormatter(millions))
        axes[0].set_facecolor('#F8F9FA')
        if not low_days.empty:
            axes[0].legend(loc='upper right', framealpha=0.9, fontsize=9)

        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        # Plot Trend
        axes[1].plot(
            group_data.index, 
            group_data['Trend'], 
            color=colors['trend'], 
            linewidth=3,
            alpha=0.9
        )
        axes[1].fill_between(
            group_data.index, 
            group_data['Trend'], 
            alpha=0.15, 
            color=colors['trend']
        )
        axes[1].set_ylabel("Trend", fontsize=11, weight="bold")
        axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[1].yaxis.set_major_formatter(FuncFormatter(millions))
        axes[1].set_facecolor('#F8F9FA')

        # Plot Seasonal
        axes[2].plot(
            group_data.index, 
            group_data['Seasonal'], 
            color=colors['seasonal'], 
            linewidth=2.5,
            alpha=0.9
        )
        axes[2].fill_between(
            group_data.index, 
            group_data['Seasonal'], 
            alpha=0.15, 
            color=colors['seasonal']
        )
        axes[2].axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        axes[2].set_ylabel("Seasonal", fontsize=11, weight="bold")
        axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[2].yaxis.set_major_formatter(FuncFormatter(millions))
        axes[2].set_facecolor('#F8F9FA')

        
        # Plot Residual
        axes[3].scatter(
            group_data.index, 
            group_data['Residual'], 
            color=colors['residual'], 
            s=30,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
        axes[3].axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.6)
        axes[3].set_ylabel("Residual", fontsize=11, weight="bold")
        axes[3].set_xlabel("Date", fontsize=11, weight="bold")
        axes[3].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[3].yaxis.set_major_formatter(FuncFormatter(millions))
        axes[3].set_facecolor('#F8F9FA')
        
        # Add subtle spine styling and modify axes
        for ax in axes:
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_edgecolor('#E5E7EB')
                ax.spines[spine].set_linewidth(1.5)

            ax.tick_params(colors='#6B7280', labelsize=9)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        # Add statistics text box at the bottom
        stats_ax = fig.add_subplot(gs[4, 0])
        stats_ax.axis('off')
        
        # Format statistics text
        stats_text = (
            f"Mean Sales Volume: {mean_sales:,.0f}\n"
            f"Standard Deviation: {std_sales:,.0f}\n"
            f"Coefficient of Variation: {cv_sales:.2%}\n"
            f"Unusually Low Days: {num_low_days}/{total_days} ({num_low_days/total_days:.1%})"
        )
        
        # Create a box for statistics
        stats_ax.text(
            0.02, 0.85,
            stats_text,
            transform=stats_ax.transAxes,
            fontsize=8, 
            verticalalignment='top',
            horizontalalignment='left',
            wrap=True,
            color='#333333',
            fontfamily='sans-serif',
            fontweight='medium',
            alpha=0.9,
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='none',
                edgecolor='none',
                alpha=0.0
            )
        )
        
        # Save chart
        filename = f"{group.replace(' ', '_').replace('/', '-')}.png"
        filepath = save_directory / filename
        plt.savefig(
            filepath, 
            dpi=dpi, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Saved: {filepath}")

        plt.close(fig)
    









