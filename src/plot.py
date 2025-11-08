# plot.py

import matplotlib.pyplot as plt
import math

def make_plot(mre_plot, y_pred_plot, y_true_plot, save_path=None):
    """
    Generates 3 plots:
      1. Histogram of relative test error
      2. Predictions vs. Labels (by sample index)
      3. Predictions vs. Labels (scatter with ±3% error lines)
    
    Args:
        mre_plot (list): Mean relative errors (%) for each sample
        y_pred_plot (list): Predicted output values
        y_true_plot (list): True output values
        save_path (str, optional): If provided, saves the plot image to this path instead of showing it.
    """
    print("Generating plots...")

    plt.figure(figsize=(15, 5))
    font = {'family': 'serif', 'color': 'darkred', 'size': 12}

    # 1️⃣ Histogram of relative errors
    plt.subplot(131)
    plt.hist(mre_plot, bins=math.ceil(max(mre_plot)))
    plt.title('Histogram of Relative Test Error (%)', fontdict=font)
    plt.xlabel('Error (%)')
    plt.ylabel('Count')

    # 2️⃣ Predictions vs. True Labels (sample index)
    plt.subplot(132)
    plt.plot(range(len(y_true_plot)), y_pred_plot, 'o', color='blue', label='Predictions')
    plt.plot(range(len(y_true_plot)), y_true_plot, 'o', color='red', alpha=0.5, label='Labels')
    plt.title('Predictions vs. Labels', fontdict=font)
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.legend()

    # 3️⃣ Scatter plot of Predictions vs. Labels (Error visualization)
    plt.subplot(133)
    plt.plot(y_true_plot, y_pred_plot, 'o', color='blue', alpha=0.5, label='Predictions')
    plt.plot(y_true_plot, y_true_plot, color='red', label='Perfect Prediction (y=x)')
    plt.plot(y_true_plot, [i * 1.03 for i in y_true_plot], '--', color='gray', label='+3% error')
    plt.plot(y_true_plot, [i * 0.97 for i in y_true_plot], '--', color='gray', label='-3% error')
    plt.title('Error Plot: Predictions vs. Labels', fontdict=font)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()