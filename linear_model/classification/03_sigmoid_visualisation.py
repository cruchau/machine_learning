import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

class LogisticRegressionVisualizer:
    def __init__(self):
        # Generate sample data
        np.random.seed(42)
        n_samples = 100
        
        # Create two classes with some overlap
        class_0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
        class_1 = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], n_samples//2)
        
        self.X = np.vstack([class_0, class_1])
        self.y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Initial parameters
        self.w0 = -2.0  # bias
        self.w1 = 0.5   # weight for x1
        self.w2 = 0.3   # weight for x2
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # Create sliders
        self.create_sliders()
        
        # Initial plot
        self.update_plots()
        
    def create_sliders(self):
        # Create slider axes
        ax_w0 = plt.axes([0.2, 0.1, 0.25, 0.03])
        ax_w1 = plt.axes([0.2, 0.06, 0.25, 0.03])
        ax_w2 = plt.axes([0.2, 0.02, 0.25, 0.03])
        
        # Create sliders
        self.slider_w0 = Slider(ax_w0, 'Bias (w0)', -5.0, 5.0, valinit=self.w0, valfmt='%.2f')
        self.slider_w1 = Slider(ax_w1, 'Weight 1 (w1)', -2.0, 2.0, valinit=self.w1, valfmt='%.2f')
        self.slider_w2 = Slider(ax_w2, 'Weight 2 (w2)', -2.0, 2.0, valinit=self.w2, valfmt='%.2f')
        
        # Connect sliders to update function
        self.slider_w0.on_changed(self.update_parameters)
        self.slider_w1.on_changed(self.update_parameters)
        self.slider_w2.on_changed(self.update_parameters)
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        """Calculate predicted probabilities"""
        z = self.w0 + self.w1 * X[:, 0] + self.w2 * X[:, 1]
        return self.sigmoid(z)
    
    def update_parameters(self, val):
        """Update parameters from sliders"""
        self.w0 = self.slider_w0.val
        self.w1 = self.slider_w1.val
        self.w2 = self.slider_w2.val
        self.update_plots()
        
    def update_plots(self):
        """Update both plots"""
        self.ax1.clear()
        self.ax2.clear()
        
        self.plot_decision_boundary()
        self.plot_sigmoid_function()
        
        plt.draw()
        
    def plot_decision_boundary(self):
        """Plot scatter points and decision boundary"""
        # Plot data points
        class_0_mask = self.y == 0
        class_1_mask = self.y == 1
        
        self.ax1.scatter(self.X[class_0_mask, 0], self.X[class_0_mask, 1], 
                        c='red', marker='o', s=50, alpha=0.7, label='Class 0')
        self.ax1.scatter(self.X[class_1_mask, 0], self.X[class_1_mask, 1], 
                        c='blue', marker='s', s=50, alpha=0.7, label='Class 1')
        
        # Create decision boundary
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        
        # Create meshgrid for decision boundary
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Calculate probabilities for each point in the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        probas = self.predict_proba(grid_points)
        probas = probas.reshape(xx.shape)
        
        # Plot decision boundary (probability = 0.5)
        self.ax1.contour(xx, yy, probas, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Plot probability contours
        cs = self.ax1.contourf(xx, yy, probas, levels=20, alpha=0.3, cmap='RdYlBu')
        
        # Add colorbar
        if not hasattr(self, 'cbar1'):
            self.cbar1 = plt.colorbar(cs, ax=self.ax1)
            self.cbar1.set_label('Probability of Class 1')
        
        self.ax1.set_xlabel('Feature 1 (x1)')
        self.ax1.set_ylabel('Feature 2 (x2)')
        self.ax1.set_title('Decision Boundary and Data Points')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
    def plot_sigmoid_function(self):
        """Plot sigmoid function and projected points"""
        # Calculate z values for current data points
        z_data = self.w0 + self.w1 * self.X[:, 0] + self.w2 * self.X[:, 1]
        
        # Create z range for sigmoid curve
        z_min, z_max = min(z_data.min() - 2, -5), max(z_data.max() + 2, 5)
        z_range = np.linspace(z_min, z_max, 1000)
        sigmoid_values = self.sigmoid(z_range)
        
        # Plot sigmoid curve
        self.ax2.plot(z_range, sigmoid_values, 'g-', linewidth=2, label='Sigmoid Function')
        
        # Plot data points projected onto sigmoid
        class_0_mask = self.y == 0
        class_1_mask = self.y == 1
        
        # Plot class 0 points
        self.ax2.scatter(z_data[class_0_mask], np.zeros(np.sum(class_0_mask)), 
                        c='red', marker='o', s=50, alpha=0.7, label='Class 0 (y=0)')
        
        # Plot class 1 points
        self.ax2.scatter(z_data[class_1_mask], np.ones(np.sum(class_1_mask)), 
                        c='blue', marker='s', s=50, alpha=0.7, label='Class 1 (y=1)')
        
        # Plot predicted probabilities for each point
        probas = self.sigmoid(z_data)
        self.ax2.scatter(z_data, probas, c='purple', marker='x', s=30, alpha=0.7, 
                        label='Predicted Probabilities')
        
        # Add decision threshold line
        self.ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
        self.ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        # Add equation text
        equation = f'σ(z) = σ({self.w0:.2f} + {self.w1:.2f}x₁ + {self.w2:.2f}x₂)'
        self.ax2.text(0.02, 0.98, equation, transform=self.ax2.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax2.set_xlabel('z = w₀ + w₁x₁ + w₂x₂')
        self.ax2.set_ylabel('σ(z) = P(y=1|x)')
        self.ax2.set_title('Sigmoid Function and Data Points')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(-0.1, 1.1)
        
    def show(self):
        """Display the interactive plot"""
        plt.tight_layout()
        plt.show()

# Create and run the visualizer
if __name__ == "__main__":
    visualizer = LogisticRegressionVisualizer()
    visualizer.show()