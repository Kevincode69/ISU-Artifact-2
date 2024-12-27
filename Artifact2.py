import tkinter as tk 
from tkinter import ttk, messagebox, simpledialog 
import numpy as np 
import matplotlib.pyplot as plt 
import statistics as stat 
from scipy.stats import linregress 

textFont = "Verdana" 

def calculate_statistics(data):
    mean = np.mean(data) 
    median = np.median(data) 
    mode = stat.multimode(data) 

    if len(mode) == len(data): 
        mode = "There is no mode." 
    elif len(mode) == 1: 
        mode = stat.mode(data) 
    else: 
        mode = ", ".join(map(str, mode)) 
    
    std_dev = np.std(data, ddof=1) 
    q1 = np.percentile(data, 25) 
    q3 = np.percentile(data, 75) 
    iqr = q3 - q1 
    lower_bound = q1 - 1.5 * iqr 
    upper_bound = q3 + 1.5 * iqr 
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return { 
        "Mean:": round(mean, 2), 
        "Median:": round(median, 2), 
        "Mode:": mode, 
        "Standard Deviation:": round(std_dev, 2), 
        "Q1:": round(q1, 2), 
        "Q3:": round(q3, 2), 
        "IQR:": round(iqr, 2), 
        "Outliers:": outliers, 
    } 

def plot_box_and_whisker(data, remove_outliers=False): 
    x_label = simpledialog.askstring("Customize X-axis", "Enter label for the X-axis:")
    
    if remove_outliers:
        q1 = np.percentile(data, 25) 
        q3 = np.percentile(data, 75)
        iqr = q3 - q1 
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr 
        data = [x for x in data if lower_bound <= x <= upper_bound]

    plt.figure(figsize=(6, 4)) 
    plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="cyan")) 
    plt.title("Box and Whisker Plot") 
    plt.xlabel(x_label if x_label else "Values") 
    plt.show() 

def plot_histogram(data): 
    x_label = simpledialog.askstring("Customize X-axis", "Enter label for the X-axis:") 
    y_label = simpledialog.askstring("Customize Y-axis", "Enter label for the Y-axis:") 

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins="auto", color="red", edgecolor="green", alpha=0.7) 
    plt.title("Histogram") 
    plt.xlabel(x_label if x_label else "Values") 
    plt.ylabel(y_label if y_label else "Frequency") 
    plt.show() 

def get_single_variable_dataset(): 
    try:
        dataset_str = simpledialog.askstring("Input Dataset", "Enter numbers separated by spaces:") 
        if not dataset_str:
            messagebox.showerror("Error", "Dataset cannot be empty.") 
            return None 
        dataset = [float(x.strip()) for x in dataset_str.split()] 
        return dataset 
    except ValueError: 
        messagebox.showerror("Error", "Invalid input. Please enter only numbers separated by spaces.") 
        return None 

def get_two_variable_dataset(): 
    try:
        x_values = simpledialog.askstring("Input X Values", "Enter X values separated by spaces:")
        y_values = simpledialog.askstring("Input Y Values", "Enter Y values separated by spaces:") 
        if not x_values or not y_values: 
            messagebox.showerror("Error", "Both X and Y datasets must be provided.") 
            return None, None

        x = [float(x.strip()) for x in x_values.split()] 
        y = [float(y.strip()) for y in y_values.split()] 
        
        if len(x) != len(y): 
            messagebox.showerror("Error", "X and Y datasets must have the same number of values.") 
            return None, None 
        
        return x, y 
    except ValueError: 
        messagebox.showerror("Error", "Invalid input. Please enter only numbers separated by spaces.") 
        return None, None 

def single_variable_analysis():
    data = get_single_variable_dataset()
    if data is None: 
        return 

    data.sort() 
    messagebox.showinfo("Sorted Data", f"Sorted Data: {data}") 

    stats_results = calculate_statistics(data) 
    stats_message = "\n".join(f"{key}: {value}" for key, value in stats_results.items()) 
    messagebox.showinfo("Statistics", stats_message) 

    show_outliers = True 

    def toggle_outliers(): 
        nonlocal show_outliers
        show_outliers = not show_outliers 
        btn_toggle_outliers.config(
            text="Show box plot with outliers" if not show_outliers else "Show box plot without outliers"
        ) 
        plot_box_and_whisker(data, remove_outliers=not show_outliers) 

    root = tk.Toplevel() 
    root.title("Graphs") 

    btn_histogram = tk.Button(root, text="Show histogram", command=lambda: plot_histogram(data))
    btn_histogram.pack(pady=5)

    btn_toggle_outliers = tk.Button(
        root, text="Show box plot without outliers", command=toggle_outliers
    ) 
    btn_toggle_outliers.pack(pady=5) 

    btn_close = tk.Button(root, text="Close", command=root.destroy) 
    btn_close.pack(pady=5) 

def calculate_outliers(x, y): 
    slope, intercept, _, _, _ = linregress(x, y) 
    residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)] 
    q1 = np.percentile(residuals, 25) 
    q3 = np.percentile(residuals, 75) 
    iqr = q3 - q1 
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr 
    outliers = [(x[i], y[i]) for i, res in enumerate(residuals) if res < lower_bound or res > upper_bound] 
    return outliers 

def plot_scatter_with_outliers(x, y, remove_outliers=False): 
    if remove_outliers: 
        slope, intercept, _, _, _ = linregress(x, y) 
        residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
        q1 = np.percentile(residuals, 25)
        q3 = np.percentile(residuals, 75) 
        iqr = q3 - q1 
        lower_bound = q1 - 1.5 * iqr 
        upper_bound = q3 + 1.5 * iqr 
        filtered_indices = [i for i, res in enumerate(residuals) if lower_bound <= res <= upper_bound] 
        x = [x[i] for i in filtered_indices]
        y = [y[i] for i in filtered_indices] 

    x_label = simpledialog.askstring("Customize X-axis", "Enter label for the X-axis:") 
    y_label = simpledialog.askstring("Customize Y-axis", "Enter label for the Y-axis:")

    slope, intercept, r_value, _, _ = linregress(x, y) 
    regression_line = [slope * xi + intercept for xi in x] 

    plt.figure(figsize=(6, 4)) 
    plt.scatter(x, y, color="red", edgecolor="green", label="Data Points") 
    plt.plot(x, regression_line, color="blue", label=f"y = {slope:.2f}x + {intercept:.2f}\nCorrelation Coefficient (r) = {r_value:.2f}") 
    plt.title("Scatter Plot with Regression Line") 
    plt.xlabel(x_label if x_label else "X Values") 
    plt.ylabel(y_label if y_label else "Y Values") 
    plt.legend() 
    plt.show()

def two_variable_analysis(): 
    x, y = get_two_variable_dataset()
    if x is None or y is None: 
        return 

    outliers = calculate_outliers(x, y) 
    if outliers: 
        outlier_message = f"Outliers detected:\n{outliers}" 
    else: 
        outlier_message = "No outliers detected."
    messagebox.showinfo("Outliers", outlier_message) 

    root = tk.Toplevel() 
    root.title("Scatter plot options") 

    def show_with_outliers(): 
        plot_scatter_with_outliers(x, y, remove_outliers=False)

    def show_without_outliers(): 
        plot_scatter_with_outliers(x, y, remove_outliers=True)

    btn_with_outliers = tk.Button(root, text="Show scatter plot with outliers", command=show_with_outliers)
    btn_with_outliers.pack(pady=5) 

    btn_without_outliers = tk.Button(root, text="Show scatter plot without outliers", command=show_without_outliers) 
    btn_without_outliers.pack(pady=5) 

    btn_close = tk.Button(root, text="Close", command=root.destroy) 
    btn_close.pack(pady=5) 

def main(): 
    root = tk.Tk()
    root.title("ISU ARTIFACT 2") 

    label = tk.Label(root, text="ISU ARTIFACT 2", font=(textFont, 16)) 
    label.pack(pady=10) 

    instruction = tk.Label(root, text="Choose the type of data", font=(textFont, 12)) 
    instruction.pack(pady=5) 

    btn_single_var = tk.Button(root, text="Single Variable Dataset (Box plot, Histogram)", font=(textFont, 12), command=single_variable_analysis)
    btn_single_var.pack(pady=10) 

    btn_two_var = tk.Button(root, text="Two Variable Dataset (Scatter plot)", font=(textFont, 12), command=two_variable_analysis)
    btn_two_var.pack(pady=10) 

    btn_exit = tk.Button(root, text="Exit", font=(textFont, 12), command=root.quit) 
    btn_exit.pack(pady=10) 

    root.mainloop() 

if __name__ == "__main__":
    main()
