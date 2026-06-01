import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("data/interim/metamotion_resampled.pkl")
#df.head()
#print(df.shape)
outliers = list(df.columns[:6])  # Select only the sensor columns for outlier detection
# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use("seaborn-v0_8-deep") #"fivethirtyeight", "ggplot", "seaborn-v0_8-deep"
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

#df[["acc_x", "label"]].boxplot(by="label", figsize=(20, 10), rot=45)
df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))

# 1. Strip whitespaces from all column names to catch hidden trailing spaces
df.columns = [str(col).strip() for col in df.columns]

# 2. Look for any auto-renamed duplicates like 'label.1' or 'label_x' 
duplicate_label_cols = [col for col in df.columns if "label" in col.lower() and col != "label"]
df = df.drop(columns=duplicate_label_cols)

# 3. If the main 'label' column is still a duplicate DataFrame, force select the first one
if isinstance(df["label"], pd.DataFrame):
    clean_labels = df["label"].iloc[:, 0]
    df = df.drop(columns=["label"])
    df["label"] = clean_labels

# 4. Grab ALL 6 sensor columns from your outliers list safely
plot_cols = [col for col in outliers if col in df.columns and col != "label"]

# 5. Run the boxplots safely inside a clean, guaranteed 1-D isolated dataframe
if "label" in df.columns and len(plot_cols) > 0:
    # Build a clean temporary dataframe containing only unique target columns
    plot_dataframe = df[plot_cols].copy()
    plot_dataframe["label"] = df["label"].astype(str)
    
    # Plot the first 3 columns (Accelerometer: acc_x, acc_y, acc_z)
    acc_cols = [c for c in plot_cols[:3] if c in plot_dataframe.columns]
    if acc_cols:
        plot_dataframe[acc_cols + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, len(acc_cols)))
        plt.title("Accelerometer Outliers grouped by Exercise Label")
    
    # Plot the next 3 columns (Gyroscope: gyr_x, gyr_y, gyr_z)
    gyr_cols = [c for c in plot_cols[3:] if c in plot_dataframe.columns]
    if gyr_cols:
        plt.figure() # Creates a fresh window so they don't draw over each other
        plot_dataframe[gyr_cols + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, len(gyr_cols)))
        plt.title("Gyroscope Outliers grouped by Exercise Label")
        
    plt.show() # Renders both crisp multi-plot figures interaction boxes!
else:
    print("⚠️ Missing column targets or label tracking series.")

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column

col = "acc_x"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

# Loop over all columns

for col in df.columns:
    # 1. Skip metadata text columns explicitly
    if col in ["label", "participant", "category", "set"]:
        continue
        
    # 2. Double-check that the column data type is actually numeric (float or int)
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Skipping non-numeric column: {col}")
        continue
        
    # 3. If it passes the checks, it's a sensor column! Calculate outliers safely:
    print(f"Calculating IQR outliers for sensor column: {col}")
    df = mark_outliers_iqr(df, col)
    plot_binary_outliers(df, col, col + "_outlier", reset_index=True)



# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution

# --- STEP 1: AUTO-DETECT REAL NUMERIC COLUMNS ---
# Explicitly find all columns that contain numeric data types (floats/ints)
all_numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# Filter those numeric columns into pure accelerometer vs gyroscope lists
# (Excludes boolean outlier flags like 'acc_x_outlier' and metadata counts like 'set')
acc_hist_cols = [c for c in all_numeric_cols if "acc_" in c and "_outlier" not in c and c != "set"]
gyr_hist_cols = [c for c in all_numeric_cols if "gyr_" in c and "_outlier" not in c and c != "set"]

# --- STEP 2: PLOT ACCELEROMETER HISTOGRAMS SAFELY ---
if acc_hist_cols and "label" in df.columns:
    acc_plot_df = df[acc_hist_cols].copy()
    acc_plot_df["label"] = df["label"].astype(str)
    
    acc_plot_df.plot.hist(
        by="label", 
        bins=50, 
        figsize=(20, 15), 
        # 📂 layout line removed here to let matplotlib scale it auto
        sharex=False
    )
    plt.suptitle("Accelerometer Frequency Distributions by Exercise Type")

# --- STEP 3: PLOT GYROSCOPE HISTOGRAMS SAFELY ---
if gyr_hist_cols and "label" in df.columns:
    gyr_plot_df = df[gyr_hist_cols].copy()
    gyr_plot_df["label"] = df["label"].astype(str)
    
    plt.figure() 
    gyr_plot_df.plot.hist(
        by="label", 
        bins=50, 
        figsize=(20, 15), 
        # 📂 layout line removed here to let matplotlib scale it auto
        sharex=False
    )
    plt.suptitle("Gyroscope Frequency Distributions by Exercise Type")

plt.show()

# Insert Chauvenet's function

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Loop over all columns

for col in df.columns:
    # 1. Skip metadata text columns explicitly
    if col in ["label", "participant", "category", "set"]:
        continue
        
    # 2. Double-check that the column data type is actually numeric (float or int)
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Skipping non-numeric column: {col}")
        continue
        
    # 3. If it passes the checks, it's a sensor column! Calculate outliers safely:
    print(f"Calculating Chauvenet outliers for sensor column: {col}")
    df = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(df, col, col + "_outlier", reset_index=True)

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# Loop over all columns

dataset, outliers, X_scores = mark_outliers_lof(df, acc_hist_cols + gyr_hist_cols, n=20)

for col in df.columns:
    # 1. Skip metadata text columns explicitly
    if col in ["label", "participant", "category", "set"]:
        continue
        
    # 2. Double-check that the column data type is actually numeric (float or int)
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Skipping non-numeric column: {col}")
        continue
    
    plot_binary_outliers(df, col, col + "_outlier", reset_index=True)

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column


# Create a loop

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------