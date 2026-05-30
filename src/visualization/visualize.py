import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display as display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("data/interim/metamotion_resampled.pkl")
df.head()

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))
#plt.show()

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    #plt.title(f"Accelerometer Y-axis for {label}")
    #plt.xlabel("Time Steps")
    #plt.ylabel("Acceleration (g)")
    plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    #plt.title(f"Accelerometer Y-axis for {label}")
    #plt.xlabel("Time Steps")
    #plt.ylabel("Acceleration (g)")
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = df.query("label == '{label}'").query(f"participant == '{participant}'").reset_index()
fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("Acceleration (g)")
ax.set_xlabel("Samples")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        if not all_axis_df.empty:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_title(f"{label} - Participant {participant}")
            ax.set_ylabel("Acceleration (g)")
            ax.set_xlabel("Samples")
            plt.legend()
            plt.show()


for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        if not all_axis_df.empty:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_title(f"{label} - Participant {participant}")
            ax.set_ylabel("Angular Velocity (deg/s)")
            ax.set_xlabel("Samples")
            plt.legend()
            plt.show()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_df = (
    df.query("label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True) 
    )

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
ax[0].legend(loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
#ax[1].set_ylabel("Angular Velocity (deg/s)")
ax[1].set_xlabel("Samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
            ax[0].legend(loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[0].set_ylabel("Acceleration (g)")
            ax[1].set_ylabel("Angular Velocity (deg/s)")
            ax[1].set_xlabel("Samples")
            plt.savefig(f"reports/figures/{label}_participant_{participant}.png")
            plt.show()