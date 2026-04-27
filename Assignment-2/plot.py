import glob
import pandas as pd
import matplotlib.pyplot as plt
# loading files
files = sorted(glob.glob("results/times_P*.csv"))
dfs = []
for f in files:
    df = pd.read_csv(f)
    df = df[df["P"] != "P"]
    for col in ["P", "M", "run", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
Ps = sorted(data["P"].unique())
Ms = sorted(data["M"].unique())
base_positions = list(range(len(Ps)))
#values for plotting
offsets = [-0.18, 0.18]   
width = 0.3

fig, ax = plt.subplots(figsize=(8, 5))
for j, M in enumerate(Ms):
    box_data = []
    positions = []

    for i, P in enumerate(Ps):
        vals = data[(data["P"] == P) & (data["M"] == M)]["time"].values
        box_data.append(vals)
        positions.append(base_positions[i] + offsets[j])

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
        patch_artist=True,
        labels=None,
    	showfliers = False,
	whis=(0,100)
	)
    color = "lightblue" if M == 120 else "lightgreen"
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
# final plotting
ax.set_xticks(base_positions)
ax.set_xticklabels([str(P) for P in Ps])
ax.set_xlabel("Processes (P)")
ax.set_ylabel("Time (seconds)")
ax.set_title("Execution Time Boxplots")
# making two plots on same graph
ax.plot([], [], color="lightblue", linewidth=8, label="M = 120")
ax.plot([], [], color="lightgreen", linewidth=8, label="M = 240")
ax.legend()
plt.tight_layout()
plt.savefig("boxplot_times.png", dpi=300)
plt.show()
