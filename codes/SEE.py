import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def ecdf(values):
    x = np.sort(values)
    n = len(values)
    # y will go from 1/n up to 1.0
    y = np.arange(1, n+1) / n
    return x, y


def See(arg1, tidy, random_state=1234):
    # Filter to rows matching the given ATC code
    df = tidy[tidy["ATC"] == arg1].copy()
    
    # Sort by pnr, eksd and create a prev_eksd column (like dplyr::lag)
    df = df.sort_values(["pnr", "eksd"]).reset_index(drop=True)
    df["prev_eksd"] = df.groupby("pnr")["eksd"].shift(1)
    
    # Remove rows where 'prev_eksd' is NaN
    df = df.dropna(subset=["prev_eksd"]).copy()

    # If a pnr has only 1 row after the shift, that row is chosen. 
    # If multiple, we pick exactly 1 at random.
    df_sampled = (df.groupby("pnr", group_keys=False)
                    .apply(lambda x: x.sample(1, random_state=random_state))
                  ).reset_index(drop=True)

    # Calculate event.interval = eksd - prev_eksd, store as numeric days
    df_sampled["event.interval"] = (
        df_sampled["eksd"] - df_sampled["prev_eksd"]
    ).dt.days.astype(float)

    xvals, yvals = ecdf(df_sampled["event.interval"].values)
    
    # Combine x and y into a DataFrame to replicate 'dfper'
    dfper = pd.DataFrame({
        "x": xvals,
        "y": yvals
    })
    
    # Keep only the portion where y <= 0.8 (i.e., keep the bottom 80% of intervals)
    dfper_80 = dfper[dfper["y"] <= 0.8].copy()
    
    # The largest x in the 80% portion
    ni = dfper_80["x"].max()
    
    # Subset original sample to those whose event.interval <= ni
    df_sub = df_sampled[df_sampled["event.interval"] <= ni].copy()

    log_intervals = np.log(df_sub["event.interval"].values).reshape(-1, 1)
    
    range_k = range(2, 7)  # or 2..10, your choice
    best_k = None
    best_score = -1
    
    for k in range_k:
        kmeans_model = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans_model.fit_predict(log_intervals)
        sil_score = silhouette_score(log_intervals, labels)
        if sil_score > best_score:
            best_score = sil_score
            best_k = k

    # Now run final KMeans with the best k
    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state).fit(dfper_80[["x"]])
    dfper_80["cluster"] = final_kmeans.labels_ + 1  # Make cluster 1-based if desired
    dfper_80["log_x"] = np.log(dfper_80["x"])
    
    cluster_stats = (
        dfper_80
        .groupby("cluster")["log_x"]
        .agg(["min", "max", "median"])
        .reset_index()
        .rename(columns={"min":"log_min", "max":"log_max", "median":"log_median"})
    )

    # Exponentiate
    cluster_stats["Minimum"] = np.exp(cluster_stats["log_min"])
    cluster_stats["Maximum"] = np.exp(cluster_stats["log_max"])
    cluster_stats["Median"] = np.exp(cluster_stats["log_median"])
    
    # We'll keep only [cluster, Minimum, Maximum, Median]
    cluster_stats = cluster_stats[["cluster", "Minimum", "Maximum", "Median"]].copy()

    df_merge = df_sampled.merge(cluster_stats, how="cross")
    
    # Keep the cluster whose range includes the row's event.interval
    df_merge = df_merge[
        (df_merge["event.interval"] >= df_merge["Minimum"]) &
        (df_merge["event.interval"] <= df_merge["Maximum"])
    ].copy()
    
    # We only keep relevant columns
    df_merge = df_merge[["pnr", "Median", "cluster"]].drop_duplicates()
    
    # If multiple clusters match for one pnr, we pick the first arbitrarily.
    df_merge = df_merge.drop_duplicates(subset=["pnr"], keep="first")

    cluster_counts = df_merge["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["cluster", "Freq"]
    top_cluster = cluster_counts["cluster"].iloc[0]
    
    # The single row for top_cluster, its median
    top_row = cluster_stats[cluster_stats["cluster"] == top_cluster].iloc[0]
    top_median = top_row["Median"]
    
    df_sampled = df_sampled.merge(df_merge, on="pnr", how="left")
    df_sampled["Median"] = df_sampled["Median"].fillna(top_median)
    df_sampled["cluster"] = df_sampled["cluster"].fillna(0).astype(int)

    # Create a test column = event.interval - median (just like the R code)
    df_sampled["test"] = (df_sampled["event.interval"] - df_sampled["Median"]).round(1)
    
    keep_cols = ["pnr", "Median", "cluster"]
    df_return = df.merge(df_sampled[keep_cols].drop_duplicates("pnr"), on="pnr", how="left")
    
    df_return["Median"] = df_return["Median"].fillna(top_median)
    df_return["cluster"] = df_return["cluster"].fillna(0).astype(int)

    # df_return is analogous to "Drug_see_p0" from R
    return df_return

def see_assumption(df):
    # Sort, create prev_eksd by group
    df = df.sort_values(["pnr", "eksd"]).reset_index(drop=True)
    df["prev_eksd"] = df.groupby("pnr")["eksd"].shift(1)

    # p_number is 1-based index within each pnr
    df["p_number"] = df.groupby("pnr")["eksd"].cumcount() + 1
    
    # Keep rows where p_number >= 2
    df2 = df[df["p_number"] >= 2].copy()
    
    # Duration = eksd - prev_eksd
    df2["Duration"] = (df2["eksd"] - df2["prev_eksd"]).dt.days

    # Compute median-of-medians
    medians_of_medians = df2.groupby("pnr")["Duration"].median()
    overall_median = medians_of_medians.median()  # median of pnr-medians
    
    # Make the boxplot
    fig, ax = plt.subplots(figsize=(7,5))
    sns.boxplot(
        data=df2, 
        x="p_number", 
        y="Duration",
        ax=ax
    )
    
    # Add horizontal line for median-of-medians
    ax.axhline(overall_median, ls='--', color='red')
    ax.set_title("Duration Boxplot by Prescription Number")
    
    return fig, ax

if __name__ == "__main__":
    medA_df = See("medA", tidy=your_tidy_df)
    medB_df = See("medB", tidy=your_tidy_df)

    figA, axA = see_assumption(medA_df)
    plt.show()   # Show the boxplot for medA

    figB, axB = see_assumption(medB_df)
    plt.show()   # Show the boxplot for medB
