import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import cumfreq, gaussian_kde


# 1. Data Loading
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


# 2. ECDF Computation
def ecdf(data):
    data = np.asarray(data)
    res = cumfreq(data, numbins=len(data))
    x = res.lowerlimit + np.linspace(0, res.binsize * len(res.cumcount), len(res.cumcount))
    y = res.cumcount / len(data)
    return x, y


# 3. SEE Estimator Using K-Means
def see_estimator_kmeans(category: str, df: pd.DataFrame, 
                         k_range=range(2, 11)) -> pd.DataFrame:
    # Filter data
    df_cat = df[df['CATEGORY'] == category].copy()
    df_cat['DATE'] = pd.to_datetime(df_cat['DATE'], format='%m/%d/%Y')
    df_cat.sort_values(by=['PATIENT_ID', 'DATE'], inplace=True)
    df_cat['prev_DATE'] = df_cat.groupby('PATIENT_ID')['DATE'].shift(1)
    df_sample = df_cat.dropna(subset=['prev_DATE']).copy()

    # Randomly sample one consecutive pair per patient
    df_sample = df_sample.groupby('PATIENT_ID').apply(lambda x: x.sample(1, random_state=1234)).reset_index(drop=True)
    df_sample = df_sample[['PATIENT_ID', 'DATE', 'prev_DATE']].copy()
    df_sample['event_interval'] = (df_sample['DATE'] - df_sample['prev_DATE']).dt.days.astype(float)

    # Compute ECDF
    x_ecdf, y_ecdf = ecdf(df_sample['event_interval'].values)
    df_ecdf = pd.DataFrame({'x': x_ecdf, 'y': y_ecdf})

    # Trim to lower 80% of ECDF
    df_ecdf_80 = df_ecdf[df_ecdf['y'] <= 0.8]
    ni = df_ecdf_80['x'].max()
    df_filtered = df_sample[df_sample['event_interval'] <= ni].copy()

    # Density on log intervals
    log_intervals = np.log(df_filtered['event_interval'])
    kde = gaussian_kde(log_intervals)
    x1 = np.linspace(log_intervals.min(), log_intervals.max(), 100)
    y1 = kde(x1)

    # Silhouette analysis to find best_k
    X = df_ecdf[['x']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = 2
    best_score = -1
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=1234)
        labels = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Final K-Means with best_k
    kmeans_final = KMeans(n_clusters=best_k, random_state=1234)
    df_ecdf['cluster'] = kmeans_final.fit_predict(X_scaled)

    # Compute cluster boundaries on log(x)
    cluster_stats = df_ecdf.groupby('cluster')['x'].agg(
        min_log=lambda s: np.log(s).min(),
        max_log=lambda s: np.log(s).max(),
        median_log=lambda s: np.log(s).median()
    ).reset_index()
    cluster_stats['Minimum'] = np.exp(cluster_stats['min_log'])
    cluster_stats['Maximum'] = np.exp(cluster_stats['max_log'])
    cluster_stats['Median'] = np.exp(cluster_stats['median_log'])
    cluster_stats = cluster_stats[cluster_stats['Median'] > 0]

    # Assign cluster to each sampled pair
    def assign_cluster(interval):
        for _, row in cluster_stats.iterrows():
            if interval >= row['Minimum'] and interval <= row['Maximum']:
                return row['cluster']
        return np.nan

    df_sample['Final_cluster'] = df_sample['event_interval'].apply(assign_cluster)
    df_sample = df_sample.dropna(subset=['Final_cluster']).copy()
    df_sample = df_sample.merge(cluster_stats[['cluster', 'Median']], 
                                left_on='Final_cluster', right_on='cluster', how='left')

    if not df_sample.empty:
        most_freq_cluster = df_sample['Final_cluster'].value_counts().idxmax()
        default_median = cluster_stats.loc[cluster_stats['cluster'] == most_freq_cluster, 'Median'].values[0]
    else:
        default_median = np.nan

    df_sample['Median_est'] = df_sample['Median'].fillna(default_median)
    df_result = df_cat.merge(df_sample[['PATIENT_ID', 'Median_est', 'Final_cluster']], 
                             on='PATIENT_ID', how='left')
    df_result['Median_est'] = df_result['Median_est'].fillna(default_median)
    df_result['Final_cluster'] = df_result['Final_cluster'].fillna(-1)
    
    return df_result


# 4. SEE Estimator Using DBSCAN
def see_estimator_dbscan(category: str, df: pd.DataFrame, eps=None, min_samples=5) -> pd.DataFrame:
    df_cat = df[df['CATEGORY'] == category].copy()
    df_cat['DATE'] = pd.to_datetime(df_cat['DATE'], format='%m/%d/%Y')
    df_cat.sort_values(by=['PATIENT_ID', 'DATE'], inplace=True)
    df_cat['prev_DATE'] = df_cat.groupby('PATIENT_ID')['DATE'].shift(1)
    df_sample = df_cat.dropna(subset=['prev_DATE']).copy()
    df_sample = df_sample.groupby('PATIENT_ID').apply(lambda x: x.sample(1, random_state=1234)).reset_index(drop=True)
    df_sample = df_sample[['PATIENT_ID', 'DATE', 'prev_DATE']].copy()
    df_sample['event_interval'] = (df_sample['DATE'] - df_sample['prev_DATE']).dt.days.astype(float)

    # Compute ECDF
    x_ecdf, y_ecdf = ecdf(df_sample['event_interval'].values)
    df_ecdf = pd.DataFrame({'x': x_ecdf, 'y': y_ecdf})
    df_ecdf_80 = df_ecdf[df_ecdf['y'] <= 0.8]
    ni = df_ecdf_80['x'].max()
    df_filtered = df_sample[df_sample['event_interval'] <= ni].copy()

    # DBSCAN on the ECDF x-values
    X_db = df_ecdf['x'].values.reshape(-1, 1)
    if eps is None:
        eps = 0.1 * (X_db.max() - X_db.min())
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_ecdf['cluster'] = dbscan.fit_predict(X_db)
    
    # Exclude noise points
    df_ecdf_clustered = df_ecdf[df_ecdf['cluster'] != -1].copy()
    if df_ecdf_clustered.empty:
        # If no clusters found
        return df_cat.assign(Median_est=np.nan, Final_cluster=-1)

    # Compute cluster boundaries
    cluster_stats = df_ecdf_clustered.groupby('cluster')['x'].agg(
        min_log=lambda s: np.log(s).min(),
        max_log=lambda s: np.log(s).max(),
        median_log=lambda s: np.log(s).median()
    ).reset_index()
    cluster_stats['Minimum'] = np.exp(cluster_stats['min_log'])
    cluster_stats['Maximum'] = np.exp(cluster_stats['max_log'])
    cluster_stats['Median'] = np.exp(cluster_stats['median_log'])
    cluster_stats = cluster_stats[cluster_stats['Median'] > 0]

    def assign_cluster_dbscan(interval):
        for _, row in cluster_stats.iterrows():
            if interval >= row['Minimum'] and interval <= row['Maximum']:
                return row['cluster']
        return np.nan

    df_sample['Final_cluster'] = df_sample['event_interval'].apply(assign_cluster_dbscan)
    df_sample = df_sample.dropna(subset=['Final_cluster']).copy()
    df_sample = df_sample.merge(cluster_stats[['cluster', 'Median']], 
                                left_on='Final_cluster', right_on='cluster', how='left')

    if not df_sample.empty:
        most_freq_cluster = df_sample['Final_cluster'].value_counts().idxmax()
        default_median = cluster_stats.loc[cluster_stats['cluster'] == most_freq_cluster, 'Median'].values[0]
    else:
        default_median = np.nan

    df_sample['Median_est'] = df_sample['Median'].fillna(default_median)
    df_result = df_cat.merge(df_sample[['PATIENT_ID', 'Median_est', 'Final_cluster']], 
                             on='PATIENT_ID', how='left')
    df_result['Median_est'] = df_result['Median_est'].fillna(default_median)
    df_result['Final_cluster'] = df_result['Final_cluster'].fillna(-1)
    
    return df_result


# 5. Silhouette Comparison (K-Means only)
def silhouette_comparison(category: str, df: pd.DataFrame, k_range=range(2, 11)) -> int:
    df_cat = df[df['CATEGORY'] == category].copy()
    df_cat['DATE'] = pd.to_datetime(df_cat['DATE'], format='%m/%d/%Y')
    df_cat.sort_values(by=['PATIENT_ID', 'DATE'], inplace=True)
    df_cat['prev_DATE'] = df_cat.groupby('PATIENT_ID')['DATE'].shift(1)
    
    df_sample = df_cat.dropna(subset=['prev_DATE']).copy()
    df_sample = df_sample.groupby('PATIENT_ID').apply(lambda x: x.sample(1, random_state=1234)).reset_index(drop=True)
    df_sample = df_sample[['PATIENT_ID', 'DATE', 'prev_DATE']].copy()
    df_sample['event_interval'] = (df_sample['DATE'] - df_sample['prev_DATE']).dt.days.astype(float)

    x_ecdf, y_ecdf = ecdf(df_sample['event_interval'].values)
    df_ecdf = pd.DataFrame({'x': x_ecdf, 'y': y_ecdf})

    scaler = StandardScaler()
    X = scaler.fit_transform(df_ecdf[['x']])

    silhouette_scores = []
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=1234)
        labels = kmeans_temp.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    best_k = k_range[np.argmax(silhouette_scores)]

    # Plot silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), silhouette_scores, marker='o')
    plt.title(f'Silhouette Analysis for {category} (best_k = {best_k})')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'best_k = {best_k}')
    plt.legend()
    plt.show()

    print(f"Optimal number of clusters for {category} by silhouette: {best_k}")
    return best_k


# 6. Compare 80% ECDF for K-Means vs. DBSCAN (medA vs. medB)
def compare_ecdf80_kmeans_dbscan(df: pd.DataFrame, categories=['medA','medB'], 
                                 k=2, eps=None, min_samples=5):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for i, category in enumerate(categories):
        # Data prep
        df_cat = df[df['CATEGORY'] == category].copy()
        df_cat['DATE'] = pd.to_datetime(df_cat['DATE'], format='%m/%d/%Y')
        df_cat.sort_values(by=['PATIENT_ID', 'DATE'], inplace=True)
        df_cat['prev_DATE'] = df_cat.groupby('PATIENT_ID')['DATE'].shift(1)

        df_sample = df_cat.dropna(subset=['prev_DATE']).copy()
        df_sample = df_sample.groupby('PATIENT_ID').apply(lambda x: x.sample(1, random_state=1234)).reset_index(drop=True)
        df_sample['event_interval'] = (df_sample['DATE'] - df_sample['prev_DATE']).dt.days.astype(float)

        x_ecdf, y_ecdf = ecdf(df_sample['event_interval'].values)
        df_ecdf = pd.DataFrame({'x': x_ecdf, 'y': y_ecdf})
        df_ecdf80 = df_ecdf[df_ecdf['y'] <= 0.8].copy()
        
        # K-Means
        km = KMeans(n_clusters=k, random_state=1234)
        df_ecdf80['cluster_km'] = km.fit_predict(df_ecdf80[['x']])
        cluster_stats_km = df_ecdf80.groupby('cluster_km')['x'].agg(
            min_log=lambda s: np.log(s).min(),
            max_log=lambda s: np.log(s).max()
        ).reset_index()
        cluster_stats_km['Minimum'] = np.exp(cluster_stats_km['min_log'])
        cluster_stats_km['Maximum'] = np.exp(cluster_stats_km['max_log'])

        # DBSCAN
        X_db = df_ecdf80['x'].values.reshape(-1, 1)
        eps_val = eps
        if eps_val is None:
            eps_val = 0.1 * (X_db.max() - X_db.min())
        db = DBSCAN(eps=eps_val, min_samples=min_samples)
        df_ecdf80['cluster_db'] = db.fit_predict(X_db)
        df_ecdf80_db = df_ecdf80[df_ecdf80['cluster_db'] != -1].copy()
        cluster_stats_db = df_ecdf80_db.groupby('cluster_db')['x'].agg(
            min_log=lambda s: np.log(s).min(),
            max_log=lambda s: np.log(s).max()
        ).reset_index()
        cluster_stats_db['Minimum'] = np.exp(cluster_stats_db['min_log'])
        cluster_stats_db['Maximum'] = np.exp(cluster_stats_db['max_log'])

        # Subplots
        # Left col -> K-Means, Right col -> DBSCAN
        ax_km = axs[i, 0]
        ax_db = axs[i, 1]

        # Plot K-Means 80% ECDF
        ax_km.plot(df_ecdf80['x'], df_ecdf80['y'], marker='.', linestyle='none')
        for idx, row in cluster_stats_km.iterrows():
            ax_km.axvline(row['Minimum'], color='red', linestyle='--', label='Boundary' if idx == 0 else "")
            ax_km.axvline(row['Maximum'], color='red', linestyle='--')
        ax_km.set_title(f"80% ECDF (K-Means, {category})")
        ax_km.set_xlabel('Event Interval (days)')
        ax_km.set_ylabel('ECDF')
        ax_km.legend()

        # Plot DBSCAN 80% ECDF
        ax_db.plot(df_ecdf80['x'], df_ecdf80['y'], marker='.', linestyle='none')
        for idx, row in cluster_stats_db.iterrows():
            ax_db.axvline(row['Minimum'], color='green', linestyle='--', label='Boundary' if idx == 0 else "")
            ax_db.axvline(row['Maximum'], color='green', linestyle='--')
        ax_db.set_title(f"80% ECDF (DBSCAN, {category})")
        ax_db.set_xlabel('Event Interval (days)')
        ax_db.set_ylabel('ECDF')
        ax_db.legend()

    plt.tight_layout()
    plt.show()


# Example main() function to demonstrate usage
def main():
    # Step 1: Load Data
    df = load_data("../data/med_events.csv")
    
    # Step 2: K-Means SEE for medA
    medA_kmeans = see_estimator_kmeans("medA", df)
    print("K-Means SEE for medA (sample):")
    print(medA_kmeans.head())

    # Step 3: DBSCAN SEE for medB
    medB_dbscan = see_estimator_dbscan("medB", df, eps=None, min_samples=5)
    print("\nDBSCAN SEE for medB (sample):")
    print(medB_dbscan.head())

    # Step 4: Silhouette comparison for medA
    best_k_medA = silhouette_comparison("medA", df)
    print(f"\nOptimal k for medA is {best_k_medA}.")

    # Step 5: Compare 80% ECDF for medA and medB
    compare_ecdf80_kmeans_dbscan(df, categories=['medA','medB'], k=2, eps=None, min_samples=5)


# Allow running from the command line
if __name__ == "__main__":
    main()
