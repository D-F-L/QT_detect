import pandas as pd
import networkx as nx

def find_spectral_clusters(
    csv_path,
    mean_diff_thresh,
    median_diff_thresh,
    min_cluster_size=2,
    k_core=None,
    round_digits=None,
    output_csv=None
):
    df = pd.read_csv(csv_path)

    required_cols = ['mean_diff', 'median_diff', 'f1', 'f2']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中缺少必要列: {col}")

    if round_digits is not None:
        df['f1'] = df['f1'].round(round_digits)
        df['f2'] = df['f2'].round(round_digits)

    # 过滤相似谱线对
    filtered_df = df[
        (df['mean_diff'] <= mean_diff_thresh) &
        (df['median_diff'] <= median_diff_thresh)
    ].copy()

    # 构图
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        G.add_edge(row['f1'], row['f2'])

    # 可选 k-core
    if k_core is not None:
        G = nx.k_core(G, k=k_core)

    # 查找连通分量
    components = list(nx.connected_components(G))

    clusters = []
    summary_rows = []

    for i, comp in enumerate(components, 1):
        cluster = sorted(list(comp))
        if len(cluster) < min_cluster_size:
            continue

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)

    for i, cluster in enumerate(clusters, 1):
        summary_rows.append({
            'cluster_id': i,
            'cluster_size': len(cluster),
            'min_freq': min(cluster),
            'max_freq': max(cluster),
            'mean_freq': sum(cluster) / len(cluster),
            'frequencies': ','.join(map(str, cluster))
        })

    summary_df = pd.DataFrame(summary_rows)

    print("共找到簇数量:", len(clusters))
    print(summary_df)

    if output_csv is not None:
        summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {output_csv}")

    return clusters, summary_df, filtered_df
	
	
clusters, summary_df, filtered_df = find_spectral_clusters(
    csv_path='doa.csv',
    mean_diff_thresh=5,
    median_diff_thresh=3,
    min_cluster_size=3,
    k_core=2,
    round_digits=3,
    output_csv='cluster_summary.csv'
)