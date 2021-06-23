from neo4j import GraphDatabase
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test"))

with driver.session(database="tweet") as session:
    session.run('''
    CALL gds.alpha.node2vec.write({
    nodeProjection: "Tweet",
    relationshipProjection: {
    Retweets: {type: "RETWEETS",
      orientation: "UNDIRECTED"},
    Reply: {type: "REPLY_TO",
      orientation: "UNDIRECTED"}},
    embeddingDimension: 2,
    iterations: 5,
    walkLength: 10,
    writeProperty: "embeddingNode2vec"
    });
    ''')

    result = session.run("""
    MATCH (u:Tweet)
    RETURN u.name AS User, u.embeddingNode2vec AS embedding
    """)
    X = pd.DataFrame([dict(record) for record in result])

for i in range(len(X.embedding[0])):
    temp=[]
    for j in range(len(X.embedding)):
        temp.append(X.embedding[j][i])
    X.insert(1, "emb_"+str(i), temp)

emb_only = X.drop(['User', 'embedding'],  axis='columns')

scaler = MinMaxScaler()
emb_only[:] = scaler.fit_transform(emb_only[:])


def compute_clustering_and_plot(K, data, results):
    # Train the model
    kmeans = KMeans(n_clusters=K, max_iter=1000)
    kmeans.fit(emb_only);
    data['kmeans_labels'] = kmeans.labels_
    # Plot results
    sns.scatterplot(x='emb_0',
                    y='emb_1',
                    hue='kmeans_labels',
                    data=data);
    plt.plot(kmeans.cluster_centers_[:, 0],
             kmeans.cluster_centers_[:, 1],
             'bo', markersize=12, alpha=0.7);
    plt.title('K Means predictions with K={}'.format(K));
    plt.show();
    # Compute metrics
    CH = calinski_harabasz_score(emb_only, kmeans.labels_)
    S = silhouette_score(emb_only, kmeans.labels_)
    DB = davies_bouldin_score(emb_only, kmeans.labels_)

    results.loc[('kmeans', k), :] = [CH, S, DB]
    return results


K_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

index = pd.MultiIndex.from_arrays([['kmeans'], [2]], names=('model', 'K'))
results_df = pd.DataFrame(index=index, columns=['CH score', 'Silhouette score', 'DB score'])

for k in K_values:
    results = compute_clustering_and_plot(k, emb_only, results_df)

print(results)

results = results.sort_index()
results = results.astype(float)

fig, ax = plt.subplots(1,3, figsize=(16,6))
sns.lineplot(x='K', y='CH score',data=results.reset_index(),label='CH score', color='r', ax=ax[0]);
sns.lineplot(x='K', y='Silhouette score',data=results.reset_index(),label='Silhouette',ax=ax[1]);
sns.lineplot(x='K', y='DB score',data=results.reset_index(),label='DB',ax=ax[2]);
plt.show()
