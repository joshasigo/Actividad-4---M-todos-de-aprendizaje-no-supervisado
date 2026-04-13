# =============================================================
# PROYECTO: Agrupamiento de Estaciones de TransMilenio
# MÉTODO: Aprendizaje No Supervisado - Clustering Jerárquico
# CURSO: Inteligencia Artificial
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin ventana gráfica

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ---------------------------------------------------------------
print("=" * 60)
print("  CLUSTERING JERÁRQUICO - ESTACIONES TRANSMILENIO")
print("=" * 60)

df = pd.read_csv('transmilenio_dataset.csv')

print("\n[1] Información general del dataset:")
print(f"    Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
print(f"\n    Columnas: {list(df.columns)}")
print(f"\n    Estadísticas básicas:")
print(df.describe().round(2).to_string())

print(f"\n    Estaciones únicas: {df['estacion'].nunique()}")
print(f"    Estaciones: {list(df['estacion'].unique())}")

# ---------------------------------------------------------------
# 2. PREPROCESAMIENTO - AGRUPACIÓN POR ESTACIÓN
# ---------------------------------------------------------------
print("\n[2] Preprocesando datos...")

# Promediar métricas por estación para representar el perfil global
df_estacion = df.groupby('estacion').agg(
    pasajeros_promedio    = ('pasajeros_hora', 'mean'),
    pasajeros_max         = ('pasajeros_hora', 'max'),
    tiempo_espera_prom    = ('tiempo_espera_min', 'mean'),
    buses_disponibles_prom= ('buses_disponibles', 'mean'),
    distancia_centro_km   = ('distancia_centro_km', 'mean')
).reset_index()

print("\n    Perfil por estación (primeras 5 filas):")
print(df_estacion.head().to_string())

# Selección de variables numéricas para el modelo
variables = [
    'pasajeros_promedio',
    'pasajeros_max',
    'tiempo_espera_prom',
    'buses_disponibles_prom',
    'distancia_centro_km'
]

X = df_estacion[variables].values
etiquetas = df_estacion['estacion'].values

# Normalización con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n    Variables seleccionadas: {variables}")
print(f"    Datos normalizados con StandardScaler ✓")

# ---------------------------------------------------------------
# 3. DENDROGRAMA Y SELECCIÓN DE CLUSTERS
# ---------------------------------------------------------------
print("\n[3] Generando dendrograma con método Ward...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Análisis de Clustering Jerárquico - Estaciones TransMilenio',
             fontsize=14, fontweight='bold', y=1.02)

# Calcular linkage con método Ward
Z = linkage(X_scaled, method='ward', metric='euclidean')

# Dendrograma completo
ax1 = axes[0]
dendrogram(
    Z,
    labels=etiquetas,
    orientation='top',
    color_threshold=4.5,
    leaf_rotation=45,
    leaf_font_size=10,
    ax=ax1
)
ax1.set_title('Dendrograma - Método Ward', fontsize=12, fontweight='bold')
ax1.set_xlabel('Estaciones', fontsize=10)
ax1.set_ylabel('Distancia Euclidiana', fontsize=10)
ax1.axhline(y=4.5, color='red', linestyle='--', linewidth=1.5,
            label='Corte seleccionado (k=3)')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# ---------------------------------------------------------------
# 4. EVALUACIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
# ---------------------------------------------------------------
print("\n[4] Evaluando número óptimo de clusters...")

k_range = range(2, 6)
silhouette_scores = []
db_scores = []
ch_scores = []

for k in k_range:
    labels_k = fcluster(Z, t=k, criterion='maxclust')
    sil = silhouette_score(X_scaled, labels_k)
    db  = davies_bouldin_score(X_scaled, labels_k)
    ch  = calinski_harabasz_score(X_scaled, labels_k)
    silhouette_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)
    print(f"    k={k} | Silhouette={sil:.3f} | Davies-Bouldin={db:.3f} | Calinski-Harabasz={ch:.1f}")

# Gráfica de métricas
ax2 = axes[1]
ax2.plot(list(k_range), silhouette_scores, 'bo-', linewidth=2,
         markersize=8, label='Silhouette Score')
ax2.set_xlabel('Número de Clusters (k)', fontsize=10)
ax2.set_ylabel('Silhouette Score', fontsize=10, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_title('Evaluación de Métricas por Número de Clusters', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(list(k_range))

plt.tight_layout()
plt.savefig('dendrograma_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    Gráfica guardada: dendrograma_metricas.png ✓")

# ---------------------------------------------------------------
# 5. MODELO FINAL CON k=3 CLUSTERS
# ---------------------------------------------------------------
K_OPTIMO = 3
print(f"\n[5] Aplicando clustering jerárquico con k={K_OPTIMO}...")

labels_final = fcluster(Z, t=K_OPTIMO, criterion='maxclust')
df_estacion['cluster'] = labels_final

# Renombrar clusters según características
nombres_cluster = {1: 'Alta Demanda', 2: 'Demanda Media', 3: 'Baja Demanda'}
df_estacion['tipo_estacion'] = df_estacion['cluster'].map(nombres_cluster)

print("\n    Asignación de estaciones por cluster:")
print("    " + "-" * 50)
for cluster_id in sorted(df_estacion['cluster'].unique()):
    estaciones = df_estacion[df_estacion['cluster'] == cluster_id]['estacion'].tolist()
    nombre = nombres_cluster[cluster_id]
    print(f"    Cluster {cluster_id} - {nombre}: {estaciones}")

# Perfil de cada cluster
print("\n    Perfil promedio por cluster:")
perfil = df_estacion.groupby('tipo_estacion')[variables].mean().round(2)
print(perfil.to_string())

# ---------------------------------------------------------------
# 6. VISUALIZACIÓN PCA - CLUSTERS EN 2D
# ---------------------------------------------------------------
print("\n[6] Generando visualización PCA 2D...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

varianza_explicada = pca.explained_variance_ratio_ * 100
print(f"    Varianza explicada PC1: {varianza_explicada[0]:.1f}%")
print(f"    Varianza explicada PC2: {varianza_explicada[1]:.1f}%")
print(f"    Varianza total explicada: {sum(varianza_explicada):.1f}%")

colores = {1: '#E74C3C', 2: '#3498DB', 3: '#2ECC71'}
marcadores = {1: 'o', 2: 's', 3: '^'}

fig2, ax = plt.subplots(figsize=(10, 7))

for cluster_id in sorted(df_estacion['cluster'].unique()):
    mask = df_estacion['cluster'] == cluster_id
    nombre = nombres_cluster[cluster_id]
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colores[cluster_id],
        marker=marcadores[cluster_id],
        s=200, label=f'Cluster {cluster_id}: {nombre}',
        edgecolors='black', linewidth=1.2, zorder=3
    )
    for i, est in enumerate(etiquetas[mask]):
        ax.annotate(est, (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color='#2C3E50')

ax.set_xlabel(f'Componente Principal 1 ({varianza_explicada[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'Componente Principal 2 ({varianza_explicada[1]:.1f}%)', fontsize=11)
ax.set_title('Clustering Jerárquico de Estaciones TransMilenio\n'
             f'(Visualización PCA - Varianza total: {sum(varianza_explicada):.1f}%)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('clusters_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Gráfica guardada: clusters_pca.png ✓")

# ---------------------------------------------------------------
# 7. MÉTRICAS FINALES DEL MODELO
# ---------------------------------------------------------------
print("\n[7] Métricas de evaluación del modelo final (k=3):")
sil_final = silhouette_score(X_scaled, labels_final)
db_final  = davies_bouldin_score(X_scaled, labels_final)
ch_final  = calinski_harabasz_score(X_scaled, labels_final)

print(f"    Silhouette Score      : {sil_final:.4f}  (más cercano a 1 es mejor)")
print(f"    Davies-Bouldin Score  : {db_final:.4f}  (más cercano a 0 es mejor)")
print(f"    Calinski-Harabasz     : {ch_final:.2f} (mayor es mejor)")

# ---------------------------------------------------------------
# 8. EXPORTAR RESULTADOS
# ---------------------------------------------------------------
df_resultado = df_estacion[['estacion', 'cluster', 'tipo_estacion'] + variables]
df_resultado.to_csv('resultados_clustering.csv', index=False)
print("\n[8] Resultados exportados: resultados_clustering.csv ✓")

# ---------------------------------------------------------------
# RESUMEN FINAL
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("  RESUMEN DEL MODELO")
print("=" * 60)
print(f"  Algoritmo       : Clustering Jerárquico (Método Ward)")
print(f"  Dataset         : {df.shape[0]} registros de {df['estacion'].nunique()} estaciones")
print(f"  Variables       : {len(variables)} características")
print(f"  Clusters finales: {K_OPTIMO}")
print(f"  Silhouette Score: {sil_final:.4f}")
print(f"\n  Archivos generados:")
print("    - dendrograma_metricas.png")
print("    - clusters_pca.png")
print("    - resultados_clustering.csv")
print("=" * 60)
print("  Proceso completado exitosamente ✓")
print("=" * 60)
