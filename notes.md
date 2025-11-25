### 📓 Carnet de Bord : Développement U-Net Deblurring

#### 1. Problématique Initiale & Contraintes
* **Architecture de base :** U-Net standard (2015).
* **Problème :** Modèle de ~28M à 30M de paramètres trop lourd pour les ressources disponibles (Google Colab Free / MacBook MPS).
    * *Symptômes :* Timeout après quelques epochs, OOM (Out of Memory), itérations trop lentes.
* **Objectif :** Créer une architecture "Lightweight" (< 5M params) capable d'apprendre efficacement sans sacrifier la capacité de reconstruction.

#### 2. Optimisation de l'Architecture (V1)
* **Réduction des paramètres :** Passage de 28M à ~3M.
    * Remplacement des Convolutions Standards par des **Depthwise Separable Convolutions (DSConv)** (Gain : facteur ~8 sur les poids).
    * Réduction du nombre de filtres initiaux (`start_filters`) de 64 à 32.
    * Remplacement des `ConvTranspose2d` (lourdes, artefacts damier) par `Upsample Bilinear` + `Conv`.
* **Choix structurels :**
    * Maintien d'une convolution standard 4x4 (stride 2) pour le *Downsampling* afin de préserver l'information spatiale critique (responsable d'1M de params à elle seule, mais jugée nécessaire).
    * **Global Residual Connection :** Adoption de la stratégie $Output = Input + Network(Input)$ pour forcer le réseau à apprendre uniquement le résidu (le flou) plutôt que de reconstruire l'image entière.

#### 3. Stratégie de Données (Data Pipeline)
* **Training vs Inference :**
    * Entraînement sur des **Random Crops (256x256)** pour gérer la VRAM et augmenter la diversité locale.
    * Validation/Test sur image complète via stratégie de **Tiling (Tuilage)** avec *Overlap* et *Blending* pour éviter les effets de bord sur les images HD (1280x720).
* **Data Augmentation (Crucial pour le défloutage) :**
    * Flip Horizontal & Vertical.
    * **Rotation 90°/180°/270° :** Indispensable pour varier la direction des vecteurs de flou de mouvement (transformer un flou gauche-droite en haut-bas).
    * *Note :* Application stricte de la "Joint Transform" (même seed aléatoire pour l'image floue et l'image nette).

#### 4. Adaptation Matérielle (MacBook MPS)
* **Optimisation I/O :** Passage de `num_workers=4` à `num_workers=0` pour éviter les instabilités du multiprocessing sur puce Apple Silicon.
* **Gestion Mémoire (VRAM 16Go) :**
    * Détection d'un OOM avec Batch Size 32 + Modèle 3M.
    * Ajustement du **Batch Size à 16** (ou 32 avec *Gradient Accumulation* simulé).
    * Utilisation de `torch.mps.empty_cache()` et `clip_grad_norm_` pour la stabilité.

#### 5. Analyse du Run #1 (Night Run)
* **Résultats :**
    * Training PSNR : **29.32 dB**
    * Validation PSNR : **27.50 dB**
* **Observations :**
    * Le modèle apprend bien (pas de divergence).
    * L'écart Train/Val (1.8 dB) suggère un léger "Generalization Gap", potentiellement dû au fait que les crops d'entraînement sont parfois "faciles" (ciel uni) vs les crops de validation centrés (objets complexes).
    * **Problème Majeur :** Le Learning Rate est resté constant (`2e-4`). Le scheduler `ReduceLROnPlateau` était trop timide (patience trop élevée ou seuil non atteint), empêchant le modèle d'affiner les micro-détails (convergence fine).

#### 6. Plan d'Action pour le Run #2 (V2)
Pour viser > 28.5 dB en validation :

1.  **Architecture (Scale Up) :** Augmentation de la largeur du modèle. Passage de `start_filters=32` à **48** (environ 6.5M params) pour augmenter la capacité de mémorisation des textures complexes.
2.  **Scheduler :** Remplacement de `ReduceLROnPlateau` par **`CosineAnnealingLR`**.
    * *But :* Forcer mathématiquement la descente du LR jusqu'à `1e-6` à la fin de l'entraînement pour garantir la phase de finition ("Fine-tuning").
3.  **Régularisation :**
    * Augmentation du **Weight Decay** de `1e-4` à `1e-3` pour compenser l'augmentation de la taille du modèle et éviter l'overfitting.
    * Ajout de **ColorJitter** (très léger) dans l'augmentation de données pour robustifier le modèle face aux variations colorimétriques.