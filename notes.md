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

#### 7. Intermediate Analysis (Run with Cosine Scheduler)
* **Results Achieved:**
    * **Validation PSNR:** Reached a stable plateau at **28.03 dB** (Target 28+ reached).
    * **Train PSNR:** Oscillating around **29.7 dB**.
    * **Scheduler Behavior:** The switch to `CosineAnnealingLR` is a total success. The learning curve shows a "smooth landing" and much better convergence compared to the previous plateau strategy.
* **Gap Diagnosis (Generalization Gap ~1.7 dB):**
    * The model learns well but seems to hit a ceiling in validation.
    * **Critical Discovery (Bug Fix):** Identified a logic error in `GoProDataset`. Random rotations (0, 90, 180, 270°) were being applied **during validation as well** (missing indentation under `if self.is_train:`).
    * *Impact:* Validation was artificially noisy and harder than intended, likely underestimating the model's true performance.

#### 8. Optimisation Finale pour la V3 (Objectif 28.5+ dB)
Pour maximiser la performance et corriger le gap Train/Val, la stratégie suivante est adoptée :

* **A. Correction du Pipeline de Données :**
    * **Fix Rotation :** Restriction des rotations aléatoires au mode `train` uniquement.
    * **Fix `ColorJitter` :** Correction du crash `TypeError` sur les paramètres `hue` et implémentation d'une synchronisation manuelle stricte pour garantir que l'image floue et nette subissent exactement la même variation colorimétrique.

* **B. Stratégie "Data Scale-Up" Rigoureuse (Split par Séquence) :**
    * *Problème Identifié :* Le dataset GoPro est constitué de séquences vidéo. Un "Random Shuffle" simple des images créerait une **fuite de données (Data Leakage)** massive : le modèle verrait la frame $t$ dans le Train et la frame $t+1$ (quasi-identique) dans la Validation, faussant le score. 
    * *Solution Scientifique :* Adoption d'un **Split par Séquence Vidéo**.
        * Les images sont groupées par dossier parent (Séquence).
        * Le mélange et la découpe se font sur les *noms de séquences*, et non sur les images individuelles.
    * *Action :* Réallocation dynamique de séquences du set de Test original vers le Train pour atteindre un ratio **~90% Train / 10% Val** (au lieu de 66/33).
    * *Gain Double :*
        1.  **Performance :** Le modèle s'entraîne sur ~2800 images (+33%), augmentant la diversité des scènes apprises.
        2.  **Rigueur :** Garantie absolue qu'aucune image de validation ne provient d'une vidéo vue à l'entraînement. La validation teste réellement la généralisation sur une scène inconnue.

* **C. Raffinements de Régularisation :**
    * **Soft Color Jitter :** Ajout d'une variation aléatoire légère ($\pm 15\%$) de luminosité, contraste et saturation.
    * *But :* Forcer le modèle à généraliser sur les structures géométriques plutôt que de mémoriser les histogrammes de couleurs spécifiques des scènes GoPro.

* **Configuration Finale Run V3 :**
    * **Architecture :** `LightweightUNet` (48 filtres initiaux).
    * **Batch Size :** 8 (avec *Gradient Accumulation* si instable).
    * **Scheduler :** `CosineAnnealingLR` (T_max=150).
    * **Dataset :** Split par Séquence 90/10 + Augmentations corrigées.