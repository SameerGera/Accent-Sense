# AccentSense Dataset Specification (`DATASET.md`)

This document details the dataset provenance, metadata structure, speaker-disjoint splitting mathematics, and regional target curation for **AccentSense**.

---

## 📊 Dataset Provenance: AI4Bharat Svarah

- **Identifier**: `ai4bharat/Svarah` (Hugging Face Datasets)
- **Primary Domain**: Indian-accented English speech
- **Total Audio**: 9.6 Hours of transcribed English speech
- **Speaker Count**: 117 speakers
- **Geographic Coverage**: 65 districts across 19 Indian states
- **Audio Format**: 16 kHz WAV, single-channel mono

### Metadata Schema (`meta_speaker_stats.csv`)
| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `speaker_id` | String | `spk_042` | Unique speaker identifier (**Grouping variable**) |
| `duration` | Float | `14.2` | Duration of audio segment in seconds |
| `text` | String | `"The meeting is at five."`| Ground-truth transcription of English speech |
| `gender` | Categorical | `Male` / `Female` | Speaker gender |
| `age-group` | Categorical | `18-30`, `30-45`, ... | Age bracket |
| `native_place_state` | String | `Madhya Pradesh` | Speaker's home state |
| `native_place_district`| String | `Bhopal` | Speaker's home district |
| `primary_language` | String | `Hindi`, `Gujarati`, ...| Speaker's reported primary language |

---

## 🎯 The 4 Regional Phonological Anchors

To eliminate the statistical variance of 19 under-sampled classes, we filter Svarah into **4 high-contrast regional phonological anchors**:

```
                       ┌───────────────────────────────────────────────┐
                       │     4-Class Regional Phonological Anchors     │
                       └───────────────────────┬───────────────────────┘
                                               │
            ┌──────────────────────────────────┴──────────────────────────────────┐
            ▼                                                                     ▼
┌───────────────────────────────────────┐             ┌───────────────────────────────────────┐
│          INDO-ARYAN FAMILY            │             │           DRAVIDIAN FAMILY            │
├───────────────────────────────────────┤             ├───────────────────────────────────────┤
│ 1. Northern_Hindi (Delhi / UP)        │             │ 4. Southern_Tamil (Tamil Nadu)        │
│ 2. Central_MP (Madhya Pradesh / Malwa)│             │    [Cross-family Dravidian control]   │
│ 3. Western_Gujarati (Gujarat)         │             │                                       │
└───────────────────────────────────────┘             └───────────────────────────────────────┘
```

### Classification Criteria
```python
def label_regional_4class(row):
    state = str(row.get("native_place_state", ""))
    lang = str(row.get("primary_language", ""))
    
    if "Madhya Pradesh" in state:
        return "Central_MP"
    elif lang == "Gujarati" or "Gujarat" in state:
        return "Western_Gujarati"
    elif lang == "Tamil" or "Tamil Nadu" in state:
        return "Southern_Tamil"
    elif lang == "Hindi":
        return "Northern_Hindi"
    return "Other"
```

---

## 🛡️ Leakage-Free Splitting: Speaker-Disjoint Mathematics

### The Speaker Memorization Threat
If speaker $S_k$ appears in both training and test sets:
$$\text{Train} \cap \text{Test} = \{ S_k \}$$
A deep acoustic network will learn:
$$f(\text{Audio}) \to \text{Vocal Tract Timbre of } S_k \to \text{Class } Y$$
This creates an artificial test accuracy of $>90\%$, but drops to chance level when deployed on a new user.

### The Mathematical Guarantee
We enforce **Speaker-Disjoint Group K-Fold**:
Let $\mathcal{S}$ be the set of all unique speaker IDs.
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

Partitioning is performed using `sklearn.model_selection.StratifiedGroupKFold`:
```python
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(X=df, y=df["target"], groups=df["speaker_id"]))
```
Every model evaluation is strictly performed on speakers the model has never heard before.
