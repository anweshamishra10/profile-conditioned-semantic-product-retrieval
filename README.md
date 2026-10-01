```
Profile-Conditioned Semantic Product Retrieval
MSc Data Science Individual Project
Project title: Design and Evaluation of Query-Conditioned Recommendation Systems
using Synthetic User-Query Datasets
Implementation focus: Profile-Conditioned Semantic Product Retrieval
Programme: MSc Data Science, City St George's, University of London
```

# `1. Project Overview` 

```
This project investigates whether historical user preference information can
improve semantic product retrieval when combined with a user's current natural-
language product query.
```

```
The central research question is:
Does incorporating historical user preference information into the query
representation improve semantic product retrieval compared with query-only
retrieval?
```

```
The project uses the Amazon Reviews 2023 Electronics dataset to construct a
synthetic query-conditioned evaluation dataset. Historical user interactions are
used to construct textual preference profiles, while held-out products are used
as ground-truth targets for synthetic shopping queries.
The final retrieval system uses:
```

```
    • Qwen2.5-7B-Instruct for synthetic query generation
```

```
    • E5-base-v2 for semantic text encoding
```

- `FAISS IndexFlatIP for large-scale vector retrieval` 

- `A catalogue containing 1,610,012 unique products` 

- `A final evaluation dataset containing 5,559 examples` 

```
2. Final Experimental Pipeline
```

```
The final project pipeline is:
Amazon Reviews 2023 — Electronics
            |
            v
User interaction filtering
            |
            v
Five historical interactions + held-out target
            |
            +----------------------+
            |                      |
            v                      v
User preference profile      Held-out target
            |                      |
            +----------+-----------+
                       |
                       v
             Synthetic query generation
                 (Qwen2.5-7B-Instruct)
                       |
                       v
             -------------------------
             |                       |
             v                       v
       Query-only baseline     Profile + Query
             |                       |
             +-----------+-----------+
                         |
                         v
                    E5-base-v2
                         |
                         v
                 768-dimensional
                     embeddings
                         |
                         v
                   Normalisation
```



```
              Recall@K / MRR@K / NDCG@K
```

```
The baseline and proposed system use the same product catalogue, encoder and
FAISS index. The principal experimental difference is whether historical
preference information is included in the retrieval input.
```

```
3. Repository / Project Structure
The accompanying project files are organised as follows:
Project_Files/
|
```

```
|-- README.md
|
```

```
|-- notebooks/
|   |-- 01_dataset_construction.ipynb
|   |-- 02_baseline_methods.ipynb
|   |-- 03_metadata_enrichment.ipynb
|   |-- 04_user_preference_profiles.ipynb
|   |-- 05_generate_synthetic_queries.ipynb
|   |-- 08_faiss_semantic_retrieval.ipynb
|   |-- 09_blair_baseline.ipynb
|   |-- 10_proposed_model.ipynb
|   |-- 10a_proposed_model_ablation.ipynb
|   `-- 11_evaluation.ipynb
|
|-- results/
|   |-- final_evaluation_results.parquet
|   |-- final_evaluation_results.csv
|   |-- final_ablation_results.parquet
|   |-- final_ablation_results.csv
|   |-- ground_truth_quality_diagnostics.parquet
|   `-- ground_truth_quality_diagnostics.csv
|
|-- generated_data/
|   `-- synthetic_queries.parquet
|
`-- documentation/
The exact contents of the submitted archive may differ slightly depending on
file-size constraints. The notebooks listed above constitute the final
implementation notebooks.
```

```
4. Notebook Execution Order
The final notebooks should be understood in the following order.
01 — Dataset Construction
01_dataset_construction.ipynb
Prepares the Amazon Reviews 2023 Electronics data and constructs the datasets
required for subsequent processing.
02 — Baseline Methods
02_baseline_methods.ipynb
Contains the initial baseline-related analysis and preparation used during the
project.
03 — Metadata Enrichment
03_metadata_enrichment.ipynb
Prepares and enriches product metadata used to construct product
representations.
```

```
04 — User Preference Profiles
04_user_preference_profiles.ipynb
Constructs textual preference profiles from the historical interactions
associated with each evaluation user.
05 — Synthetic Query Generation
05_generate_synthetic_queries.ipynb
Generates natural-language shopping queries using Qwen2.5-7B-Instruct. The
generation process uses the user's preference profile and held-out target
product while applying constraints intended to prevent direct reproduction of
target identifiers.
08 — Semantic Retrieval Infrastructure
08_faiss_semantic_retrieval.ipynb
Creates the semantic product representations and FAISS retrieval infrastructure
using E5-base-v2.
The complete product catalogue contains 1,610,012 unique products.
09 — Query-Only Baseline
09_blair_baseline.ipynb
Runs the query-only semantic retrieval baseline against the full product
catalogue.
10 — Proposed Profile-Conditioned Model
10_proposed_model.ipynb
Runs the proposed profile-conditioned retrieval approach, where the preference
profile and current query are concatenated into a single textual input before
semantic encoding.
10a — Proposed Model Ablation
10a_proposed_model_ablation.ipynb
Evaluates the effect of input token length and ordering, including:
    • Profile → Query, 256 tokens
    • Profile → Query, 512 tokens
    • Query → Profile, 512 tokens
11 — Evaluation
11_evaluation.ipynb
Calculates the final retrieval metrics, exact hit counts, paired comparisons and
ground-truth quality diagnostics.
```

```
5. Data
Source dataset
The project uses the:
Amazon Reviews 2023 — Electronics dataset. (https://amazon-
reviews-2023.github.io/)
The original dataset contains user–product interactions and product metadata. It
does not provide genuine product-search sessions in which a user's natural-
language query is directly linked to the product selected by that user.
Therefore, the project constructs a synthetic query-conditioned evaluation
dataset.
Evaluation construction
Users with at least six relevant interactions were eligible for evaluation
construction.
For each evaluation example:
    • Five historical product interactions are used to represent the user's
previous preferences.
    • A separate product is retained as the held-out target.
    • The historical interactions are used to construct a textual preference
profile.
    • Qwen2.5-7B-Instruct generates a synthetic shopping query.
    • The held-out product is treated as the exact ground-truth target during
evaluation.
The final evaluation dataset contains:
5,559 evaluation examples
The retrieval catalogue contains:
1,610,012 unique products
```

```
6. Data File Formats
```

```
The project primarily uses Parquet for processed datasets because it preserves
```

```
data types and is efficient for large tabular data.
CSV copies are also provided for selected smaller final results to make them
easier to inspect without requiring a Parquet-compatible environment.
Reading Parquet
import pandas as pd
```

```
df = pd.read_parquet("path/to/file.parquet")
print(df.head())
Reading CSV
import pandas as pd
df = pd.read_csv("path/to/file.csv")
print(df.head())
The CSV and Parquet versions of the same result files represent the same
underlying data. The Parquet files remain the primary project format.
```

```
7. Retrieval Method
Product representation
Each catalogue product is represented using available:
    • Product title
    • Brand/store information
    • Product category
These product texts are encoded using E5-base-v2, producing 768-dimensional
embeddings.
Similarity and retrieval
The embeddings are normalised before retrieval.
The project uses:
FAISS IndexFlatIP
Because the vectors are normalised, inner-product similarity corresponds to
cosine similarity.
The retrieval process searches the complete product catalogue rather than a
manually pre-filtered candidate set.
Query-only baseline
Query
  |
E5-base-v2
  |
Normalised embedding
  |
FAISS
  |
Ranked products
Profile-conditioned retrieval
Preference Profile + Query
             |
        E5-base-v2
             |
      Normalised embedding
             |
            FAISS
             |
       Ranked products
No separate profile/query embedding averaging, keyword retrieval stage, manual
product filtering or additional re-ranking model is used in the final retrieval
architecture.
8. Final Evaluation
The retrieval systems are evaluated using:
    • Recall@K
    • Mean Reciprocal Rank (MRR@K)
    • Normalised Discounted Cumulative Gain (NDCG@K)
    • Exact hit counts
    • Paired comparison of retrieval outcomes
Evaluation is performed at:
```

```
    • Top-1
```

```
    • Top-5
```

```
    • Top-10
```

```
    • Top-20
```

```
    • Top-50
```

```
The final results are contained in:
results/final_evaluation_results.parquet
results/final_ablation_results.parquet
results/ground_truth_quality_diagnostics.parquet
CSV copies of the smaller result files are provided where applicable.
```

# `9. Ground-Truth and Evaluation Limitation` 

```
A key limitation of the project is that Amazon Reviews 2023 does not contain
genuine search-query/target-product pairs.
```

```
The query–target pairs used in this project are therefore constructed
synthetically.
```

```
A lexical diagnostic found that approximately 41% of query–ground-truth pairs
had zero lexical overlap between the query and the target product title. This
diagnostic is used to characterise the constructed evaluation data; it is not
treated as a direct relevance metric.
```

```
The results should therefore be interpreted as evidence about retrieval
```

```
behaviour on the constructed evaluation dataset rather than as a direct estimate
of real-world recommendation or search effectiveness.
```

# `10. GenAI Use` 

```
Qwen2.5-7B-Instruct was used to generate the synthetic shopping queries.
The model was provided with the user's preference profile and held-out target
product information during query generation. Prompt constraints were used to
reduce direct reproduction of target identifiers and product-title information.
The project report documents the use of Generative AI, including the purpose,
methodology and limitations of its use.
```

```
Where AI-assisted or automatically generated code was used during development,
its provenance should be identified in accordance with the project submission
requirements.
```

```
Third-party libraries and pretrained models are not reproduced in this source-
code appendix; only project-specific source-code fragments are included.
```

# `11. Source-Code Provenance` 

```
The project uses a combination of:
```

`1. Original project code developed specifically for the implementation.` 

`2. Third-party libraries and frameworks, which are imported and used but whose underlying library source code is not reproduced.` 

`3. Pretrained models, including E5-base-v2 and Qwen2.5-7B-Instruct.` 

```
    4. AI-assisted or automatically generated code, where applicable, which
should be identified in the relevant documentation and report.
The separate file:
```

```
Appendix_Source_Code.txt
```

```
contains the Python source-code cells extracted from the final project notebooks
for originality checking. Notebook JSON, execution outputs, image MIME data and
other non-source notebook content have been removed.
```

# `12. Reproducibility` 

```
The notebooks are provided in their final project order. Reproducing the
complete pipeline requires access to:
```

- `The Amazon Reviews 2023 Electronics source data` 

- `The required Python packages` 

- `E5-base-v2` 

- `Qwen2.5-7B-Instruct for query generation` 

- `Sufficient computational resources for large-scale embedding and FAISS retrieval` 

- `The required project data paths` 

```
The full product embedding matrix is approximately 4.6 GB and is therefore not
included in the small Moodle project archive. Large project artefacts should be
accessed through the accompanying University cloud storage where provided.
```

```
The notebooks contain the project-specific implementation required to understand
how the datasets, profiles, queries, retrieval system and evaluation results
were produced.
```

# `13. Large Files` 

```
Large files that cannot reasonably be included in the Moodle Additional Files
archive is provided through University OneDrive storage.
These may include:
```

- `Large source/processed datasets` 

- `Product embeddings` 

- `Other large generated artefacts` 

```
    • Project presentation video
Link to the OneDrive folder with supporting files: https://cityuni-
my.sharepoint.com/:f:/g/personal/anwesha_mishra_city_ac_uk/
IgCoyKW2u3qFT7JQ-9ZTEq8aAYmdi7sP9gFgzkZotieRxYw?e=xMpAFk
```

# `14. Development Evidence` 

```
The project submission should be considered together with the available
development evidence, including where applicable:
```

- `Preliminary and Original project proposal` 

- `Supervisor meeting records and feedback` 

`15. Final Project Outputs` 

```
The principal final outputs are:
```

- `Final project report` 

- `Source-code appendix` 

- `Final implementation notebooks` 

- `Synthetic evaluation dataset` 

- `Final retrieval results` 

- `Ablation results` 

- `Ground-truth quality diagnostics` 

```
    • Final project presentation/video
Together, these materials provide the project outcomes and supporting evidence
required to audit the implementation and development of the project.
```

# `16. Contact / Project Information` 

```
Student: Anwesha Mishra
Programme: MSc Data Science
Institution: City St George's, University of London
```

```
Project: Design and Evaluation of Query-Conditioned Recommendation Systems using
Synthetic User-Query Datasets
```

