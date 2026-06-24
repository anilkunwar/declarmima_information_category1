# The code that is used to retrieve the above datasets is:

[![meaningtowords](https://img.shields.io/badge/vectorlessragmodel-v88.e-yellow)](https://vectorless-reasoning-rag-information-retrieval-parallel88e.streamlit.app/) (v88e is v88d with more finetuning on the postprocessing part)

# Physical Quantities 
[![meaningtowords](https://img.shields.io/badge/physicalquantities-v1.0-violet)](https://physical-quantities-retrieval-performance1.streamlit.app/) (Comparison among different LLM models)

[![meaningtowords](https://img.shields.io/badge/physicalquantities-v2.0-violet)](https://physical-quantities-retrieval-performance2.streamlit.app/) (Comparison among different LLM models, Figure Customization)

[![meaningtowords](https://img.shields.io/badge/physicalquantities-v3.0-violet)](https://physical-quantities-retrieval-performance3.streamlit.app/) (Comparison among different LLM models, Figure Customization)

[![meaningtowords](https://img.shields.io/badge/physicalquantities-v4.0-violet)](https://physical-quantities-retrieval-performance4.streamlit.app/) (Comparison among different LLM models, Figure Customization and Enhancement)

[![meaningtowords](https://img.shields.io/badge/physicalquantities-v5.0-violet)](https://physical-quantities-retrieval-performance5.streamlit.app/) (Comparison among different LLM models, Figure Customization and Enhancement)

# Alloys Materials 

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v1.0-green)](https://alloys-materials-retrieval-performance1.streamlit.app/) (Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v2.0-green)](https://alloys-materials-retrieval-performance2.streamlit.app/) (Comparison among different LLM models, a document for a given model is given a unique label)


[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v3.0-green)](https://alloys-materials-retrieval-performance3.streamlit.app/) (Unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v4.0-green)](https://alloys-materials-retrieval-performance4.streamlit.app/) (Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v5.0-green)](https://alloys-materials-retrieval-performance5.streamlit.app/) (Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v6.0-green)](https://alloys-materials-retrieval-performance6.streamlit.app/) (Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v7.0-green)](https://alloys-materials-retrieval-performance7.streamlit.app/) (Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v8.0-green)](https://alloys-materials-retrieval-performance8.streamlit.app/) (Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v9.0-green)](https://alloys-materials-retrieval-performance9.streamlit.app/) (Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v10.0-green)](https://alloys-materials-retrieval-performance10.streamlit.app/) (Number of Labels is Equal to the Number of Nodes in Chord Diagram, Previous versions have labels less than the number of nodes in chord diagram, Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v11.0-green)](https://alloys-materials-retrieval-performance11.streamlit.app/) (Number of Labels is Equal to the Number of Nodes in Chord Diagram, Previous versions have labels less than the number of nodes in chord diagram, Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v12.0-green)](https://alloys-materials-retrieval-performance12.streamlit.app/) (v10 with no unreferenced numbers, Number of Labels is Equal to the Number of Nodes in Chord Diagram, Previous versions have labels less than the number of nodes in chord diagram, Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)

[![meaningtowords](https://img.shields.io/badge/alloysmaterials-v13.0-green)](https://alloys-materials-retrieval-performance13.streamlit.app/) (v10 with no unreferenced numbers, Number of Labels is Equal to the Number of Nodes in Chord Diagram, Previous versions have labels less than the number of nodes in chord diagram, Enhanced unified visualizations, Comparison among different LLM models, a document for a given model is given a unique label)



# LLM Names :

| Model Alias | LLM Name    |
| :---------- | :---------  |
| Model A     | Falcon3 10B |
| Model B     | Mistral 7B  |
| Model C     | Qwen 14B    |
| Model D     | Qwen 7B     |


## Evaluation Configuration & Context Window

To benchmark model performance on the retrieval-augmented extraction tasks, we evaluated several open-weight LLMs under consistent experimental conditions, with one key distinction: **maximum retrieved context length**.

The system’s `max_retrieval_chars` parameter (which controls the number of characters passed from the retrieved document sections to the LLM for extraction) was set differently based on hardware constraints and the model's native context window capacity.

| Model Alias | Full Model Name | Max Retrieval Context (characters) |
| :---------- | :-------------- | :--------------------------------- |
| Model A     | Falcon 10B      | 10,000                             |
| Model B     | Mistral 7B      | 50,000                             |
| Model C     | Qwen 14B        | 50,000                             |
| Model D     | Qwen 7B         | 50,000                             |

> **Note:** The performance CSV results reported for **Model A (Falcon 10B)** were obtained with the **10,000-character** limit, while Models B, C, and D were evaluated with the extended **50,000-character** setting. This distinction is important when comparing extraction completeness, as longer contexts may provide more surrounding evidence for numerical values, whereas shorter contexts force stricter summarization and may favor precision over recall.

All other hyperparameters (e.g., confidence threshold, page window size, retrieval top-k) were kept identical across models to ensure a fair comparison of underlying reasoning capability.
