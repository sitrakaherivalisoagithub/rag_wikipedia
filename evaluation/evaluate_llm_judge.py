import requests
import uuid
import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from app.core.dependencies import llm

# ------------------------------------------------------------------
# 1. Modèle Pydantic pour l'Output Structuré du Judge
# ------------------------------------------------------------------
class EvaluationOutput(BaseModel):
    retriever_score: int = Field(
        ..., 
        description="Score de 0 à 5 mesurant si les sources récupérées correspondent aux sources de la réponse attendue (Ground Truth Source)."
    )
    relevance_score: int = Field(
        ..., 
        description="Score de 0 à 5 : La réponse de l'agent répond-elle directement à la question posée ?"
    )
    correctness_score: int = Field(
        ..., 
        description="Score de 0 à 5 : La réponse de l'agent est-elle factuellement conforme à la réponse attendue (Ground Truth Answer) ?"
    )
    faithfulness_score: int = Field(
        ..., 
        description="Score de 0 à 5 (0 = Hallucination totale, 5 = Absence totale d'hallucination / Parfaitement ancré dans les sources)."
    )
    precision_score: int = Field(
        ..., 
        description="Score de 0 à 5 : La réponse est-elle concise, exacte et débarrassée d'informations superflues ?"
    )
    explanation: str = Field(
        ..., 
        description="Explication concise justifiant les notes attribuées par le juge."
    )

# ------------------------------------------------------------------
# 2. Prompt Template pour le LLM Judge
# ------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """
Vous êtes un expert en évaluation de systèmes RAG (Retrieval-Augmented Generation). 
Votre rôle est d'évaluer la performance d'un agent RAG en comparant sa réponse et ses sources récupérées avec la vérité terrain (Ground Truth).

### Données d'évaluation :
- **Question posée :** {question}
- **Réponse attendue (Ground Truth Answer) :** {ground_truth_answer}
- **Source attendue (Ground Truth Source) :** {ground_truth_source}
- **Réponse générée par l'Agent :** {agent_response}
- **Sources récupérées par le Retriever :** {retrieved_sources}

### Consignes d'évaluation (Grille de notation 0 à 5) :
1. **retriever_score (0-5) :** Comparez 'Source attendue' et 'Sources récupérées'. Les bons segments/tableaux ont-ils été extraits ?
2. **relevance_score (0-5) :** La réponse générée s'adresse-t-elle bien au sujet de la question ?
3. **correctness_score (0-5) :** La réponse générée est-elle vraie par rapport à la 'Réponse attendue' ? (Pour les questions pièges/hors périmètre, l'agent doit admettre son ignorance pour avoir 5/5).
4. **faithfulness_score (0-5) :** L'agent a-t-il inventé des faits (hallucination) ? 5 = 100% fidèle / sans hallucination, 0 = invention pure.
5. **precision_score (0-5) :** La réponse est-elle claire, directe et concise ?

Évaluez ces éléments et fournissez votre analyse structurée.
"""

# ------------------------------------------------------------------
# 3. Initialisation du Judge avec Output Structuré
# ------------------------------------------------------------------


# Application de la structure Pydantic au LLM
structured_judge = llm.with_structured_output(EvaluationOutput)

prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
eval_chain = prompt | structured_judge

# ------------------------------------------------------------------
# 4. Boucle d'Éécution sur le Jeu de Données
# ------------------------------------------------------------------
API_URL = "http://localhost:8000/chat"
from .dataset_eval import EVAL_DATASET  # Import du dataset

results = []

print("🚀 Démarrage de l'évaluation LLM-as-a-Judge...\n")

for item in EVAL_DATASET:
    thread_id = str(uuid.uuid4())
    payload = {"message": item["question"], "thread_id": thread_id}
    
    # A. Interrogation du RAG local
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        rag_data = response.json()
        agent_response = rag_data.get("response", "")
        sources = rag_data.get("sources", [])
        retrieved_sources_str = ", ".join([f"{s.get('section')}/{s.get('sub_section')} ({s.get('type')})" for s in sources])
    except Exception as e:
        print(f"❌ Erreur API sur la question {item['id']}: {e}")
        continue

    # B. Évaluation par le LLM-as-a-Judge
    eval_inputs = {
        "question": item["question"],
        "ground_truth_answer": item["ground_truth_answer"],
        "ground_truth_source": item["ground_truth_source"],
        "agent_response": agent_response,
        "retrieved_sources": retrieved_sources_str if retrieved_sources_str else "Aucune source"
    }
    
    eval_result: EvaluationOutput = eval_chain.invoke(eval_inputs)
    
    # C. Stockage des métriques
    results.append({
        "ID": item["id"],
        "Catégorie": item["category"],
        "Question": item["question"],
        "Réponse Agent": agent_response,
        "Retriever (0-5)": eval_result.retriever_score,
        "Relevance (0-5)": eval_result.relevance_score,
        "Correctness (0-5)": eval_result.correctness_score,
        "Faithfulness/No-Hallu (0-5)": eval_result.faithfulness_score,
        "Precision (0-5)": eval_result.precision_score,
        "Explication Juge": eval_result.explanation
    })
    
    print(f"✅ Question {item['id']} évaluée -> Correctness: {eval_result.correctness_score}/5 | Retriever: {eval_result.retriever_score}/5")

# ------------------------------------------------------------------
# 5. Affichage du Tableau Synthétique
# ------------------------------------------------------------------
df_eval = pd.DataFrame(results)

# Calcul des moyennes globales
print("\n=== MOYENNES DES SCORES RAG ===")
print(f"Retriever Score      : {df_eval['Retriever (0-5)'].mean():.2f} / 5")
print(f"Relevance Score      : {df_eval['Relevance (0-5)'].mean():.2f} / 5")
print(f"Correctness Score    : {df_eval['Correctness (0-5)'].mean():.2f} / 5")
print(f"Faithfulness Score   : {df_eval['Faithfulness/No-Hallu (0-5)'].mean():.2f} / 5")
print(f"Precision Score      : {df_eval['Precision (0-5)'].mean():.2f} / 5")

# Export des résultats en CSV pour le rapport
df_eval.to_csv("rag_eval_results.csv", index=False)

# 1. Générer la chaîne au format Markdown
markdown_table = df_eval.to_markdown(index=False)

# 2. Sauvegarder dans un fichier Markdown séparé (ex: EVALUATION.md)
with open("EVALUATION.md", "w", encoding="utf-8") as f:
    f.write("# 📊 Rapport d'Évaluation RAG (LLM-as-a-Judge)\n\n")
    f.write("### Moyennes des Scores\n")
    f.write(f"- **Retriever Score :** {df_eval['Retriever (0-5)'].mean():.2f} / 5\n")
    f.write(f"- **Relevance Score :** {df_eval['Relevance (0-5)'].mean():.2f} / 5\n")
    f.write(f"- **Correctness Score :** {df_eval['Correctness (0-5)'].mean():.2f} / 5\n")
    f.write(f"- **Faithfulness Score :** {df_eval['Faithfulness/No-Hallu (0-5)'].mean():.2f} / 5\n")
    f.write(f"- **Precision Score :** {df_eval['Precision (0-5)'].mean():.2f} / 5\n\n")
    f.write("### Résultats Détaillés par Question\n\n")
    f.write(markdown_table)

print("✅ Rapport généré avec succès dans 'EVALUATION.md' !")