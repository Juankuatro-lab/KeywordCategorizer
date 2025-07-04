import streamlit as st
import pandas as pd
import re
import json
import io
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import unicodedata
import numpy as np

# Imports pour NLP avancé
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

class AdvancedKeywordCategorizer:
    """
    Catégoriseur avancé avec IA qui préserve la structure du fichier d'entrée
    """
    
    def __init__(self):
        self.categories = {}
        self.use_advanced_nlp = False
        self.model = None
        self.tfidf_vectorizer = None
        
        # Tentative d'initialisation du modèle NLP
        if NLP_AVAILABLE:
            try:
                st.info("Chargement du modèle de langue avancé...")
                # Modèle multilingue optimisé
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.tfidf_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=1000,
                    stop_words=None,
                    lowercase=True
                )
                self.use_advanced_nlp = True
                st.success("✅ Modèle NLP avancé chargé avec succès!")
            except Exception as e:
                st.warning(f"⚠️ Impossible de charger le modèle NLP avancé: {str(e)}")
                st.info("📝 Utilisation de l'algorithme de base")
                self.use_advanced_nlp = False
        else:
            st.warning("⚠️ Dépendances NLP manquantes. Installation recommandée :")
            st.code("pip install sentence-transformers scikit-learn torch")
            st.info("📝 Utilisation de l'algorithme de base")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalise le texte pour une meilleure comparaison
        """
        # Convertir en minuscules
        text = text.lower().strip()
        # Supprimer les accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Supprimer la ponctuation et caractères spéciaux
        text = re.sub(r'[^\w\s]', ' ', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_similarity_basic(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes avec l'algorithme de base
        """
        text1_norm = self.normalize_text(text1)
        text2_norm = self.normalize_text(text2)
        
        # Similarité de séquence globale
        seq_similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        # Similarité par mots
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if len(words1) == 0 and len(words2) == 0:
            word_similarity = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            word_similarity = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_similarity = intersection / union if union > 0 else 0.0
        
        # Similarité de contenance (un mot contient l'autre)
        containment_similarity = 0.0
        if text1_norm in text2_norm or text2_norm in text1_norm:
            containment_similarity = 0.8
        
        # Score final pondéré
        final_score = (seq_similarity * 0.4 + word_similarity * 0.4 + containment_similarity * 0.2)
        
        return final_score
    
    def calculate_similarity_advanced(self, keyword: str, category_terms: List[str]) -> Tuple[str, float]:
        """
        Calcule la similarité avec des embeddings sémantiques avancés
        """
        try:
            # Créer les embeddings
            keyword_embedding = self.model.encode([keyword])
            terms_embeddings = self.model.encode(category_terms)
            
            # Calculer la similarité cosinus
            similarities = cosine_similarity(keyword_embedding, terms_embeddings)[0]
            
            # Trouver le meilleur match
            best_idx = np.argmax(similarities)
            best_term = category_terms[best_idx]
            best_score = similarities[best_idx]
            
            # Bonus pour correspondance exacte ou partielle
            keyword_norm = self.normalize_text(keyword)
            for i, term in enumerate(category_terms):
                term_norm = self.normalize_text(term)
                
                # Correspondance exacte
                if keyword_norm == term_norm:
                    return term, 1.0
                
                # Correspondance partielle forte
                if keyword_norm in term_norm or term_norm in keyword_norm:
                    if similarities[i] + 0.3 > best_score:
                        best_score = min(1.0, similarities[i] + 0.3)
                        best_term = term
                        best_idx = i
            
            return best_term, float(best_score)
            
        except Exception as e:
            st.warning(f"Erreur dans l'analyse avancée: {str(e)}")
            # Fallback vers l'algorithme de base
            best_score = 0.0
            best_term = category_terms[0]
            
            for term in category_terms:
                score = self.calculate_similarity_basic(keyword, term)
                if score > best_score:
                    best_score = score
                    best_term = term
            
            return best_term, best_score
    
    def calculate_tfidf_similarity(self, keyword: str, all_terms: List[str]) -> Dict[str, float]:
        """
        Calcule la similarité TF-IDF pour une analyse complémentaire
        """
        try:
            # Préparer le corpus
            corpus = [keyword] + all_terms
            
            # Vectorisation TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Similarité cosinus
            keyword_vector = tfidf_matrix[0:1]
            terms_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(keyword_vector, terms_vectors)[0]
            
            # Créer le dictionnaire de scores
            scores = {}
            for i, term in enumerate(all_terms):
                scores[term] = float(similarities[i])
                
            return scores
            
        except Exception as e:
            # Retour aux scores de base
            return {term: self.calculate_similarity_basic(keyword, term) for term in all_terms}
    
    def set_categories(self, categories_dict: Dict[str, List[str]]):
        """
        Définit les catégories avec leurs termes de référence
        """
        self.categories = categories_dict
    
    def categorize_keyword(self, keyword: str) -> Tuple[str, str, float]:
        """
        Trouve OBLIGATOIREMENT la meilleure catégorie pour un mot-clé avec IA avancée
        """
        if not self.categories:
            return ("Aucune catégorie", "Non défini", 0.0)
        
        best_category = None
        best_term = None
        best_score = -1.0
        
        if self.use_advanced_nlp:
            # Approche IA avancée
            try:
                # Préparer tous les termes pour l'analyse globale
                all_terms = []
                term_to_category = {}
                
                for category_name, terms_list in self.categories.items():
                    for term in terms_list:
                        all_terms.append(term)
                        term_to_category[term] = category_name
                
                # Analyse sémantique par catégorie
                category_scores = {}
                
                for category_name, terms_list in self.categories.items():
                    matched_term, score = self.calculate_similarity_advanced(keyword, terms_list)
                    category_scores[category_name] = (matched_term, score)
                
                # Analyse TF-IDF complémentaire
                tfidf_scores = self.calculate_tfidf_similarity(keyword, all_terms)
                
                # Combinaison des scores
                for category_name, (matched_term, semantic_score) in category_scores.items():
                    # Score TF-IDF pour ce terme
                    tfidf_score = tfidf_scores.get(matched_term, 0.0)
                    
                    # Score final combiné (70% sémantique, 30% TF-IDF)
                    combined_score = (semantic_score * 0.7) + (tfidf_score * 0.3)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_category = category_name
                        best_term = matched_term
                
            except Exception as e:
                st.warning(f"Erreur dans l'analyse IA: {str(e)}")
                # Fallback vers l'algorithme de base
                self.use_advanced_nlp = False
        
        if not self.use_advanced_nlp:
            # Approche de base en cas de problème
            for category_name, terms_list in self.categories.items():
                for term in terms_list:
                    similarity = self.calculate_similarity_basic(keyword, term)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_category = category_name
                        best_term = term
        
        # Si aucune similarité trouvée, attribuer à la première catégorie
        if best_category is None:
            first_category = list(self.categories.keys())[0]
            first_term = self.categories[first_category][0] if self.categories[first_category] else "Non défini"
            return (first_category, first_term, 0.0)
        
        return (best_category, best_term, best_score)
    
    def process_dataframe_preserve_structure(self, df: pd.DataFrame, keyword_column: str, 
                                           output_columns_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Traite le DataFrame en préservant EXACTEMENT la structure originale
        Ne modifie QUE les colonnes spécifiées dans output_columns_mapping
        """
        # Créer une copie exacte du DataFrame original
        result_df = df.copy()
        
        # Vérifier que la colonne des mots-clés existe
        if keyword_column not in df.columns:
            raise ValueError(f"Colonne '{keyword_column}' non trouvée dans le fichier")
        
        # Vérifier que toutes les colonnes de sortie existent
        for output_col in output_columns_mapping.values():
            if output_col not in df.columns:
                raise ValueError(f"Colonne de sortie '{output_col}' non trouvée dans le fichier")
        
        # Créer un DataFrame de statistiques séparé
        stats_data = []
        
        # Traitement ligne par ligne
        for idx, row in df.iterrows():
            keyword = str(row[keyword_column])
            
            # Traiter même les valeurs vides
            if pd.isna(keyword) or keyword.strip() == '' or keyword.lower() == 'nan':
                keyword = f"Vide_ligne_{idx+1}"
            
            # Catégorisation
            category, matched_term, confidence = self.categorize_keyword(keyword)
            
            # Mettre à jour UNIQUEMENT les colonnes spécifiées
            for category_name, output_column in output_columns_mapping.items():
                if category_name == category:
                    result_df.at[idx, output_column] = matched_term
                else:
                    # Laisser vide ou mettre une valeur par défaut si ce n'est pas la bonne catégorie
                    if pd.isna(result_df.at[idx, output_column]) or str(result_df.at[idx, output_column]).strip() == '':
                        result_df.at[idx, output_column] = ''
            
            # Collecter les statistiques
            stats_data.append({
                'Ligne': idx + 1,
                'Mot_cle_original': keyword,
                'Categorie_attribuee': category,
                'Terme_reference': matched_term,
                'Score_similarite': round(confidence, 3)
            })
        
        # Créer le DataFrame de statistiques
        stats_df = pd.DataFrame(stats_data)
        
        return result_df, stats_df


def main():
    st.set_page_config(
        page_title="Script de catégorisation de mots-clés",
        page_icon="https://jc-espinosa.com/wp-content/uploads/2024/06/JC-Espinosa-Favicon-blue-150x150.png",
        layout="wide"
    )
    
    st.title("Script de catégorisation de mots-clés")
    st.markdown("**Préserve exactement** la mise en forme de votre fichier d'entrée, ne modifie que les colonnes spécifiées")
    
    # Encart d'informations et étapes d'utilisation
    with st.expander("Informations et étapes d'utilisation"):
        st.markdown("""
        ### Description de l'outil
        
        Cet outil utilise l'Intelligence Artificielle pour catégoriser automatiquement des mots-clés en préservant exactement la structure de votre fichier d'entrée. 
        Il combine plusieurs techniques NLP pour une catégorisation précise et contextuelle.
        
        ### Technologies utilisées
        
        - **Embeddings sémantiques** : Modèle Sentence-BERT multilingue pour comprendre le sens,
        - **Similarité cosinus** : Mesure de proximité sémantique dans l'espace vectoriel,
        - **TF-IDF** : Analyse de fréquence des termes pour affiner les résultats,
        - **Algorithme hybride** : Combinaison de 3 approches pour maximiser la précision.
        
        ### Fonctionnement
        
        - **Attribution forcée** : Chaque mot-clé est obligatoirement attribué à une catégorie,
        - **Préservation de structure** : Votre fichier garde exactement sa mise en forme originale,
        - **IA contextuelle** : Comprend le sens et le contexte, pas seulement les mots-clés,
        - **Multilingue** : Fonctionne en français, anglais et autres langues européennes,
        - **Colonnes ciblées** : Seules les colonnes que vous spécifiez sont modifiées.
        
        ### Étapes d'utilisation
        
        1. **Configurez vos catégories** : Créez vos catégories et ajoutez les termes de référence pour chacune,
        2. **Importez votre fichier** : Chargez votre fichier Excel ou CSV avec la structure finale souhaitée,
        3. **Configurez les colonnes** : Sélectionnez la colonne des mots-clés et assignez les colonnes de sortie pour chaque catégorie,
        4. **Lancez le traitement IA** : L'outil analyse sémantiquement tous vos mots-clés et les place dans les bonnes colonnes,
        5. **Exportez les résultats** : Téléchargez votre fichier final avec les statistiques de confiance.
        
        ### Format de fichier attendu
        
        Votre fichier d'entrée doit contenir :
        - Une colonne avec vos mots-clés à catégoriser,
        - Des colonnes vides ou existantes où placer les résultats de chaque catégorie,
        - Toute autre donnée que vous souhaitez conserver (elle sera préservée).
        
        ### Scores de confiance
        
        - **0.8-1.0** : Correspondance excellente (IA très confiante),
        - **0.6-0.8** : Correspondance bonne (contexte sémantique fort),
        - **0.4-0.6** : Correspondance acceptable (similarité détectée),
        - **0.0-0.4** : Correspondance faible (attribution par défaut).
        """)
    
    # CSS pour les boutons en couleur fcf192
    st.markdown("""
    <style>
    .stButton > button[kind="primary"] {
        background-color: #fcf192 !important;
        color: #000000 !important;
        border: 1px solid #e6e600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #f0e85c !important;
        color: #000000 !important;
        border: 1px solid #cccc00 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialisation
    if 'categorizer' not in st.session_state:
        st.session_state.categorizer = AdvancedKeywordCategorizer()
        st.session_state.categories_config = {}
        st.session_state.df_loaded = None
        st.session_state.columns_mapping = {}
    
    # Étape 1: Configuration des catégories
    st.header("Étape 1 : Configuration des catégories")
    
    col_config1, col_config2 = st.columns([1, 1])
    
    with col_config1:
        st.subheader("Ajouter une catégorie")
        
        new_category_name = st.text_input(
            "Nom de la catégorie :",
            placeholder="Ex: Produits, Services, Marques, Thématiques..."
        )
        
        new_category_terms = st.text_area(
            "Termes de référence pour cette catégorie (un par ligne) :",
            placeholder="smartphone\ntablette\nordinateur\n...",
            height=120
        )
        
        if st.button("Ajouter cette catégorie", type="primary"):
            if new_category_name and new_category_terms:
                # Vérifier si la catégorie existe déjà
                if new_category_name in st.session_state.categories_config:
                    st.warning(f"La catégorie '{new_category_name}' existe déjà. Utilisez un autre nom.")
                else:
                    terms_list = [term.strip() for term in new_category_terms.split('\n') if term.strip()]
                    if len(terms_list) > 0:
                        st.session_state.categories_config[new_category_name] = terms_list
                        st.session_state.categorizer.set_categories(st.session_state.categories_config)
                        st.success(f"Catégorie '{new_category_name}' ajoutée avec {len(terms_list)} termes !")
                        # Forcer le rafraîchissement
                        st.rerun()
                    else:
                        st.error("Veuillez ajouter au moins un terme de référence")
            else:
                st.error("Veuillez remplir le nom et les termes de référence")
        
        # Bouton pour vider les champs
        if st.button("Vider les champs"):
            st.rerun()
    
    with col_config2:
        st.subheader("Catégories configurées")
        
        if st.session_state.categories_config:
            for cat_name, terms in st.session_state.categories_config.items():
                with st.expander(f"{cat_name} ({len(terms)} termes)"):
                    st.write("**Termes de référence :**")
                    for i, term in enumerate(terms[:8], 1):
                        st.write(f"{i}. {term}")
                    if len(terms) > 8:
                        st.write(f"... et {len(terms) - 8} autres termes")
                    
                    if st.button(f"Supprimer {cat_name}", key=f"del_{cat_name}"):
                        del st.session_state.categories_config[cat_name]
                        st.session_state.categorizer.set_categories(st.session_state.categories_config)
                        st.rerun()
        else:
            st.info("Aucune catégorie configurée. Ajoutez au moins une catégorie pour commencer.")
    
    st.divider()
    
    # Étape 2: Import du fichier
    st.header("Étape 2 : Import du fichier structure")
    
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier Excel ou CSV avec la structure finale souhaitée",
        type=['xlsx', 'xls', 'csv'],
        help="Ce fichier doit contenir : colonne des mots-clés + colonnes vides pour chaque catégorie"
    )
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df_loaded = df
            st.success(f"Fichier chargé : **{len(df):,} lignes** et **{len(df.columns)} colonnes**")
            
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
            return
    
    # ÉTAPE 3: Configuration des colonnes
    if st.session_state.df_loaded is not None and st.session_state.categories_config:
        st.header("Étape 3 : Configuration des colonnes")
        
        df = st.session_state.df_loaded
        
        col_mapping1, col_mapping2 = st.columns([1, 1])
        
        with col_mapping1:
            st.subheader("Colonne des mots-clés")
            
            keyword_column = st.selectbox(
                "Sélectionnez la colonne contenant les mots-clés à catégoriser :",
                options=df.columns.tolist(),
                help="Cette colonne contient les termes que vous voulez catégoriser"
            )
            
            # Aperçu de la colonne sélectionnée
            if keyword_column:
                st.write("**Aperçu des mots-clés :**")
                sample_keywords = df[keyword_column].dropna().head(10).tolist()
                for i, kw in enumerate(sample_keywords, 1):
                    st.write(f"{i}. {kw}")
                if len(df[keyword_column].dropna()) > 10:
                    st.write(f"... et {len(df[keyword_column].dropna()) - 10} autres")
        
        with col_mapping2:
            st.subheader("Attribution des colonnes de sortie")
            
            st.write("Pour chaque catégorie, choisissez dans quelle colonne mettre le résultat :")
            
            columns_mapping = {}
            
            for category_name in st.session_state.categories_config.keys():
                output_column = st.selectbox(
                    f"Colonne pour '{category_name}' :",
                    options=df.columns.tolist(),
                    key=f"mapping_{category_name}",
                    help=f"Les termes de la catégorie '{category_name}' seront placés dans cette colonne"
                )
                columns_mapping[category_name] = output_column
            
            st.session_state.columns_mapping = columns_mapping
            
            # Validation
            if len(set(columns_mapping.values())) != len(columns_mapping.values()):
                st.warning("Attention : Vous avez assigné plusieurs catégories à la même colonne !")
        
        # Aperçu de la structure
        st.subheader("Aperçu de la structure du fichier")
        st.dataframe(df.head(), use_container_width=True)
        
        # Résumé de la configuration
        st.subheader("Résumé de la configuration")
        
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.write("**Configuration :**")
            st.write(f"• Colonne mots-clés : `{keyword_column}`")
            st.write(f"• Nombre de lignes : `{len(df):,}`")
            st.write(f"• Catégories configurées : `{len(st.session_state.categories_config)}`")
        
        with col_summary2:
            st.write("**Attribution des colonnes :**")
            for cat, col in columns_mapping.items():
                st.write(f"• {cat} → `{col}`")
        
        st.divider()
        
        # Étape 4: Traitement
        st.header("Étape 4 : Traitement")
        
        if st.button("Lancer la catégorisation", type="primary", use_container_width=True):
            if not columns_mapping:
                st.error("Veuillez configurer l'attribution des colonnes")
                return
            
            with st.spinner("Traitement en cours... Préservation de la structure originale..."):
                try:
                    # Traitement avec préservation de structure
                    result_df, stats_df = st.session_state.categorizer.process_dataframe_preserve_structure(
                        df, keyword_column, columns_mapping
                    )
                    
                    # Sauvegarde dans la session
                    st.session_state.result_df = result_df
                    st.session_state.stats_df = stats_df
                    
                    st.success("Traitement terminé ! Structure originale préservée.")
                    
                    # Statistiques rapides
                    total_processed = len(stats_df)
                    categories_stats = stats_df['Categorie_attribuee'].value_counts()
                    avg_confidence = stats_df['Score_similarite'].mean()
                    
                    col_stats = st.columns(3)
                    with col_stats[0]:
                        st.metric("Mots-clés traités", f"{total_processed:,}")
                    with col_stats[1]:
                        st.metric("Score moyen", f"{avg_confidence:.3f}")
                    with col_stats[2]:
                        high_conf = len(stats_df[stats_df['Score_similarite'] >= 0.5])
                        st.metric("Haute confiance", f"{high_conf:,}")
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement : {str(e)}")
        
        # Étape 5: Résultats et export
        if 'result_df' in st.session_state:
            st.header("Étape 5 : Résultats et Export")
            
            result_df = st.session_state.result_df
            stats_df = st.session_state.stats_df
            
            # Onglets pour organiser l'affichage
            tab1, tab2, tab3 = st.tabs(["Fichier Final", "Statistiques", "Export"])
            
            with tab1:
                st.subheader("Aperçu du fichier final (structure préservée)")
                st.write("**Structure exactement identique à l'entrée**, seules les colonnes assignées ont été modifiées :")
                
                # Afficher avec mise en évidence des colonnes modifiées
                display_df = result_df.copy()
                
                # Mettre en évidence les colonnes modifiées
                modified_columns = list(st.session_state.columns_mapping.values())
                st.write(f"**Colonnes modifiées :** {', '.join(modified_columns)}")
                
                # Sélecteur pour le nombre de lignes à afficher
                show_rows = st.selectbox("Lignes à afficher :", [10, 50, 100, 500], index=1)
                st.dataframe(display_df.head(show_rows), use_container_width=True)
                
                st.write(f"**Total :** {len(result_df):,} lignes × {len(result_df.columns)} colonnes")
            
            with tab2:
                st.subheader("Statistiques détaillées")
                
                # Répartition par catégorie
                categories_stats = stats_df['Categorie_attribuee'].value_counts()
                st.bar_chart(categories_stats)
                
                # Distribution des scores
                st.subheader("Distribution des scores de similarité")
                
                # Créer les bins pour l'histogramme sans matplotlib
                bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                hist_data = pd.cut(stats_df['Score_similarite'], bins=bins, include_lowest=True).value_counts().sort_index()
                
                # Renommer les index pour plus de clarté
                hist_data.index = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in hist_data.index]
                
                st.bar_chart(hist_data)
                
                # Détails par catégorie
                st.subheader("Détails par catégorie")
                for category in categories_stats.index:
                    with st.expander(f"{category} ({categories_stats[category]} éléments)"):
                        cat_data = stats_df[stats_df['Categorie_attribuee'] == category]
                        avg_score = cat_data['Score_similarite'].mean()
                        st.write(f"**Score moyen :** {avg_score:.3f}")
                        
                        # Exemples avec scores les plus élevés
                        top_matches = cat_data.nlargest(5, 'Score_similarite')
                        st.write("**Meilleurs matches :**")
                        for _, row in top_matches.iterrows():
                            st.write(f"• `{row['Mot_cle_original']}` → `{row['Terme_reference']}` (score: {row['Score_similarite']:.3f})")
            
            with tab3:
                st.subheader("Export des résultats")
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    st.write("**Fichier principal (structure préservée)**")
                    
                    # Export Excel du fichier final
                    output_main = io.BytesIO()
                    result_df.to_excel(output_main, index=False, engine='openpyxl')
                    excel_main_data = output_main.getvalue()
                    
                    st.download_button(
                        label="Télécharger fichier final (.xlsx)",
                        data=excel_main_data,
                        file_name=f"fichier_categorise_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # Export CSV
                    csv_main_data = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="Télécharger fichier final (.csv)",
                        data=csv_main_data,
                        file_name=f"fichier_categorise_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_export2:
                    st.write("**Rapport d'analyse**")
                    
                    # Export Excel complet avec statistiques
                    output_complete = io.BytesIO()
                    with pd.ExcelWriter(output_complete, engine='openpyxl') as writer:
                        # Fichier final
                        result_df.to_excel(writer, sheet_name='Fichier_final', index=False)
                        
                        # Statistiques détaillées
                        stats_df.to_excel(writer, sheet_name='Statistiques_detail', index=False)
                        
                        # Synthèse
                        synthesis_data = []
                        for category in categories_stats.index:
                            cat_data = stats_df[stats_df['Categorie_attribuee'] == category]
                            synthesis_data.append({
                                'Categorie': category,
                                'Nombre_elements': len(cat_data),
                                'Score_moyen': cat_data['Score_similarite'].mean(),
                                'Pourcentage': (len(cat_data) / len(stats_df)) * 100,
                                'Colonne_assignee': st.session_state.columns_mapping.get(category, 'Non assigné')
                            })
                        
                        synthesis_df = pd.DataFrame(synthesis_data)
                        synthesis_df.to_excel(writer, sheet_name='Synthese', index=False)
                        
                        # Configuration
                        config_data = []
                        for cat_name, terms in st.session_state.categories_config.items():
                            for term in terms:
                                config_data.append({
                                    'Categorie': cat_name,
                                    'Terme_reference': term,
                                    'Colonne_sortie': st.session_state.columns_mapping.get(cat_name, 'Non assigné')
                                })
                        
                        config_df = pd.DataFrame(config_data)
                        config_df.to_excel(writer, sheet_name='Configuration', index=False)
                    
                    excel_complete_data = output_complete.getvalue()
                    
                    st.download_button(
                        label="Télécharger rapport complet (.xlsx)",
                        data=excel_complete_data,
                        file_name=f"rapport_complet_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                # Informations sur l'export
                st.info("""
                **Contenu des exports :**
                • **Fichier final** : Structure exactement identique à votre fichier d'entrée avec les colonnes catégorisées
                • **Rapport complet** : Fichier final + statistiques + configuration pour traçabilité
                """)


if __name__ == "__main__":
    main()
