import streamlit as st
import pandas as pd
import re
import json
import io
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class KeywordCategorizer:
    """
    Classe pour catégoriser des mots-clés basée sur des règles définies
    """
    
    def __init__(self):
        """
        Initialise le catégoriseur
        """
        self.categories = {}
        self.setup_default_categories()
    
    def setup_default_categories(self):
        """
        Configure des catégories par défaut comme exemple
        """
        self.categories = {
            "E-commerce": {
                "Produits": ["produit", "article", "vente", "achat", "boutique", "magasin"],
                "Paiement": ["paiement", "carte", "paypal", "transaction", "facture"],
                "Livraison": ["livraison", "expédition", "transport", "colis", "délai"]
            },
            "Marketing": {
                "SEO": ["seo", "référencement", "google", "moteur", "recherche", "ranking"],
                "Social Media": ["facebook", "instagram", "twitter", "linkedin", "social", "réseau"],
                "Publicité": ["pub", "publicité", "campagne", "annonce", "marketing"]
            },
            "Technologie": {
                "Développement": ["dev", "développement", "code", "programmation", "software"],
                "Infrastructure": ["serveur", "hosting", "cloud", "infrastructure", "réseau"],
                "Sécurité": ["sécurité", "protection", "antivirus", "firewall", "encryption"]
            }
        }
    
    def load_config_from_dict(self, config_dict: dict):
        """
        Charge la configuration depuis un dictionnaire
        """
        self.categories = config_dict.get('categories', {})
    
    def add_category(self, main_category: str, sub_category: str, keywords: List[str]):
        """
        Ajoute une nouvelle catégorie avec ses mots-clés
        """
        if main_category not in self.categories:
            self.categories[main_category] = {}
        
        self.categories[main_category][sub_category] = keywords
    
    def categorize_keyword(self, keyword: str) -> Tuple[str, str, float]:
        """
        Catégorise un mot-clé et retourne (catégorie_principale, sous_catégorie, score_confiance)
        """
        keyword_lower = keyword.lower().strip()
        best_match = ("Non classé", "Autre", 0.0)
        
        for main_cat, sub_cats in self.categories.items():
            for sub_cat, cat_keywords in sub_cats.items():
                for cat_keyword in cat_keywords:
                    # Recherche exacte
                    if cat_keyword.lower() == keyword_lower:
                        return (main_cat, sub_cat, 1.0)
                    
                    # Recherche partielle
                    if cat_keyword.lower() in keyword_lower:
                        score = len(cat_keyword) / len(keyword_lower)
                        if score > best_match[2]:
                            best_match = (main_cat, sub_cat, score)
                    
                    # Recherche avec regex pour plus de flexibilité
                    pattern = re.compile(r'\b' + re.escape(cat_keyword.lower()) + r'\b')
                    if pattern.search(keyword_lower):
                        score = 0.8  # Score élevé pour correspondance de mot complet
                        if score > best_match[2]:
                            best_match = (main_cat, sub_cat, score)
        
        return best_match
    
    def process_dataframe(self, df: pd.DataFrame, keyword_column: str) -> pd.DataFrame:
        """
        Traite un DataFrame et ajoute les colonnes de catégorisation
        """
        # Vérification de la colonne
        if keyword_column not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(f"Colonne '{keyword_column}' non trouvée. Colonnes disponibles: {available_cols}")
        
        # Traitement des mots-clés
        results = []
        for _, row in df.iterrows():
            keyword = str(row[keyword_column])
            if pd.notna(keyword) and keyword.strip():
                main_cat, sub_cat, confidence = self.categorize_keyword(keyword)
                
                # Création d'une ligne de résultat
                result_row = {
                    'Mot-clé': keyword,
                    'Catégorie principale': main_cat,
                    'Sous-catégorie': sub_cat,
                    'Score de confiance': round(confidence, 2)
                }
                
                # Ajout des autres colonnes du DataFrame original
                for col in df.columns:
                    if col != keyword_column:
                        result_row[f'Original_{col}'] = row[col]
                
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Retourne des statistiques sur la catégorisation
        """
        stats = {
            'total_keywords': len(df),
            'main_categories': df['Catégorie principale'].value_counts().to_dict(),
            'avg_confidence': df['Score de confiance'].mean(),
            'unclassified': df[df['Catégorie principale'] == 'Non classé']['Mot-clé'].tolist()
        }
        return stats


def main():
    """
    Application Streamlit principale
    """
    st.set_page_config(
        page_title="Catégoriseur de mots-clés",
        page_icon="🏷️",
        layout="wide"
    )
    
    st.title("🏷️ Catégoriseur de mots-clés")
    st.markdown("Uploadez votre fichier Excel ou CSV pour catégoriser automatiquement vos mots-clés.")
    
    # Initialisation du catégoriseur dans la session
    if 'categorizer' not in st.session_state:
        st.session_state.categorizer = KeywordCategorizer()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gestion des catégories
        with st.expander("🏷️ Gérer les catégories"):
            st.subheader("Ajouter une catégorie")
            new_main_cat = st.text_input("Catégorie principale", key="new_main")
            new_sub_cat = st.text_input("Sous-catégorie", key="new_sub")
            new_keywords = st.text_area("Mots-clés (un par ligne)", key="new_keywords")
            
            if st.button("Ajouter catégorie"):
                if new_main_cat and new_sub_cat and new_keywords:
                    keywords_list = [kw.strip() for kw in new_keywords.split('\n') if kw.strip()]
                    st.session_state.categorizer.add_category(new_main_cat, new_sub_cat, keywords_list)
                    st.success(f"Catégorie '{new_main_cat} > {new_sub_cat}' ajoutée!")
                    st.rerun()
                else:
                    st.error("Veuillez remplir tous les champs")
        
        # Affichage des catégories actuelles
        with st.expander("📋 Catégories actuelles"):
            for main_cat, sub_cats in st.session_state.categorizer.categories.items():
                st.write(f"**{main_cat}**")
                for sub_cat, keywords in sub_cats.items():
                    st.write(f"  - {sub_cat}: {', '.join(keywords[:5])}")
                    if len(keywords) > 5:
                        st.write(f"    ... et {len(keywords) - 5} autres")
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload du fichier")
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier",
            type=['xlsx', 'xls', 'csv'],
            help="Formats supportés: Excel (.xlsx, .xls) et CSV"
        )
        
        if uploaded_file is not None:
            try:
                # Lecture du fichier
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Fichier chargé: {len(df)} lignes")
                
                # Sélection de la colonne des mots-clés
                keyword_column = st.selectbox(
                    "Colonne contenant les mots-clés:",
                    options=df.columns.tolist(),
                    help="Sélectionnez la colonne qui contient vos mots-clés"
                )
                
                # Aperçu des données
                st.subheader("📊 Aperçu des données")
                st.dataframe(df.head(), use_container_width=True)
                
                # Bouton de traitement
                if st.button("🚀 Lancer la catégorisation", type="primary"):
                    with st.spinner("Traitement en cours..."):
                        try:
                            # Traitement
                            results_df = st.session_state.categorizer.process_dataframe(df, keyword_column)
                            
                            # Sauvegarde dans la session
                            st.session_state.results_df = results_df
                            st.session_state.stats = st.session_state.categorizer.get_statistics(results_df)
                            
                            st.success("Catégorisation terminée!")
                            
                        except Exception as e:
                            st.error(f"Erreur lors du traitement: {str(e)}")
                            
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    
    with col2:
        st.header("📊 Résultats")
        
        if 'results_df' in st.session_state:
            results_df = st.session_state.results_df
            stats = st.session_state.stats
            
            # Métriques
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("Total mots-clés", stats['total_keywords'])
            with col_metrics[1]:
                st.metric("Score moyen", f"{stats['avg_confidence']:.2f}")
            with col_metrics[2]:
                st.metric("Non classés", len(stats['unclassified']))
            
            # Graphique des catégories
            st.subheader("📈 Répartition par catégorie")
            main_cats_df = pd.DataFrame(
                list(stats['main_categories'].items()),
                columns=['Catégorie', 'Nombre']
            )
            st.bar_chart(main_cats_df.set_index('Catégorie'))
            
            # Résultats détaillés
            st.subheader("📋 Résultats détaillés")
            
            # Filtres
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                filter_category = st.selectbox(
                    "Filtrer par catégorie:",
                    options=["Toutes"] + list(stats['main_categories'].keys())
                )
            with filter_col2:
                min_confidence = st.slider(
                    "Score minimum:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            # Application des filtres
            filtered_df = results_df.copy()
            if filter_category != "Toutes":
                filtered_df = filtered_df[filtered_df['Catégorie principale'] == filter_category]
            filtered_df = filtered_df[filtered_df['Score de confiance'] >= min_confidence]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Téléchargement
            st.subheader("💾 Télécharger les résultats")
            
            # Conversion en Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Résultats', index=False)
                
                # Feuille des statistiques
                stats_df = pd.DataFrame([
                    ['Total mots-clés', stats['total_keywords']],
                    ['Score moyen', f"{stats['avg_confidence']:.2f}"],
                    ['Non classés', len(stats['unclassified'])]
                ], columns=['Métrique', 'Valeur'])
                stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
            
            excel_data = output.getvalue()
            
            col_download = st.columns(2)
            with col_download[0]:
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=excel_data,
                    file_name="mots_cles_categorises.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col_download[1]:
                csv_data = filtered_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name="mots_cles_categorises.csv",
                    mime="text/csv"
                )
            
            # Mots-clés non classés
            if stats['unclassified']:
                with st.expander(f"⚠️ Mots-clés non classés ({len(stats['unclassified'])})"):
                    for keyword in stats['unclassified'][:20]:
                        st.write(f"- {keyword}")
                    if len(stats['unclassified']) > 20:
                        st.write(f"... et {len(stats['unclassified']) - 20} autres")
        
        else:
            st.info("Uploadez un fichier et lancez la catégorisation pour voir les résultats ici.")


if __name__ == "__main__":
    main()
