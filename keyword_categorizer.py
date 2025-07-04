import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

class KeywordCategorizer:
    """
    Classe pour catégoriser des mots-clés basée sur des règles définies
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialise le catégoriseur avec un fichier de configuration optionnel
        """
        self.categories = {}
        self.rules = {}
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
        else:
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
    
    def load_config(self, config_file: str):
        """
        Charge la configuration depuis un fichier JSON
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.categories = config.get('categories', {})
                self.rules = config.get('rules', {})
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")
            self.setup_default_categories()
    
    def save_config(self, config_file: str):
        """
        Sauvegarde la configuration dans un fichier JSON
        """
        config = {
            'categories': self.categories,
            'rules': self.rules
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
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
    
    def process_file(self, input_file: str, output_file: str, keyword_column: str = 'A', 
                    additional_columns: List[str] = None):
        """
        Traite un fichier Excel ou CSV et ajoute les colonnes de catégorisation
        """
        # Détection du type de fichier
        file_extension = Path(input_file).suffix.lower()
        
        try:
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(input_file)
            elif file_extension == '.csv':
                df = pd.read_csv(input_file, encoding='utf-8')
            else:
                raise ValueError(f"Format de fichier non supporté: {file_extension}")
            
            # Identification de la colonne des mots-clés
            if keyword_column in df.columns:
                keyword_col = keyword_column
            elif keyword_column.upper() in df.columns:
                keyword_col = keyword_column.upper()
            else:
                # Si la colonne n'existe pas, prendre la première colonne
                keyword_col = df.columns[0]
                print(f"Colonne '{keyword_column}' non trouvée. Utilisation de '{keyword_col}'")
            
            # Traitement des mots-clés
            results = []
            for _, row in df.iterrows():
                keyword = str(row[keyword_col])
                if pd.notna(keyword) and keyword.strip():
                    main_cat, sub_cat, confidence = self.categorize_keyword(keyword)
                    results.append({
                        'Mot-clé': keyword,
                        'Catégorie principale': main_cat,
                        'Sous-catégorie': sub_cat,
                        'Score de confiance': round(confidence, 2),
                        'Données originales': row.to_dict()
                    })
            
            # Création du DataFrame de résultats
            results_df = pd.DataFrame(results)
            
            # Ajout des colonnes supplémentaires si spécifiées
            if additional_columns:
                for col in additional_columns:
                    if col in df.columns:
                        results_df[col] = df[col]
            
            # Sauvegarde
            output_extension = Path(output_file).suffix.lower()
            if output_extension in ['.xlsx', '.xls']:
                results_df.to_excel(output_file, index=False)
            else:
                results_df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"Traitement terminé. Résultats sauvegardés dans: {output_file}")
            self.print_statistics(results_df)
            
            return results_df
            
        except Exception as e:
            print(f"Erreur lors du traitement du fichier: {e}")
            return None
    
    def print_statistics(self, df: pd.DataFrame):
        """
        Affiche des statistiques sur la catégorisation
        """
        print("\n=== STATISTIQUES DE CATÉGORISATION ===")
        print(f"Nombre total de mots-clés traités: {len(df)}")
        
        # Répartition par catégorie principale
        print("\nRépartition par catégorie principale:")
        main_cats = df['Catégorie principale'].value_counts()
        for cat, count in main_cats.items():
            percentage = (count / len(df)) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")
        
        # Scores de confiance moyens
        avg_confidence = df['Score de confiance'].mean()
        print(f"\nScore de confiance moyen: {avg_confidence:.2f}")
        
        # Mots-clés non classés
        unclassified = df[df['Catégorie principale'] == 'Non classé']
        if len(unclassified) > 0:
            print(f"\nMots-clés non classés ({len(unclassified)}):")
            for keyword in unclassified['Mot-clé'].head(10):
                print(f"  - {keyword}")
            if len(unclassified) > 10:
                print(f"  ... et {len(unclassified) - 10} autres")


def main():
    """
    Fonction principale pour utiliser le script
    """
    # Exemple d'utilisation
    categorizer = KeywordCategorizer()
    
    # Optionnel: Ajouter des catégories personnalisées
    categorizer.add_category("Finance", "Banque", ["banque", "crédit", "prêt", "compte"])
    categorizer.add_category("Finance", "Assurance", ["assurance", "garantie", "police", "sinistre"])
    
    # Exemple de traitement d'un fichier
    # categorizer.process_file("mots_cles.xlsx", "mots_cles_categorises.xlsx", "A")
    
    # Sauvegarde de la configuration
    categorizer.save_config("config_categories.json")
    
    print("Script de catégorisation initialisé.")
    print("Utilisez categorizer.process_file('input.xlsx', 'output.xlsx', 'A') pour traiter un fichier.")


if __name__ == "__main__":
    main()


# EXEMPLE D'UTILISATION RAPIDE:
"""
# 1. Initialisation
categorizer = KeywordCategorizer()

# 2. Traitement d'un fichier
df_results = categorizer.process_file(
    input_file="mes_mots_cles.xlsx",
    output_file="mots_cles_categorises.xlsx",
    keyword_column="A",  # ou le nom de la colonne
    additional_columns=["B", "C"]  # colonnes supplémentaires à conserver
)

# 3. Ajout de nouvelles catégories
categorizer.add_category("Voyage", "Transport", ["avion", "train", "bus", "voiture"])
categorizer.add_category("Voyage", "Hébergement", ["hotel", "camping", "airbnb"])

# 4. Sauvegarde de la configuration
categorizer.save_config("ma_config.json")
"""