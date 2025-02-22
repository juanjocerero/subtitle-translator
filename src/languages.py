"""Language management for NLLB translation."""
from typing import Dict, List, Optional

class LanguageManager:
    def __init__(self, use_extended_languages: bool = False):
        self.use_extended_languages = use_extended_languages
        
        # Basic codes (default behavior)
        self.basic_codes = {
            "english": "eng_Latn",
            "spanish": "spa_Latn",
            "french": "fra_Latn",
            "german": "deu_Latn",
            "italian": "ita_Latn",
            "portuguese": "por_Latn"
        }
        
        # Load extended codes only if requested
        self.extended_codes = self._load_extended_codes() if use_extended_languages else {}
    
    def _load_extended_codes(self) -> Dict[str, str]:
        """Load all NLLB supported language codes."""
        return {
            # African Languages
            'afrikaans': 'afr_Latn',
            'amharic': 'amh_Ethi',
            'bambara': 'bam_Latn',
            'hausa': 'hau_Latn',
            'igbo': 'ibo_Latn',
            'luganda': 'lug_Latn',
            'yoruba': 'yor_Latn',
            'zulu': 'zul_Latn',
            
            # Asian Languages
            'bengali': 'ben_Beng',
            'chinese (simplified)': 'zho_Hans',
            'chinese (traditional)': 'zho_Hant',
            'gujarati': 'guj_Gujr',
            'hindi': 'hin_Deva',
            'japanese': 'jpn_Jpan',
            'kannada': 'kan_Knda',
            'korean': 'kor_Hang',
            'malayalam': 'mal_Mlym',
            'marathi': 'mar_Deva',
            'tamil': 'tam_Taml',
            'telugu': 'tel_Telu',
            'thai': 'tha_Thai',
            'vietnamese': 'vie_Latn',
            
            # European Languages
            'bulgarian': 'bul_Cyrl',
            'croatian': 'hrv_Latn',
            'czech': 'ces_Latn',
            'danish': 'dan_Latn',
            'dutch': 'nld_Latn',
            'english': 'eng_Latn',
            'estonian': 'est_Latn',
            'finnish': 'fin_Latn',
            'french': 'fra_Latn',
            'german': 'deu_Latn',
            'greek': 'ell_Grek',
            'hungarian': 'hun_Latn',
            'italian': 'ita_Latn',
            'latvian': 'lav_Latn',
            'lithuanian': 'lit_Latn',
            'norwegian': 'nob_Latn',
            'polish': 'pol_Latn',
            'portuguese': 'por_Latn',
            'romanian': 'ron_Latn',
            'russian': 'rus_Cyrl',
            'serbian': 'srp_Cyrl',
            'slovak': 'slk_Latn',
            'slovenian': 'slv_Latn',
            'spanish': 'spa_Latn',
            'swedish': 'swe_Latn',
            'ukrainian': 'ukr_Cyrl',
            
            # Middle Eastern Languages
            'arabic': 'ara_Arab',
            'hebrew': 'heb_Hebr',
            'persian': 'fas_Arab',
            'turkish': 'tur_Latn',
            'urdu': 'urd_Arab',
            
            # Common Aliases
            'español': 'spa_Latn',
            'inglés': 'eng_Latn',
            'français': 'fra_Latn',
            'deutsch': 'deu_Latn',
            'italiano': 'ita_Latn',
            'português': 'por_Latn',
            '中文': 'zho_Hans',
            '日本語': 'jpn_Jpan',
            '한국어': 'kor_Hang',
            'русский': 'rus_Cyrl',
            'العربية': 'ara_Arab',
            'עברית': 'heb_Hebr',
            'हिन्दी': 'hin_Deva',
            
            # Regional Variants
            'portuguese (brazil)': 'por_Latn',
            'portuguese (portugal)': 'por_Latn',
            'spanish (latin america)': 'spa_Latn',
            'spanish (spain)': 'spa_Latn'
        }
    
    def get_language_code(self, language: str) -> str:
        """Get language code based on current mode."""
        language = language.lower()
        
        # Try basic codes first
        if language in self.basic_codes:
            return self.basic_codes[language]
            
        # If extended mode is enabled, try extended codes
        if self.use_extended_languages:
            if language in self.extended_codes:
                return self.extended_codes[language]
            # Try partial matches
            matches = [code for name, code in self.extended_codes.items() 
                      if language in name.lower()]
            if matches:
                return matches[0]
            
            raise ValueError(
                f"Unsupported language: {language}\n" +
                "Use --list-languages to see all available languages"
            )
        else:
            # In basic mode, show only basic languages
            raise ValueError(
                f"Unsupported language: {language}\n" +
                f"Supported languages: {', '.join(sorted(self.basic_codes.keys()))}\n" +
                "Use --extended-languages to access more languages"
            )
    
    def get_suggestions(self, query: str, max_suggestions: int = 3) -> List[str]:
        """Get language suggestions based on partial matches."""
        query = query.lower()
        codes = self.extended_codes if self.use_extended_languages else self.basic_codes
        matches = [name for name in codes.keys() 
                  if query in name.lower() or name.lower() in query]
        return sorted(matches)[:max_suggestions]
    
    def list_languages(self) -> List[str]:
        """List all supported languages in current mode."""
        if self.use_extended_languages:
            return sorted(self.extended_codes.keys())
        return sorted(self.basic_codes.keys())
