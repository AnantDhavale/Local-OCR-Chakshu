"""
Text correction engines for OCR output
All run locally - no external API calls

Improvements:
- Dictionary-based spell checking
- Character confusion patterns
- Confidence-based filtering
- Word-level LLM processing
"""

from typing import List, Dict, Set, Tuple, Optional
import re
from enum import Enum
from pathlib import Path
from difflib import get_close_matches


class CorrectionStrategy(Enum):
    """Available correction strategies"""
    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"


class BaseCorrector:
    """Base class for all correctors"""
    
    def correct(self, text: str, confidences: List[float] = None) -> str:
        raise NotImplementedError


class RuleBasedCorrector(BaseCorrector):
    """
    Enhanced rule-based correction with:
    - Dictionary-based spell checking
    - Character confusion patterns
    - Confidence-based filtering
    - Context-aware corrections
    """
    
    def __init__(self, language: str = "en", confidence_threshold: float = 70.0):
        # Normalize language code (eng -> en, spa -> es, etc.)
        lang_map = {
            'eng': 'en',
            'spa': 'es',
            'fra': 'fr',
            'deu': 'de',
        }
        self.language = lang_map.get(language, language[:2] if len(language) > 2 else language)
        self.confidence_threshold = confidence_threshold
        self.dictionary = self._load_dictionary()
        self.char_confusion = self._load_char_confusion()
        self.rules = self._load_rules()
    
    def _load_dictionary(self) -> Set[str]:
        """Load dictionary from file"""
        # Map Tesseract language codes to dictionary file names
        lang_map = {
            'eng': 'en',
            'spa': 'es',
            'fra': 'fr',
            'deu': 'de',
        }
        dict_lang = lang_map.get(self.language, self.language[:2])  # Default to first 2 chars
        dict_path = Path(__file__).parent.parent / "dictionaries" / f"{dict_lang}_common.txt"
        
        dictionary = set()
        if dict_path.exists():
            with open(dict_path, 'r', encoding='utf-8') as f:
                dictionary = {line.strip().lower() for line in f if line.strip()}
        
        # Add common words if dictionary is empty or small
        if len(dictionary) < 50:
            dictionary.update([
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
                'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
                'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
                'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some',
                'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
                'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
                'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
                'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was',
                'are', 'been', 'has', 'had', 'were', 'said', 'did', 'get', 'may',
                'down', 'side', 'been', 'now', 'find', 'long', 'down', 'day', 'get',
                'made', 'find', 'use', 'may', 'water', 'long', 'little', 'very', 'after'
            ])
        
        return dictionary
    
    def _load_char_confusion(self) -> Dict[str, List[str]]:
        """
        Load character confusion patterns common in OCR
        Maps commonly confused characters to their correct forms
        """
        return {
            # Single character confusions
            'rn': ['m'],  # rn looks like m
            'vv': ['w'],  # vv looks like w
            'cl': ['d'],  # cl looks like d
            'ii': ['u'],  # ii looks like u
            'iii': ['m'],  # iii looks like m
            'nn': ['m'],  # nn looks like m in some fonts
            
            # Number/letter confusions
            '0': ['O', 'o'],  # zero vs letter O
            'O': ['0'],  # letter O vs zero
            '1': ['l', 'I', 'i'],  # one vs l vs I
            'l': ['1', 'I'],  # lowercase L
            'I': ['1', 'l'],  # uppercase i
            '5': ['S', 's'],  # five vs S
            'S': ['5'],  # S vs 5
            '8': ['B'],  # eight vs B
            'B': ['8'],  # B vs 8
            '6': ['b', 'G'],  # six vs b/G
            '2': ['Z'],  # two vs Z
            
            # Special confusions
            '|': ['I', 'l', '1'],  # vertical bar
            'i': ['1', 'l'],  # lowercase i
        }
    
    def _load_rules(self) -> Dict[str, str]:
        """Load correction rules for language"""
        rules = {
            "en": {
                # Common OCR word errors
                r'\btbe\b': 'the',
                r'\baod\b': 'and',
                r'\bfrorn\b': 'from',
                r'\bwitb\b': 'with',
                r'\btbis\b': 'this',
                r'\btbat\b': 'that',
                r'\bwhicb\b': 'which',
                r'\bwbat\b': 'what',
                r'\bwben\b': 'when',
                r'\bwbere\b': 'where',
                r'\bhave\b': 'have',
                r'\bwbo\b': 'who',
                r'\bcau\b': 'can',
                r'\bwiil\b': 'will',
                r'\byoii\b': 'you',
                r'\brnore\b': 'more',
                r'\bsorne\b': 'some',
                r'\btirne\b': 'time',
                
                # Numbers with trailing letters
                r'(\d+)l\b': r'\1',  # 1l -> 1
                r'(\d+)O\b': r'\g<1>0',  # 10O -> 100 (use \g<1> to avoid \10 backreference error)
                r'\bO(\d+)': r'0\1',  # O5 -> 05
                
                # Multiple spaces
                r'\s+': ' ',
            }
        }
        return rules.get(self.language, {})
    
    def _is_likely_number(self, word: str) -> bool:
        """Check if word is likely a number despite OCR errors"""
        # Check if majority of characters are digits
        digit_count = sum(c.isdigit() for c in word)
        return digit_count > len(word) / 2
    
    def _fix_number_confusions(self, word: str) -> str:
        """Fix common number/letter confusions in numeric contexts"""
        if not self._is_likely_number(word):
            return word
        
        # Fix common confusions in numbers
        fixes = {
            'O': '0', 'o': '0',  # Letter O to zero
            'l': '1', 'I': '1',  # Letter l/I to one
            'S': '5', 's': '5',  # Letter S to five
            'B': '8',            # Letter B to eight
        }
        
        fixed = ''
        for char in word:
            fixed += fixes.get(char, char)
        
        return fixed
    
    def _spell_check_word(self, word: str, confidence: float = 100.0) -> str:
        """
        Spell check a single word using dictionary
        Only correct low-confidence words
        """
        # Skip high-confidence words
        if confidence >= self.confidence_threshold:
            return word
        
        # Skip numbers, punctuation, short words
        if len(word) < 3 or not word.isalpha():
            return word
        
        word_lower = word.lower()
        
        # Word is in dictionary - keep it
        if word_lower in self.dictionary:
            return word
        
        # Only try character confusion fixes for VERY low confidence (<50)
        if confidence < 50:
            fixed_word = self._try_char_confusion_fix(word_lower)
            if fixed_word and fixed_word in self.dictionary:
                # Preserve original case
                if word.isupper():
                    return fixed_word.upper()
                elif word[0].isupper():
                    return fixed_word.capitalize()
                return fixed_word
        
        # Find close matches in dictionary (only for low confidence)
        if confidence < 60:
            matches = get_close_matches(word_lower, self.dictionary, n=1, cutoff=0.85)
            if matches:
                match = matches[0]
                # Preserve original case
                if word.isupper():
                    return match.upper()
                elif word[0].isupper():
                    return match.capitalize()
                return match
        
        return word
    
    def _try_char_confusion_fix(self, word: str) -> Optional[str]:
        """Try to fix word using character confusion patterns"""
        # Try replacing confused characters
        for confused, corrections in self.char_confusion.items():
            if confused in word:
                for correction in corrections:
                    candidate = word.replace(confused, correction)
                    if candidate in self.dictionary:
                        return candidate
        return None
    
    def correct(self, text: str, confidences: List[float] = None) -> str:
        """
        Apply rule-based corrections with confidence filtering
        
        Args:
            text: Text to correct
            confidences: Word-level confidence scores (0-100)
            
        Returns:
            Corrected text
        """
        # Apply pattern-based rules first
        corrected = text
        for pattern, replacement in self.rules.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Word-level spell checking with confidence filtering
        words = corrected.split()
        
        # If no confidences provided, assume high confidence (don't over-correct)
        if not confidences:
            confidences = [100.0] * len(words)  # Assume high confidence - only correct obvious errors
        
        # Ensure confidences list matches words
        if len(confidences) < len(words):
            confidences.extend([100.0] * (len(words) - len(confidences)))
        
        corrected_words = []
        for word, conf in zip(words, confidences):
            # Check if it's a number and fix number confusions
            if self._is_likely_number(word):
                corrected_words.append(self._fix_number_confusions(word))
            else:
                # Spell check with confidence filtering
                corrected_words.append(self._spell_check_word(word, conf))
        
        return ' '.join(corrected_words)


class LLMCorrector(BaseCorrector):
    """
    LLM-based correction using small local models
    - Runs on CPU (no GPU needed)
    - Uses ~80MB-500MB RAM
    - Better quality than rules
    - Now processes word-by-word for better context
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Args:
            model_name: HuggingFace model to use
                       Default is tiny (80MB) and CPU-friendly
        """
        try:
            from transformers import pipeline
            
            self.model = pipeline(
                "text2text-generation",
                model=model_name,
                device=-1,  # CPU only
                max_length=512
            )
            print(f"✓ Loaded {model_name} on CPU")
            
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
    
    def correct(self, text: str, confidences: List[float] = None) -> str:
        """
        Use LLM to correct OCR errors with word-level processing
        
        Args:
            text: Text to correct
            confidences: Word-level confidence scores
            
        Returns:
            Corrected text
        """
        # Split into sentences for better context
        sentences = re.split(r'([.!?]+)', text)
        corrected_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                corrected_sentences.append(sentence)
                continue
            
            # Skip punctuation-only sentences
            if not any(c.isalnum() for c in sentence):
                corrected_sentences.append(sentence)
                continue
            
            # Create a focused prompt for better results
            prompt = f"Correct OCR errors: {sentence}"
            
            try:
                result = self.model(
                    prompt,
                    max_length=min(len(sentence) + 50, 512),
                    num_return_sequences=1,
                    do_sample=False  # Deterministic output
                )
                
                corrected = result[0]['generated_text']
                corrected_sentences.append(corrected)
            except Exception:
                # Fall back to original if LLM fails
                corrected_sentences.append(sentence)
        
        return ''.join(corrected_sentences)


class HybridCorrector(BaseCorrector):
    """
    Combines rule-based + LLM correction
    - Fast rule-based for high-confidence words
    - LLM for low-confidence sections only
    """
    
    def __init__(self, confidence_threshold: float = 70.0, language: str = "en"):
        self.confidence_threshold = confidence_threshold
        self.language = language
        self.rules = RuleBasedCorrector(language=language, confidence_threshold=confidence_threshold)
        self.llm = None  # Lazy load
    
    def _ensure_llm(self):
        """Lazy load LLM only when needed"""
        if self.llm is None:
            try:
                self.llm = LLMCorrector()
            except ImportError:
                print("⚠ LLM not available, using rules only")
                self.llm = self.rules
    
    def correct(self, text: str, confidences: List[float] = None) -> str:
        """
        Apply hybrid correction strategy
        
        - Always apply rule-based corrections
        - Use LLM only for low-confidence sections
        
        Args:
            text: Text to correct
            confidences: Word-level confidence scores
            
        Returns:
            Corrected text
        """
        # Always apply rule-based corrections first
        text = self.rules.correct(text, confidences)
        
        # If we have confidence data, identify low-confidence sections
        if confidences and len(confidences) > 0:
            # Count low-confidence words
            low_conf_count = sum(1 for c in confidences if c < self.confidence_threshold)
            total_words = len(confidences)
            
            # If more than 20% of words have low confidence, use LLM
            if total_words > 0 and (low_conf_count / total_words) > 0.2:
                self._ensure_llm()
                if self.llm != self.rules:  # Only if LLM is available
                    text = self.llm.correct(text, confidences)
        
        return text


def get_corrector(strategy: CorrectionStrategy, language: str = "en") -> BaseCorrector:
    """
    Factory function to get the appropriate corrector
    
    Args:
        strategy: Correction strategy to use
        language: Language code for correction
        
    Returns:
        Corrector instance
    """
    if strategy == CorrectionStrategy.RULE_BASED:
        return RuleBasedCorrector(language=language)
    elif strategy == CorrectionStrategy.LLM:
        return LLMCorrector()
    elif strategy == CorrectionStrategy.HYBRID:
        return HybridCorrector(language=language)
    else:
        raise ValueError(f"Unknown correction strategy: {strategy}")
