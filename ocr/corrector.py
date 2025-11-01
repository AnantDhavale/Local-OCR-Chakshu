"""
Text correction engines for OCR output
All run locally - no external API calls
"""

from typing import List, Dict
import re


class BaseCorrector:
    """Base class for all correctors"""
    
    def correct(self, text: str, confidences: List[float] = None) -> str:
        raise NotImplementedError


class RuleBasedCorrector(BaseCorrector):
    """
    Simple rule-based correction
    - No external dependencies
    - Instant processing
    - 100% deterministic
    """
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict:
        """Load correction rules for language"""
        rules = {
            "en": {
                # Character-level fixes
                r'\brn\b': 'm',
                r'\bcl\b': 'd',
                r'(\d+)l\b': r'\1',  # Numbers ending in l
                
                # Common word fixes
                r'\btbe\b': 'the',
                r'\baod\b': 'and',
                r'\bfrorn\b': 'from',
            }
        }
        return rules.get(self.language, {})
    
    def correct(self, text: str, confidences: List[float] = None) -> str:
        """Apply rule-based corrections"""
        corrected = text
        
        for pattern, replacement in self.rules.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected


class LLMCorrector(BaseCorrector):
    """
    LLM-based correction using small local models
    - Runs on CPU (no GPU needed)
    - Uses ~80MB-500MB RAM
    - Better quality than rules
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
        """Use LLM to correct OCR errors"""
        prompt = f"Fix OCR errors in this text: {text}"
        
        result = self.model(
            prompt,
            max_length=len(text) + 100,
            num_return_sequences=1
        )
        
        return result[0]['generated_text']


class HybridCorrector(BaseCorrector):
    """
    Combines rule-based + LLM correction
    - Fast rule-based for high-confidence words
    - LLM for low-confidence words
    """
    
    def __init__(self):
        self.rules = RuleBasedCorrector()
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
        """Apply hybrid correction strategy"""
        # Quick rule-based pass first
        text = self.rules.correct(text, confidences)
        
        # If we have low confidence sections, use LLM
        if confidences and min(confidences) < 60:
            self._ensure_llm()
            text = self.llm.correct(text, confidences)
        
        return text
