import pysrt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import chardet
import re
import json
import os
import logging
import traceback
from typing import Dict, List, Optional, Tuple
from languages import LanguageManager
from langdetect import detect, LangDetectException
from langdetect.lang_detect_exception import ErrorCode

class TranslationContext:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
    
    def get_context(self, subs: List[pysrt.SubRipItem], current_index: int) -> Dict:
        """Get context for translation."""
        start = max(0, current_index - self.window_size)
        end = min(len(subs), current_index + self.window_size + 1)
        
        context = {
            'previous': [subs[i].text for i in range(start, current_index)],
            'current': subs[current_index].text,
            'next': [subs[i].text for i in range(current_index + 1, end)],
            'has_continuation': '...' in subs[current_index].text,
            'is_continuation': current_index > 0 and '...' in subs[current_index - 1].text
        }
        
        return context

    def prepare_text_with_context(self, context: Dict) -> str:
        """Prepare text with context markers."""
        parts = []
        if context['is_continuation']:
            parts.append("[CONT]")
        if context['previous']:
            parts.append(f"[PREV]{context['previous'][-1]}")
        parts.append(context['current'])
        if context['has_continuation']:
            parts.append("[NEXT]")
        return " ".join(parts)

class QualityAnalyzer:
    def __init__(self):
        self.quality_checks = {
            'length_ratio': self.check_length_ratio,
            'punctuation': self.check_punctuation,
            'numbers': self.check_numbers,
            'names': self.check_proper_names,
            'context': self.check_context_coherence
        }
        
        self.weights = {
            'length_ratio': 0.3,
            'punctuation': 0.2,
            'numbers': 0.2,
            'names': 0.2,
            'context': 0.1
        }
    
    def analyze(self, original: str, translated: str, context: Dict) -> Dict:
        """Analyze translation quality."""
        score = 100
        issues = []
        
        for check_name, check_func in self.quality_checks.items():
            check_score, check_issues = check_func(original, translated, context)
            score -= (100 - check_score) * self.weights[check_name]
            issues.extend(check_issues)
        
        return {
            'score': max(0, min(100, score)),
            'issues': issues,
            'needs_review': score < 70
        }
    
    def check_length_ratio(self, original: str, translated: str, context: Dict) -> Tuple[float, List[str]]:
        """Check if translation length is reasonable."""
        orig_words = len(original.split())
        trans_words = len(translated.split())
        ratio = abs(orig_words - trans_words) / max(orig_words, trans_words)
        
        score = 100
        issues = []
        
        if ratio > 0.5:
            score -= min(50, ratio * 100)
            issues.append(f"Length ratio issue: {ratio:.2f}")
        
        return score, issues
    
    def check_punctuation(self, original: str, translated: str, context: Dict) -> Tuple[float, List[str]]:
        """Check punctuation consistency."""
        score = 100
        issues = []
        
        # Check ending punctuation
        if original[-1] in '.!?' and translated[-1] not in '.!?':
            score -= 20
            issues.append("Missing end punctuation")
        
        # Check question marks
        if original.count('?') != translated.count('?'):
            score -= 15
            issues.append("Question mark mismatch")
        
        # Check exclamation marks
        if original.count('!') != translated.count('!'):
            score -= 15
            issues.append("Exclamation mark mismatch")
        
        return score, issues
    
    def check_numbers(self, original: str, translated: str, context: Dict) -> Tuple[float, List[str]]:
        """Check if numbers are preserved."""
        orig_numbers = set(re.findall(r'\d+', original))
        trans_numbers = set(re.findall(r'\d+', translated))
        
        score = 100
        issues = []
        
        if orig_numbers != trans_numbers:
            score -= 25 * len(orig_numbers - trans_numbers)
            issues.append("Numbers not preserved")
        
        return score, issues
    
    def check_proper_names(self, original: str, translated: str, context: Dict) -> Tuple[float, List[str]]:
        """Check if proper names are preserved."""
        orig_names = set(re.findall(r'\b[A-Z][a-z]+\b', original))
        trans_names = set(re.findall(r'\b[A-Z][a-z]+\b', translated))
        
        score = 100
        issues = []
        
        missing_names = orig_names - trans_names
        if missing_names:
            score -= 20 * len(missing_names)
            issues.append(f"Missing names: {', '.join(missing_names)}")
        
        return score, issues
    
    def check_context_coherence(self, original: str, translated: str, context: Dict) -> Tuple[float, List[str]]:
        """Check coherence with context."""
        score = 100
        issues = []
        
        # Check continuation
        if context['is_continuation'] and not translated.startswith('...'):
            score -= 20
            issues.append("Missing continuation marks")
        
        # Check dialogue consistency
        if original.startswith('-') and not translated.startswith('-'):
            score -= 15
            issues.append("Missing dialogue marker")
        
        return score, issues

class TranslationCheckpoint:
    def __init__(self, save_interval: int = 50):
        self.save_interval = save_interval
        self.checkpoint_file = "translation_checkpoint.json"
    
    def save(self, state: Dict) -> None:
        """Save current translation state."""
        if state['current_index'] % self.save_interval == 0:
            checkpoint = {
                'timestamp': time.time(),
                'input_file': state['input_file'],
                'current_index': state['current_index'],
                'translations': state['translations'],
                'metrics': state['metrics']
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def load(self) -> Optional[Dict]:
        """Load last checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, encoding='utf-8') as f:
                return json.load(f)
        return None

class TranslatorConfig:
    def __init__(self, config_file: Optional[str] = None):
        # Current default values
        self.defaults = {
            'max_length': 512,
            'context_window': 3,
            'quality_threshold': 70,
            'save_interval': 50,
            'retry_attempts': 3,
            'batch_size': None  # Will be calculated automatically
        }
        
        self.config = self.load_config(config_file)
    
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration maintaining defaults."""
        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                user_config = json.load(f)
                return {**self.defaults, **user_config}
        return self.defaults
    
    def get(self, key: str) -> any:
        """Get configuration value."""
        return self.config.get(key, self.defaults.get(key))

class TranslationLogger:
    def __init__(self, log_file: str = "translation.log"):
        self.log_file = log_file
        self.start_time = time.time()
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Detailed error logging."""
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        logging.error(json.dumps(error_info, indent=2))
    
    def log_translation(self, index: int, original: str, translated: str, metrics: Dict) -> None:
        """Log translations and metrics."""
        logging.info(json.dumps({
            'index': index,
            'original': original,
            'translated': translated,
            'quality_score': metrics['score'],
            'issues': metrics['issues']
        }, indent=2))

class SubtitleTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", 
                 config_file: Optional[str] = None,
                 use_extended_languages: bool = False):
        # Mapeo de códigos de langdetect a códigos NLLB
        self.langdetect_to_nllb = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ja': 'japanese',
            'ko': 'korean',
            'zh': 'chinese (simplified)',
            'ru': 'russian',
            'ar': 'arabic',
            'hi': 'hindi',
            'bn': 'bengali',
            'nl': 'dutch',
            'pl': 'polish',
            'tr': 'turkish',
            'vi': 'vietnamese',
            'th': 'thai',
            'el': 'greek',
            'he': 'hebrew'
        }
        # Load configuration
        self.config = TranslatorConfig(config_file)
        
        # Initialize components
        self.context = TranslationContext(self.config.get('context_window'))
        self.quality_analyzer = QualityAnalyzer()
        self.checkpoint = TranslationCheckpoint(self.config.get('save_interval'))
        self.logger = TranslationLogger()
        
        # GPU/CUDA setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = self.config.get('max_length')
        
        print(f"Using device: {self.device}")
        
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Calculate optimal batch size
        self.batch_size = self.calculate_optimal_batch_size()
        print(f"Batch size: {self.batch_size}")
        
        # Initialize language manager
        self.language_manager = LanguageManager(use_extended_languages)

    def calculate_optimal_batch_size(self) -> int:
        """Calculate conservative batch size based on available memory."""
        if not torch.cuda.is_available():
            return 4  # Default CPU batch size
            
        # Get total and allocated memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory
        
        # Estimate memory per sample (based on max_length)
        estimated_sample_memory = 2 * self.max_length * 4  # tokens * 4 bytes
        
        # Use 70% of free memory
        safe_memory = free_memory * 0.7
        
        # Calculate optimal batch size
        optimal_batch = max(1, min(32, int(safe_memory / estimated_sample_memory)))
        
        # Reduce by 20% for more conservative usage
        conservative_batch = max(1, int(optimal_batch * 0.8))
        
        print(f"GPU Memory - Total: {total_memory/1024**2:.1f}MB, Free: {free_memory/1024**2:.1f}MB")
        print(f"Optimal batch size: {optimal_batch}")
        print(f"Conservative batch size: {conservative_batch}")
        return conservative_batch

    def estimate_translation_time(self, subs: List[pysrt.SubRipItem], total_subs: int) -> float:
        """Estimate translation time using actual batch processing."""
        if total_subs < 1:
            return 0
        
        # Use a full batch for more accurate estimation
        sample_size = min(self.batch_size, total_subs)
        sample_texts = [subs[i].text for i in range(sample_size)]
        
        start_time = time.time()
        self.translate_batch(sample_texts, self.src_lang, self.tgt_lang)
        sample_time = time.time() - start_time
        
        # Estimate based on number of full batches
        total_batches = (total_subs + self.batch_size - 1) // self.batch_size
        estimated_time = (sample_time * total_batches)
        
        print(f"\nTime Estimation:")
        print(f"- Total subtitles: {total_subs}")
        print(f"- Total batches: {total_batches}")
        print(f"- Estimated time: {estimated_time:.1f} seconds")
        print(f"- Time per batch: {sample_time:.2f} seconds")
        
        return estimated_time

    def process_subtitle_text(self, text: str) -> Tuple[str, List[str]]:
        """Preserve formatting while translating text."""
        # Extract and preserve format tags
        format_tags = re.findall(r'<[^>]+>', text)
        # Extract clean text
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        return clean_text, format_tags

    def restore_format_tags(self, translated_text: str, format_tags: List[str], original_text: str) -> str:
        """Restore format tags to translated text."""
        if not format_tags:
            return translated_text
            
        # Maintain relative positions of tags
        result = translated_text
        for tag in format_tags:
            # Calculate relative position from original text
            relative_pos = original_text.index(tag) / len(original_text)
            # Apply to translated text
            insert_pos = int(len(result) * relative_pos)
            result = result[:insert_pos] + tag + result[insert_pos:]
        
        return result

    def translate_batch_with_retry(self, texts: List[str], max_retries: int = 3) -> List[str]:
        """Translate a batch of texts with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.translate_batch(texts, self.src_lang, self.tgt_lang)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 2  # Exponential backoff
                    print(f"\nBatch translation error (attempt {attempt + 1}/{max_retries}):")
                    print(f"Error: {str(e)}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"\n❌ Final error after {max_retries} attempts:")
                    print(f"Error: {str(e)}")
                    self.logger.log_error(e, {'texts': texts, 'attempt': attempt})
                    raise

    def validate_languages(self, source_lang: str, target_lang: str) -> Tuple[str, str]:
        """Validate languages using language manager."""
        try:
            src_code = self.language_manager.get_language_code(source_lang)
            tgt_code = self.language_manager.get_language_code(target_lang)
            return src_code, tgt_code
        except ValueError as e:
            # Add suggestions if in extended mode
            if self.language_manager.use_extended_languages:
                suggestions = self.language_manager.get_suggestions(source_lang)
                if suggestions:
                    raise ValueError(f"{str(e)}\nDid you mean: {', '.join(suggestions)}?")
            raise

    def validate_srt(self, file_path: str) -> pysrt.SubRipFile:
        """Validate SRT file and ensure UTF-8 encoding."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # Detect encoding
                encoding = chardet.detect(content)['encoding']
                
                # Convert to UTF-8 if necessary
                if encoding and encoding.lower() != 'utf-8':
                    content = content.decode(encoding).encode('utf-8')
                    with open(file_path, 'wb') as f:
                        f.write(content)
                
                # Validate SRT format
                subs = pysrt.from_string(content.decode('utf-8'))
                return subs
        except Exception as e:
            self.logger.log_error(e, {'file_path': file_path})
            raise ValueError(f"Invalid SRT file: {str(e)}")

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Translate a batch of texts."""
        if not texts:
            return []
            
        # Prepare input
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate translations
        with torch.no_grad():  # Optimize memory usage during inference
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer._convert_token_to_id_with_added_voc(tgt_lang),
                max_length=self.max_length
            )
        
        # Decode translations
        results = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return results

    def detect_language(self, subs: pysrt.SubRipFile) -> str:
        """
        Detect the language of the subtitle file.
        Uses a representative sample of subtitles for better accuracy.
        """
        try:
            # Take a representative sample (maximum 100 subtitles)
            sample_size = min(100, len(subs))
            step = max(1, len(subs) // sample_size)
            sample_texts = [subs[i].text for i in range(0, len(subs), step)][:sample_size]
            
            # Concatenate texts for better detection
            full_text = ' '.join(sample_texts)
            
            # Detect language
            detected_code = detect(full_text)
            
            # Convert langdetect code to NLLB language name
            if detected_code in self.langdetect_to_nllb:
                return self.langdetect_to_nllb[detected_code]
            
            raise ValueError(f"Detected language '{detected_code}' not supported")
            
        except LangDetectException as e:
            if e.code == ErrorCode.CantDetectError:
                raise ValueError("Could not automatically detect language")
            raise

    def print_quality_summary(self, translated_texts: List[str], metrics: List[Dict], subs: pysrt.SubRipFile) -> None:
        """Print a summary of translation quality."""
        total_subs = len(translated_texts)
        avg_score = sum(m['score'] for m in metrics) / total_subs
        
        print("\nQuality Summary:")
        print("=" * 50)
        print(f"Average score: {avg_score:.1f}/100")
        
        # Quality distribution
        excellent = sum(1 for m in metrics if m['score'] >= 90)
        good = sum(1 for m in metrics if 80 <= m['score'] < 90)
        acceptable = sum(1 for m in metrics if 70 <= m['score'] < 80)
        needs_review = sum(1 for m in metrics if m['score'] < 70)
        
        print("\nQuality Distribution:")
        print(f"- Excellent (90+): {excellent} ({excellent/total_subs*100:.1f}%)")
        print(f"- Good (80-89): {good} ({good/total_subs*100:.1f}%)")
        print(f"- Acceptable (70-79): {acceptable} ({acceptable/total_subs*100:.1f}%)")
        print(f"- Needs review (<70): {needs_review} ({needs_review/total_subs*100:.1f}%)")
        
        # Recommendations based on overall quality
        if needs_review > 0:
            print("\nRecommendations:")
            if needs_review > total_subs * 0.1:  # More than 10% needs review
                print("- Manual review recommended")
            if any('Missing names' in ' '.join(m['issues']) for m in metrics):
                print("- Verify proper name preservation")
            if any('Question mark mismatch' in ' '.join(m['issues']) for m in metrics):
                print("- Review question mark punctuation")
            if any('Missing dialogue marker' in ' '.join(m['issues']) for m in metrics):
                print("- Check dialogue markers consistency")
            if any('Length ratio issue' in ' '.join(m['issues']) for m in metrics):
                print("- Review translations with significant length differences")

    def translate_srt(self, input_path: str, output_path: str, source_lang: Optional[str] = None, target_lang: str = "spanish") -> None:
        try:
            # Validate and load SRT file
            print(f"Opening and validating subtitle file: {input_path}")
            subs = self.validate_srt(input_path)
            
            # Detect language if not specified
            if source_lang is None:
                print("Detecting source language...")
                source_lang = self.detect_language(subs)
                print(f"Detected language: {source_lang}")
            
            # Validate languages
            self.src_lang, self.tgt_lang = self.validate_languages(source_lang, target_lang)
            
            # Check for existing checkpoint
            checkpoint = self.checkpoint.load()
            start_index = 0
            translated_texts = []
            
            if checkpoint and checkpoint['input_file'] == input_path:
                start_index = checkpoint['current_index']
                translated_texts = checkpoint['translations']
                print(f"Resuming from checkpoint at subtitle {start_index}")
            
            # Prepare for translation
            total_subs = len(subs)
            print(f"\nTranslating subtitles from {source_lang} to {target_lang}")
            print(f"Total lines to translate: {total_subs}")
            
            # Estimate total time
            if start_index == 0:
                estimated_time = self.estimate_translation_time(subs, total_subs)
            
            start_time = time.time()
            
            # Process subtitles in batches
            format_info = []
            texts_to_translate = []
            
            # First, process all texts and extract formatting
            for i in range(start_index, total_subs):
                clean_text, tags = self.process_subtitle_text(subs[i].text)
                context = self.context.get_context(subs, i)
                prepared_text = self.context.prepare_text_with_context(context)
                texts_to_translate.append(prepared_text)
                format_info.append((tags, subs[i].text, context))
            
            # Translate in batches with more granular progress updates
            with tqdm(total=len(texts_to_translate), desc="Translating", initial=start_index) as pbar:
                for i in range(0, len(texts_to_translate), self.batch_size):
                    try:
                        batch = texts_to_translate[i:i + self.batch_size]
                        
                        # Process in smaller sub-batches for better progress feedback
                        sub_batch_size = max(1, self.batch_size // 4)
                        for j in range(0, len(batch), sub_batch_size):
                            sub_batch = batch[j:j + sub_batch_size]
                            translations = self.translate_batch_with_retry(sub_batch)
                            
                            # Process each translation
                            for k, translation in enumerate(translations):
                                idx = i + j + k
                                tags, original, context = format_info[idx]
                                
                                # Analyze quality
                                metrics = self.quality_analyzer.analyze(
                                    original, translation, context
                                )
                                
                                # Log translation and metrics
                                self.logger.log_translation(
                                    start_index + idx,
                                    original,
                                    translation,
                                    metrics
                                )
                                
                                # Format and store translation
                                formatted_translation = self.restore_format_tags(
                                    translation, tags, original
                                )
                                translated_texts.append(formatted_translation)
                            
                            # Update progress
                            pbar.update(len(sub_batch))
                            
                            # Save checkpoint
                            self.checkpoint.save({
                                'input_file': input_path,
                                'current_index': start_index + i + j + len(sub_batch),
                                'translations': translated_texts,
                                'metrics': {'time': time.time() - start_time}
                            })
                            
                            # Calculate and display ETA
                            elapsed = time.time() - start_time
                            progress = len(translated_texts) / total_subs
                            if progress > 0:
                                eta = (elapsed / progress) * (1 - progress)
                                pbar.set_postfix({
                                    'ETA': f'{eta:.1f}s',
                                    'Processed': f'{len(translated_texts)}/{total_subs}'
                                })
                            
                            # Free up memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                    except Exception as e:
                        current_position = start_index + i + (j if 'j' in locals() else 0)
                        self.logger.log_error(e, {
                            'position': current_position,
                            'batch': sub_batch if 'sub_batch' in locals() else batch
                        })
                        raise
            
            # Update subtitles with translations
            for i, translation in enumerate(translated_texts):
                subs[i].text = translation
            
            # Save translated subtitles
            print(f"\nSaving translated subtitles to: {output_path}")
            subs.save(output_path, encoding='utf-8')
            
            # Clean up checkpoint if translation completed
            if os.path.exists(self.checkpoint.checkpoint_file):
                os.remove(self.checkpoint.checkpoint_file)
            
            # Print completion statistics
            total_time = time.time() - start_time
            print(f"\nTranslation completed!")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average time per subtitle: {total_time/total_subs:.2f} seconds")
            if start_index == 0:
                print(f"Initial estimate was: {estimated_time:.1f} seconds")
            
            # Print quality summary
            metrics = [
                self.quality_analyzer.analyze(info[1], trans, info[2])
                for info, trans in zip(format_info, translated_texts)
            ]
            self.print_quality_summary(translated_texts, metrics, subs)
            
        except Exception as e:
            self.logger.log_error(e)
            print(f"Error during translation: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def generate_output_path(input_path: str, target_lang: str) -> str:
    """Generate default output path based on input path and target language."""
    path = Path(input_path)
    # Get language code (first two letters)
    lang_code = target_lang.lower()[:2]
    # Create new filename: original_name_langcode.srt
    new_name = f"{path.stem}_{lang_code}{path.suffix}"
    return str(path.parent / new_name)

def list_supported_languages(extended: bool = False) -> None:
    """List all supported languages."""
    manager = LanguageManager(extended)
    languages = manager.list_languages()
    
    print("\nSupported Languages:")
    print("=" * 50)
    for lang in sorted(languages):
        print(f"- {lang}")
    if not extended:
        print("\nNote: Use --extended-languages to see all available languages")

def main():
    try:
        parser = argparse.ArgumentParser(description="Translate subtitles using NLLB")
        
        # Create mutually exclusive group for modes
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument("--list-languages", action="store_true",
                              help="List supported languages")
        mode_group.add_argument("--translate", action="store_true",
                              help="Explicitly use translate mode")
        
        # Global arguments
        parser.add_argument("--extended-languages", action="store_true",
                          help="Enable support for all NLLB languages")
        
        # Translation arguments
        parser.add_argument("input", nargs='?', help="Input SRT file path")
        parser.add_argument("output", nargs='?', help="Output SRT file path (optional)")
        parser.add_argument("--source", help="Source language (autodetect if not specified)")
        parser.add_argument("--target", default="spanish", help="Target language (default: spanish)")
        parser.add_argument("--config", help="Configuration file path (optional)")
        
        args = parser.parse_args()
        
        # Handle language listing
        if args.list_languages:
            list_supported_languages(args.extended_languages)
            return
        
        # If no input file, show help
        if not args.input:
            parser.print_help()
            return
        
        # Process translation
        output_path = Path(args.output if args.output else generate_output_path(args.input, args.target))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize translator and process file
        translator = SubtitleTranslator(
            config_file=args.config,
            use_extended_languages=args.extended_languages
        )
        translator.translate_srt(args.input, str(output_path), args.source, args.target)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
