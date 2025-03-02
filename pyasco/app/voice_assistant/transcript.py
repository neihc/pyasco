# transcript.py
import logging

logger = logging.getLogger(__name__)

class TranscriptCollector:
    """Collect and manage transcript parts from Deepgram"""
    def __init__(self, max_sentences=10):
        self.reset()
        self.full_sentences = []
        self.current_partial = ""
        self.max_sentences = max_sentences
        
    def reset(self):
        """Reset the current transcript collection"""
        self.transcript_parts = []
        
    def add_part(self, part):
        """Add a transcript part"""
        self.transcript_parts.append(part)
        
    def get_full_transcript(self):
        """Get the full transcript from collected parts"""
        return ' '.join(self.transcript_parts)
    
    def update_partial(self, text):
        """Update the current partial transcript"""
        self.current_partial = text
        
    def add_sentence(self, sentence):
        """Add a completed sentence to the history"""
        if sentence.strip():
            self.full_sentences.append(sentence.strip())
            if len(self.full_sentences) > self.max_sentences:
                self.full_sentences = self.full_sentences[-self.max_sentences:]
    
    def get_combined_input(self):
        """Get combined input from the sliding window of sentences"""
        return ' '.join(self.full_sentences)
            
    def get_display_text(self):
        """Get text for display purposes"""
        history = self.full_sentences[-3:] if self.full_sentences else []
        if self.current_partial:
            return "\n".join(history + [f"🎤 {self.current_partial}"])
        else:
            return "\n".join(history + ["🎤 Listening..."])
