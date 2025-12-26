import whisper
import json
import os
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PodcastTranscriber:
    def __init__(self, model_size="base"):
        """
        Inisialisasi transcriber dengan model Whisper.
        
        Args:
            model_size (str): Ukuran model whisper 
                'tiny', 'base', 'small', 'medium', 'large'
        """
        self.model_size = model_size
        self.model = None
        logger.info(f"Menggunakan model Whisper: {model_size}")
        
    def load_model(self):
        """Load model Whisper (akan download otomatis jika belum ada)"""
        if self.model is None:
            logger.info(f"Loading model {self.model_size}...")
            start_time = time.time()
            
            try:
                self.model = whisper.load_model(self.model_size)
                load_time = time.time() - start_time
                logger.info(f"Model loaded dalam {load_time:.1f} detik")
            except Exception as e:
                logger.error(f"Gagal load model: {e}")
                raise
        
        return self.model
    
    def transcribe(self, audio_path, language=None):
        """
        Transkripsi audio ke text dengan timestamp.
        
        Args:
            audio_path (str): Path ke file audio
            language (str): Bahasa audio (None untuk auto-detect)
            
        Returns:
            dict: Hasil transkripsi dengan segments
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file tidak ditemukan: {audio_path}")
        
        logger.info(f"Memulai transkripsi: {audio_path}")
        start_time = time.time()
        
        # Load model jika belum
        model = self.load_model()
        
        # Options untuk transkripsi
        transcribe_options = {
            "task": "transcribe",
            "verbose": True,  # Whisper akan print progress
            "word_timestamps": True  # Penting untuk timestamp per kata
        }
        
        if language:
            transcribe_options["language"] = language
        
        # Transkripsi
        result = model.transcribe(audio_path, **transcribe_options)
        
        # Hitung performance
        transcribe_time = time.time() - start_time
        audio_duration = self._get_audio_duration(audio_path)
        
        logger.info(f"Transkripsi selesai dalam {transcribe_time:.1f} detik")
        logger.info(f"Durasi audio: {audio_duration:.1f} detik")
        logger.info(f"Real-time factor: {transcribe_time/audio_duration:.2f}x")
        
        return result
    
    def save_transcript(self, result, output_path):
        """
        Simpan hasil transkripsi ke JSON dan SRT.
        
        Args:
            result (dict): Hasil dari Whisper
            output_path (str): Path dasar untuk output (tanpa ekstensi)
        """
        # 1. Simpan sebagai JSON lengkap
        json_path = f"{output_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcript JSON disimpan: {json_path}")
        
        # 2. Simpan sebagai SRT (subtitle)
        srt_path = f"{output_path}.srt"
        self._save_srt(result['segments'], srt_path)
        
        # 3. Simpan sebagai text sederhana
        txt_path = f"{output_path}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        logger.info(f"Transcript disimpan dalam 3 format:")
        logger.info(f"  - JSON: {json_path}")
        logger.info(f"  - SRT: {srt_path}")
        logger.info(f"  - TXT: {txt_path}")
        
        return json_path, srt_path
    
    def _save_srt(self, segments, srt_path):
        """Convert Whisper segments ke format SRT"""
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            # Format timestamp: 00:00:00,000 --> 00:00:05,000
            start = self._format_timestamp(segment['start'])
            end = self._format_timestamp(segment['end'])
            
            srt_content += f"{i}\n"
            srt_content += f"{start} --> {end}\n"
            srt_content += f"{segment['text'].strip()}\n\n"
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
    
    def _format_timestamp(self, seconds):
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def _get_audio_duration(self, audio_path):
        """Get duration of audio file in seconds"""
        import subprocess
        try:
            cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{audio_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0

def test_transcription():
    """Test fungsi transkripsi"""
    print("\n=== TESTING TRANSCRIPTION ===")
    
    # 1. Cek file audio hasil ekstraksi
    audio_file = "data/samples/sample_podcast_audio.mp3"
    
    if not os.path.exists(audio_file):
        print("✗ File audio tidak ditemukan. Jalankan audio_extractor.py dulu!")
        return None
    
    print(f"✓ File audio ditemukan: {audio_file}")
    
    # 2. Inisialisasi transcriber dengan model kecil (cepat)
    print("Menggunakan model Whisper 'base' untuk testing cepat...")
    transcriber = PodcastTranscriber(model_size="base")
    
    # 3. Transkripsi
    try:
        print("Memulai transkripsi (mungkin perlu beberapa menit)...")
        result = transcriber.transcribe(audio_file)
        
        # 4. Simpan hasil
        output_base = "data/samples/sample_podcast_transcript"
        json_path, srt_path = transcriber.save_transcript(result, output_base)
        
        # 5. Print sample hasil
        print("\n=== SAMPLE TRANSCRIPT ===")
        print(f"Detected language: {result.get('language', 'unknown')}")
        print(f"Total segments: {len(result['segments'])}")
        
        # Tampilkan 3 segment pertama
        for i, seg in enumerate(result['segments'][:3], 1):
            print(f"\nSegment {i}:")
            print(f"  Time: {seg['start']:.1f}s - {seg['end']:.1f}s")
            print(f"  Text: {seg['text'][:100]}...")
        
        return json_path
        
    except Exception as e:
        print(f"✗ Transkripsi gagal: {e}")
        return None

if __name__ == "__main__":
    test_transcription()