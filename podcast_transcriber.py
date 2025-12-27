import json
import glob
import logging
import os
import time
import argparse
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Konfigurasi untuk proses transkripsi"""
    model_size: str = "base"
    language: Optional[str] = None
    task: str = "transcribe"
    word_timestamps: bool = True
    verbose: bool = True
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    patience: float = 1.0
    structured_output: bool = True
    output_dir: str = "transcripts"
    device: Optional[str] = None
    compute_type: str = "float32"
    threads: int = 0  # 0 untuk auto-detect
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TranscriptionConfig':
        """Buat config dari dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class PodcastTranscriber:
    """Handler untuk transkripsi podcast menggunakan Whisper AI"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        Inisialisasi transcriber dengan model Whisper.
        
        Args:
            config (TranscriptionConfig): Konfigurasi transkripsi
        """
        self.config = config or TranscriptionConfig()
        self.model = None
        logger.info(f"Menggunakan model Whisper: {self.config.model_size}")
        
        # Set device jika ditentukan
        if self.config.device:
            import torch
            if self.config.device == "cuda" and torch.cuda.is_available():
                logger.info("Menggunakan GPU (CUDA)")
            elif self.config.device == "cpu":
                logger.info("Menggunakan CPU")
    
    def load_model(self) -> whisper.Whisper:
        """Load model Whisper (akan download otomatis jika belum ada)"""
        if self.model is None:
            logger.info(f"Loading model {self.config.model_size}...")
            start_time = time.time()
            
            try:
                # Set device jika ditentukan
                kwargs = {}
                if self.config.device:
                    kwargs["device"] = self.config.device
                
                self.model = whisper.load_model(
                    self.config.model_size,
                    **kwargs
                )
                load_time = time.time() - start_time
                logger.info(f"Model loaded dalam {load_time:.1f} detik")
                
                # Set threads untuk CPU
                if self.config.threads > 0 and self.config.device == "cpu":
                    import torch
                    torch.set_num_threads(self.config.threads)
                    logger.info(f"CPU threads diatur ke: {self.config.threads}")
                    
            except Exception as e:
                logger.error(f"Gagal load model: {e}")
                raise
        
        return self.model
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
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
        
        # Get file info
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        logger.info(f"Ukuran file: {file_size:.2f} MB")
        
        # Load model jika belum
        model = self.load_model()
        
        # Options untuk transkripsi
        transcribe_options = {
            "task": self.config.task,
            "verbose": self.config.verbose,
            "word_timestamps": self.config.word_timestamps,
            "temperature": self.config.temperature,
            "best_of": self.config.best_of,
            "beam_size": self.config.beam_size,
            "patience": self.config.patience,
            "condition_on_previous_text": False,
            "fp16": False if self.config.compute_type == "float32" else True,
        }
        
        # Gunakan language dari parameter atau config
        use_language = language or self.config.language
        if use_language:
            transcribe_options["language"] = use_language
        
        # Transkripsi
        result = model.transcribe(audio_path, **transcribe_options)
        
        # Tambahkan metadata
        result['metadata'] = {
            'audio_file': os.path.basename(audio_path),
            'audio_path': audio_path,
            'file_size_mb': file_size,
            'audio_duration': self._get_audio_duration(audio_path),
            'transcription_time': time.time() - start_time,
            'model_size': self.config.model_size,
            'language': use_language or result.get('language', 'auto'),
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config)
        }
        
        # Hitung performance
        transcribe_time = result['metadata']['transcription_time']
        audio_duration = result['metadata']['audio_duration']
        
        logger.info(f"Transkripsi selesai dalam {transcribe_time:.1f} detik")
        logger.info(f"Durasi audio: {audio_duration:.1f} detik")
        logger.info(f"Real-time factor: {transcribe_time/audio_duration:.2f}x")
        
        return result
    
    def transcribe_batch(self,
                         audio_paths: List[str],
                         output_dir: Optional[str] = None,
                         language: Optional[str] = None) -> Dict[str, Dict]:
        """
        Transkripsi batch multiple files dengan struktur folder yang terorganisir.
        
        Args:
            audio_paths (list): List path file audio
            output_dir (str): Direktori utama untuk menyimpan hasil
            language (str): Bahasa untuk semua file (None untuk auto-detect)
            
        Returns:
            dict: Dictionary dengan key=filename, value=result
        """
        if not audio_paths:
            logger.warning("Tidak ada file audio untuk diproses")
            return {}
        
        # Gunakan output_dir dari config jika tidak diberikan
        use_output_dir = output_dir or self.config.output_dir
        
        # Buat struktur folder utama
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.structured_output:
            base_output_dir = os.path.join(use_output_dir, f"transcription_{timestamp}")
            transcripts_dir = os.path.join(base_output_dir, "transcripts")
            summaries_dir = os.path.join(base_output_dir, "summaries")
            logs_dir = os.path.join(base_output_dir, "logs")
            
            for dir_path in [transcripts_dir, summaries_dir, logs_dir]:
                os.makedirs(dir_path, exist_ok=True)
        else:
            transcripts_dir = use_output_dir
            summaries_dir = use_output_dir
            logs_dir = use_output_dir
            os.makedirs(use_output_dir, exist_ok=True)
        
        results = {}
        total_start_time = time.time()
        
        logger.info(f"Memulai batch processing {len(audio_paths)} file...")
        logger.info(f"Output akan disimpan di: {transcripts_dir}")
        
        for i, audio_path in enumerate(audio_paths, 1):
            if not os.path.exists(audio_path):
                logger.warning(f"File tidak ditemukan, skip: {audio_path}")
                continue
            
            logger.info(f"Processing file {i}/{len(audio_paths)}: {os.path.basename(audio_path)}")
            file_start_time = time.time()
            
            try:
                # Transkripsi individual
                result = self.transcribe(audio_path, language)
                
                # Generate nama file dan folder terstruktur
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                
                if self.config.structured_output:
                    file_output_dir = os.path.join(transcripts_dir, base_name)
                    os.makedirs(file_output_dir, exist_ok=True)
                    output_base = os.path.join(file_output_dir, base_name)
                else:
                    output_base = os.path.join(transcripts_dir, base_name)
                
                # Simpan dalam berbagai format
                saved_files = self.save_transcript(result, output_base)
                
                file_time = time.time() - file_start_time
                logger.info(f"File selesai dalam {file_time:.1f} detik")
                
                results[audio_path] = {
                    'result': result,
                    'output_files': saved_files,
                    'processing_time': file_time
                }
                
            except Exception as e:
                logger.error(f"Gagal transkripsi {audio_path}: {e}")
                results[audio_path] = {'error': str(e)}
        
        total_time = time.time() - total_start_time
        
        # Simpan log processing
        if self.config.structured_output:
            self._save_processing_log(logs_dir, timestamp, results, total_time)
        
        logger.info(f"Batch processing selesai dalam {total_time:.1f} detik")
        logger.info(f"Rata-rata per file: {total_time/len(audio_paths):.1f} detik")
        
        # Simpan batch summary
        self._save_batch_summary(results, summaries_dir)
        
        # Buat file README dengan struktur folder
        if self.config.structured_output:
            self._create_readme(base_output_dir, results, total_time)
        
        return results
    
    def transcribe_directory(self,
                             directory_path: str,
                             file_pattern: str = "*.mp3",
                             recursive: bool = False,
                             **kwargs) -> Dict[str, Dict]:
        """
        Transkripsi semua file dalam direktori.
        
        Args:
            directory_path (str): Path ke direktori
            file_pattern (str): Pattern untuk mencari file audio
            recursive (bool): Cari file secara recursive
            **kwargs: Argument untuk transcribe_batch
            
        Returns:
            dict: Hasil batch processing
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Direktori tidak ditemukan: {directory_path}")
        
        # Cari semua file audio
        audio_files = self._find_audio_files(directory_path, file_pattern, recursive)
        
        if not audio_files:
            logger.warning(f"Tidak ditemukan file audio di {directory_path}")
            return {}
        
        logger.info(f"Menemukan {len(audio_files)} file audio")
        
        # Buat output directory berdasarkan nama folder sumber
        if 'output_dir' not in kwargs:
            source_dir_name = os.path.basename(os.path.normpath(directory_path))
            timestamp = datetime.now().strftime("%Y%m%d")
            default_output = f"{self.config.output_dir}/{source_dir_name}_{timestamp}"
            kwargs['output_dir'] = default_output
        
        return self.transcribe_batch(audio_files, **kwargs)
    
    def transcribe_from_file_list(self,
                                 list_file_path: str,
                                 **kwargs) -> Dict[str, Dict]:
        """
        Transkripsi dari file yang berisi list path audio.
        
        Args:
            list_file_path (str): Path ke file teks yang berisi list audio files
            **kwargs: Argument untuk transcribe_batch
            
        Returns:
            dict: Hasil batch processing
        """
        if not os.path.exists(list_file_path):
            raise FileNotFoundError(f"File list tidak ditemukan: {list_file_path}")
        
        with open(list_file_path, 'r', encoding='utf-8') as f:
            audio_files = [line.strip() for line in f if line.strip()]
        
        # Filter hanya file yang ada
        valid_files = [f for f in audio_files if os.path.exists(f)]
        missing_files = [f for f in audio_files if not os.path.exists(f)]
        
        if missing_files:
            logger.warning(f"{len(missing_files)} file tidak ditemukan")
            for f in missing_files[:5]:  # Tampilkan maksimal 5
                logger.warning(f"  - {f}")
            if len(missing_files) > 5:
                logger.warning(f"  ... dan {len(missing_files)-5} lainnya")
        
        if not valid_files:
            logger.error("Tidak ada file yang valid untuk diproses")
            return {}
        
        logger.info(f"Memproses {len(valid_files)} dari {len(audio_files)} file")
        return self.transcribe_batch(valid_files, **kwargs)
    
    def save_transcript(self, result: Dict, output_path: str) -> Dict[str, str]:
        """
        Simpan hasil transkripsi ke berbagai format dengan struktur terorganisir.
        
        Args:
            result (dict): Hasil dari Whisper
            output_path (str): Path dasar untuk output (tanpa ekstensi)
            
        Returns:
            dict: Dictionary dengan format dan path file
        """
        saved_files = {}
        
        if self.config.structured_output:
            saved_files = self._save_structured_output(result, output_path)
        else:
            saved_files = self._save_flat_output(result, output_path)
        
        self._log_saved_files(saved_files)
        return saved_files
    
    def _find_audio_files(self, directory_path: str, file_pattern: str, recursive: bool) -> List[str]:
        """Cari semua file audio dalam direktori"""
        pattern = os.path.join(directory_path, file_pattern)
        
        if recursive:
            audio_files = glob.glob(pattern, recursive=True)
        else:
            audio_files = glob.glob(pattern)
        
        # Tambahkan format audio lain
        additional_patterns = ["*.wav", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.opus", "*.mp4", "*.m4b"]
        for pattern in additional_patterns:
            search_pattern = os.path.join(directory_path, pattern)
            if recursive:
                audio_files.extend(glob.glob(search_pattern, recursive=True))
            else:
                audio_files.extend(glob.glob(search_pattern))
        
        # Hapus duplikat dan urutkan
        return sorted(list(set(audio_files)))
    
    def _save_structured_output(self, result: Dict, output_path: str) -> Dict[str, str]:
        """Simpan output dengan struktur folder"""
        saved_files = {}
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        
        # Buat subdirectories untuk berbagai format
        dirs = {
            'json': os.path.join(base_dir, "json"),
            'text': os.path.join(base_dir, "text"),
            'subtitles': os.path.join(base_dir, "subtitles"),
            'data': os.path.join(base_dir, "data")
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 1. Simpan sebagai JSON lengkap
        json_path = os.path.join(dirs['json'], f"{base_name}_full.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved_files['json_full'] = json_path
        
        # 2. Simpan sebagai SRT (subtitle)
        srt_path = os.path.join(dirs['subtitles'], f"{base_name}.srt")
        self._save_srt(result['segments'], srt_path)
        saved_files['srt'] = srt_path
        
        # 3. Simpan sebagai text sederhana
        txt_path = os.path.join(dirs['text'], f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        saved_files['txt'] = txt_path
        
        # 4. Simpan sebagai WebVTT
        vtt_path = os.path.join(dirs['subtitles'], f"{base_name}.vtt")
        self._save_vtt(result['segments'], vtt_path)
        saved_files['vtt'] = vtt_path
        
        # 5. Simpan sebagai Word-by-Word JSON
        word_json_path = os.path.join(dirs['data'], f"{base_name}_words.json")
        self._save_word_json(result, word_json_path)
        saved_files['word_json'] = word_json_path
        
        # 6. Simpan sebagai Clean JSON
        clean_json_path = os.path.join(dirs['data'], f"{base_name}_clean.json")
        self._save_clean_json(result, clean_json_path)
        saved_files['clean_json'] = clean_json_path
        
        # 7. Simpan metadata terpisah
        meta_path = os.path.join(dirs['data'], f"{base_name}_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(result.get('metadata', {}), f, indent=2, ensure_ascii=False)
        saved_files['metadata'] = meta_path
        
        # 8. Simpan sebagai Markdown dengan timestamp
        md_path = os.path.join(dirs['text'], f"{base_name}.md")
        self._save_markdown(result, md_path)
        saved_files['markdown'] = md_path
        
        return saved_files
    
    def _save_flat_output(self, result: Dict, output_path: str) -> Dict[str, str]:
        """Simpan output dalam format flat (tanpa subfolder)"""
        saved_files = {}
        
        # 1. Simpan sebagai JSON lengkap
        json_path = f"{output_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved_files['json'] = json_path
        
        # 2. Simpan sebagai SRT (subtitle)
        srt_path = f"{output_path}.srt"
        self._save_srt(result['segments'], srt_path)
        saved_files['srt'] = srt_path
        
        # 3. Simpan sebagai text sederhana
        txt_path = f"{output_path}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        saved_files['txt'] = txt_path
        
        # 4. Simpan sebagai WebVTT
        vtt_path = f"{output_path}.vtt"
        self._save_vtt(result['segments'], vtt_path)
        saved_files['vtt'] = vtt_path
        
        # 5. Simpan sebagai Word-by-Word JSON
        word_json_path = f"{output_path}_words.json"
        self._save_word_json(result, word_json_path)
        saved_files['word_json'] = word_json_path
        
        # 6. Simpan sebagai Clean JSON
        clean_json_path = f"{output_path}_clean.json"
        self._save_clean_json(result, clean_json_path)
        saved_files['clean_json'] = clean_json_path
        
        return saved_files
    
    def _save_markdown(self, result: Dict, md_path: str):
        """Save transcript as formatted markdown with timestamps"""
        metadata = result.get('metadata', {})
        
        md_content = [
            f"# Transcript: {metadata.get('audio_file', 'Unknown')}",
            f"",
            f"**Language**: {result.get('language', 'unknown')}",
            f"**Transcription Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration**: {metadata.get('audio_duration', 0):.1f} seconds",
            f"**File Size**: {metadata.get('file_size_mb', 0):.2f} MB",
            f"",
            f"---",
            f"",
            f"## Full Transcript",
            f"",
        ]
        
        for i, segment in enumerate(result['segments'], 1):
            start_time = self._format_time_human(segment['start'])
            md_content.append(f"**[{start_time}]** {segment['text'].strip()}")
            md_content.append("")
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
    
    def _save_srt(self, segments: List[Dict], srt_path: str):
        """Convert Whisper segments ke format SRT"""
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start = self._format_srt_timestamp(segment['start'])
            end = self._format_srt_timestamp(segment['end'])
            
            srt_content.extend([
                str(i),
                f"{start} --> {end}",
                segment['text'].strip(),
                ""
            ])
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
    
    def _save_vtt(self, segments: List[Dict], vtt_path: str):
        """Convert Whisper segments ke format WebVTT"""
        vtt_content = ["WEBVTT", ""]
        
        for i, segment in enumerate(segments, 1):
            start = self._format_vtt_timestamp(segment['start'])
            end = self._format_vtt_timestamp(segment['end'])
            
            vtt_content.extend([
                str(i),
                f"{start} --> {end}",
                segment['text'].strip(),
                ""
            ])
        
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_content))
    
    def _save_word_json(self, result: Dict, output_path: str):
        """Save word-level JSON dengan timestamp per kata"""
        word_data = {
            'metadata': result.get('metadata', {}),
            'full_text': result['text'],
            'language': result.get('language', 'unknown'),
            'words': []
        }
        
        # Ekstrak kata-kata dari segments
        for segment in result['segments']:
            if 'words' in segment:
                for word_info in segment['words']:
                    word_data['words'].append({
                        'word': word_info['word'],
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'confidence': word_info.get('probability', 0.0),
                        'segment_id': segment.get('id')
                    })
            else:
                # Jika tidak ada word timestamps, split berdasarkan spasi
                words = segment['text'].split()
                word_count = len(words)
                segment_duration = segment['end'] - segment['start']
                word_duration = segment_duration / word_count if word_count > 0 else 0
                
                for j, word in enumerate(words):
                    start_time = segment['start'] + (j * word_duration)
                    end_time = start_time + word_duration
                    
                    word_data['words'].append({
                        'word': word,
                        'start': start_time,
                        'end': end_time,
                        'confidence': segment.get('confidence', 0.0),
                        'segment_id': segment.get('id')
                    })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(word_data, f, indent=2, ensure_ascii=False)
    
    def _save_clean_json(self, result: Dict, output_path: str):
        """Save simplified JSON dengan hanya data penting"""
        clean_data = {
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': [],
            'metadata': result.get('metadata', {})
        }
        
        for segment in result['segments']:
            clean_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('confidence', 0.0)
            }
            
            # Tambahkan word timestamps jika ada
            if 'words' in segment:
                clean_segment['words'] = [
                    {
                        'word': w['word'],
                        'start': w['start'],
                        'end': w['end']
                    }
                    for w in segment['words']
                ]
            
            clean_data['segments'].append(clean_segment)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
    
    def _save_processing_log(self, logs_dir: str, timestamp: str, results: Dict, total_time: float):
        """Simpan log processing"""
        log_file = os.path.join(logs_dir, f"processing_log_{timestamp}.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Batch Processing Log - {timestamp}\n")
            f.write("="*50 + "\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Total processing time: {total_time:.1f} detik\n")
            f.write("\nFile Details:\n")
            
            for file_path, data in results.items():
                if 'result' in data:
                    f.write(f"- {os.path.basename(file_path)}: {data['processing_time']:.1f}s\n")
                else:
                    f.write(f"- {os.path.basename(file_path)}: ERROR - {data.get('error', 'Unknown')}\n")
    
    def _save_batch_summary(self, results: Dict, output_dir: str):
        """Save summary dari batch processing"""
        summary = {
            'batch_info': {
                'total_files': len(results),
                'successful_files': sum(1 for r in results.values() if 'result' in r),
                'failed_files': sum(1 for r in results.values() if 'error' in r),
                'processing_date': datetime.now().isoformat()
            },
            'performance_metrics': {},
            'files': []
        }
        
        total_audio_duration = 0
        total_processing_time = 0
        processing_times = []
        
        for file_path, data in results.items():
            file_info = {
                'filename': os.path.basename(file_path),
                'status': 'success' if 'result' in data else 'failed',
                'path': file_path
            }
            
            if 'result' in data:
                result_data = data['result']
                duration = result_data['metadata']['audio_duration']
                proc_time = data['processing_time']
                
                file_info.update({
                    'duration': duration,
                    'processing_time': proc_time,
                    'language': result_data.get('language', 'unknown'),
                    'word_count': len(result_data['text'].split()),
                    'real_time_factor': proc_time / duration if duration > 0 else 0
                })
                
                total_audio_duration += duration
                total_processing_time += proc_time
                processing_times.append(proc_time)
            
            elif 'error' in data:
                file_info['error'] = data['error']
            
            summary['files'].append(file_info)
        
        if summary['batch_info']['successful_files'] > 0:
            summary['performance_metrics'] = {
                'total_audio_duration': total_audio_duration,
                'total_processing_time': total_processing_time,
                'average_processing_time': total_processing_time / summary['batch_info']['successful_files'],
                'average_real_time_factor': total_processing_time / total_audio_duration,
                'fastest_processing': min(processing_times) if processing_times else 0,
                'slowest_processing': max(processing_times) if processing_times else 0,
                'total_words': sum(f.get('word_count', 0) for f in summary['files'] if f['status'] == 'success')
            }
        
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Simpan juga sebagai CSV untuk analisis
        self._save_summary_csv(summary, output_dir)
        
        logger.info(f"Batch summary disimpan: {summary_path}")
        
        # Print summary ke console
        self._print_batch_summary(summary, total_audio_duration, total_processing_time)
    
    def _save_summary_csv(self, summary: Dict, output_dir: str):
        """Save summary sebagai CSV"""
        import csv
        
        csv_path = os.path.join(output_dir, 'batch_summary.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Batch Summary'])
            writer.writerow(['Date', summary['batch_info']['processing_date']])
            writer.writerow(['Total Files', summary['batch_info']['total_files']])
            writer.writerow(['Successful', summary['batch_info']['successful_files']])
            writer.writerow(['Failed', summary['batch_info']['failed_files']])
            writer.writerow([])
            
            # Performance metrics
            writer.writerow(['Performance Metrics'])
            for key, value in summary['performance_metrics'].items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # File details
            writer.writerow(['File Details'])
            writer.writerow(['Filename', 'Status', 'Duration (s)', 'Processing Time (s)', 
                           'Word Count', 'Language', 'Real-time Factor', 'Error'])
            
            for file_info in summary['files']:
                writer.writerow([
                    file_info['filename'],
                    file_info['status'],
                    file_info.get('duration', ''),
                    file_info.get('processing_time', ''),
                    file_info.get('word_count', ''),
                    file_info.get('language', ''),
                    file_info.get('real_time_factor', ''),
                    file_info.get('error', '')
                ])
    
    def _print_batch_summary(self, summary: Dict, total_audio_duration: float, total_processing_time: float):
        """Print summary ke console"""
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY".center(60))
        print("="*60)
        print(f"Total files: {summary['batch_info']['total_files']}")
        print(f"Successful: {summary['batch_info']['successful_files']}")
        print(f"Failed: {summary['batch_info']['failed_files']}")
        
        if summary['batch_info']['successful_files'] > 0:
            print(f"Total audio duration: {total_audio_duration/60:.1f} menit")
            print(f"Total processing time: {total_processing_time/60:.1f} menit")
            print(f"Average RT factor: {summary['performance_metrics'].get('average_real_time_factor', 0):.2f}x")
            print(f"Total words transcribed: {summary['performance_metrics'].get('total_words', 0)}")
        
        print("="*60)
    
    def _create_readme(self, base_dir: str, results: Dict, total_time: float):
        """Buat file README dengan informasi struktur folder"""
        readme_path = os.path.join(base_dir, "README.md")
        
        successful = sum(1 for r in results.values() if 'result' in r)
        failed = sum(1 for r in results.values() if 'error' in r)
        
        readme_content = f"""# Transcription Results

## Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Files Processed**: {len(results)}
- **Successful Transcriptions**: {successful}
- **Failed Transcriptions**: {failed}
- **Total Processing Time**: {total_time:.1f} seconds

## Folder Structure
{os.path.basename(base_dir)}/
├── transcripts/ # Folder untuk setiap file transkripsi
│ ├── [audio_filename_1]/ # Folder per file audio
│ │ ├── json/ # Format JSON lengkap
│ │ ├── text/ # Format teks (TXT, MD)
│ │ ├── subtitles/ # Format subtitle (SRT, VTT)
│ │ └── data/ # Data tambahan (metadata, clean JSON, word JSON)
│ ├── [audio_filename_2]/
│ └── ...
├── summaries/ # Summary batch processing
├── logs/ # Log file processing
└── README.md # File ini


## File Formats
Setiap transkripsi menghasilkan beberapa format file:

### Dalam folder json/
- `[filename]_full.json` - Full output dari Whisper dengan semua metadata

### Dalam folder text/
- `[filename].txt` - Teks transkripsi mentah
- `[filename].md` - Transkripsi dengan format markdown dan timestamp

### Dalam folder subtitles/
- `[filename].srt` - Subtitle format SRT
- `[filename].vtt` - Subtitle format WebVTT

### Dalam folder data/
- `[filename]_clean.json` - JSON sederhana dengan teks dan segments
- `[filename]_words.json` - JSON dengan timestamp per kata
- `[filename]_metadata.json` - Metadata transkripsi

## Model Information
- **Model**: Whisper {self.config.model_size}
- **Language Detection**: {'Auto' if not self.config.language else self.config.language}
- **Word Timestamps**: {'Enabled' if self.config.word_timestamps else 'Disabled'}

## Usage Notes
1. File JSON lengkap berisi semua output dari model Whisper
2. Format SRT dan VTT cocok untuk video/audio player
3. File _clean.json cocok untuk analisis data sederhana
4. File _words.json berguna untuk aplikasi yang membutuhkan timestamp per kata

## Processing Summary
Lihat `summaries/batch_summary.json` untuk detail lengkap processing.
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"README created: {readme_path}")
    
    def _log_saved_files(self, saved_files: Dict[str, str]):
        """Log informasi file yang disimpan"""
        logger.info(f"Transcript disimpan dalam {len(saved_files)} format:")
        
        for format_name, file_path in saved_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # dalam KB
                logger.info(f"  - {format_name.upper()}: {file_path} ({file_size:.1f} KB)")
    
    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"
    
    @staticmethod
    def _format_vtt_timestamp(seconds: float) -> str:
        """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{milliseconds:03d}"
    
    @staticmethod
    def _format_time_human(seconds: float) -> str:
        """Format waktu untuk display human readable"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        import subprocess
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            return duration if duration > 0 else 0
            
        except Exception as e:
            logger.warning(f"Gagal mendapatkan durasi audio {audio_path}: {e}")
            return 0


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Podcast Transcriber - Transkripsi audio menggunakan Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  %(prog)s --input audio.mp3                          # Single file
  %(prog)s --input "folder/*.mp3" --batch            # Batch files
  %(prog)s --input podcast_folder --dir              # Entire directory
  %(prog)s --input filelist.txt --filelist           # From file list
  %(prog)s --config config.json                      # Use config file
  %(prog)s --list-models                             # List available models
        """
    )
    
    # Mode input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help="Input file, folder, atau pattern"
    )
    input_group.add_argument(
        '--config',
        type=str,
        help="File konfigurasi JSON"
    )
    
    # Mode operasi
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--batch', '-b',
        action='store_true',
        help="Mode batch processing (gunakan dengan pattern)"
    )
    mode_group.add_argument(
        '--dir', '-d',
        action='store_true',
        help="Mode directory processing"
    )
    mode_group.add_argument(
        '--filelist', '-f',
        action='store_true',
        help="Mode file list processing"
    )
    mode_group.add_argument(
        '--list-models',
        action='store_true',
        help="List available Whisper models"
    )
    mode_group.add_argument(
        '--test',
        action='store_true',
        help="Run tests"
    )
    
    # Model parameters
    parser.add_argument(
        '--model', '-m',
        type=str,
        default="base",
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help="Ukuran model Whisper"
    )
    parser.add_argument(
        '--language', '-l',
        type=str,
        help="Kode bahasa (contoh: id, en, ja, dll)"
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help="Device untuk inference (cpu atau cuda)"
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=0,
        help="Jumlah CPU threads (0 untuk auto-detect)"
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="transcripts",
        help="Direktori output"
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help="Gunakan output flat (tanpa struktur folder)"
    )
    parser.add_argument(
        '--no-word-timestamps',
        action='store_true',
        help="Nonaktifkan word timestamps"
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default="*.mp3",
        help="Pattern untuk mencari file audio (default: *.mp3)"
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help="Cari file secara recursive"
    )
    
    # Performance options
    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help="Beam size untuk decoding"
    )
    parser.add_argument(
        '--best-of',
        type=int,
        default=5,
        help="Best of parameter"
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output"
    )
    parser.add_argument(
        '--log',
        type=str,
        help="File log output"
    )
    
    return parser.parse_args()


def load_config_from_file(config_file: str) -> Dict:
    """Load konfigurasi dari file JSON"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logger.info(f"Konfigurasi dimuat dari: {config_file}")
        return config_data
    except Exception as e:
        logger.error(f"Gagal memuat konfigurasi: {e}")
        return {}


def list_available_models():
    """List semua model Whisper yang tersedia"""
    print("\nAvailable Whisper Models:")
    print("-" * 40)
    models = [
        ("tiny", "39M parameters", "Fastest, lowest accuracy"),
        ("base", "74M parameters", "Good balance of speed/accuracy"),
        ("small", "244M parameters", "Better accuracy, moderate speed"),
        ("medium", "769M parameters", "High accuracy, slower"),
        ("large", "1550M parameters", "Highest accuracy, slowest"),
        ("large-v2", "1550M parameters", "Improved large model"),
        ("large-v3", "1550M parameters", "Latest large model")
    ]
    
    for name, size, desc in models:
        print(f"{name:12} {size:20} - {desc}")
    
    print("\nRecommended: 'base' untuk kebanyakan penggunaan")
    print("            'large-v3' untuk akurasi tertinggi")


def main():
    """Main entry point dengan command line interface"""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(args.log)] if args.log else [])
        ]
    )
    
    # List models jika diminta
    if args.list_models:
        list_available_models()
        return
    
    # Load config dari file jika diberikan
    if args.config:
        config_data = load_config_from_file(args.config)
        # Override dengan command line arguments
        for arg in vars(args):
            arg_value = getattr(args, arg)
            if arg_value is not None and arg != 'config':
                config_data[arg] = arg_value
        config = TranscriptionConfig.from_dict(config_data)
    else:
        # Buat config dari command line arguments
        config = TranscriptionConfig(
            model_size=args.model,
            language=args.language,
            word_timestamps=not args.no_word_timestamps,
            structured_output=not args.flat,
            output_dir=args.output,
            device=args.device,
            threads=args.threads,
            beam_size=args.beam_size,
            best_of=args.best_of,
            verbose=args.verbose
        )
    
    # Inisialisasi transcriber
    transcriber = PodcastTranscriber(config)
    
    # Run tests jika diminta
    if args.test:
        run_all_tests(transcriber)
        return
    
    # Cek input
    if not args.input and not args.test:
        logger.error("Harap berikan input dengan --input atau gunakan --test")
        return
    
    try:
        # Proses berdasarkan mode
        if args.batch:
            # Batch mode dengan pattern
            audio_files = glob.glob(args.input, recursive=args.recursive)
            if not audio_files:
                logger.error(f"Tidak ditemukan file dengan pattern: {args.input}")
                return
            
            logger.info(f"Menemukan {len(audio_files)} file untuk diproses")
            results = transcriber.transcribe_batch(
                audio_files,
                output_dir=args.output,
                language=args.language
            )
            
        elif args.dir:
            # Directory mode
            if not os.path.isdir(args.input):
                logger.error(f"Input bukan direktori: {args.input}")
                return
            
            results = transcriber.transcribe_directory(
                directory_path=args.input,
                file_pattern=args.pattern,
                recursive=args.recursive,
                output_dir=args.output,
                language=args.language
            )
            
        elif args.filelist:
            # File list mode
            results = transcriber.transcribe_from_file_list(
                list_file_path=args.input,
                output_dir=args.output,
                language=args.language
            )
            
        else:
            # Single file mode (default)
            if not os.path.isfile(args.input):
                logger.error(f"File tidak ditemukan: {args.input}")
                return
            
            result = transcriber.transcribe(args.input, args.language)
            
            # Save hasil
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_base = os.path.join(args.output, base_name)
            os.makedirs(os.path.dirname(output_base), exist_ok=True)
            
            saved_files = transcriber.save_transcript(result, output_base)
            
            # Tampilkan info
            print("\n" + "="*60)
            print("TRANSCRIPTION COMPLETE".center(60))
            print("="*60)
            print(f"File: {os.path.basename(args.input)}")
            print(f"Language: {result.get('language', 'unknown')}")
            print(f"Duration: {result['metadata']['audio_duration']:.1f}s")
            print(f"Processing time: {result['metadata']['transcription_time']:.1f}s")
            print(f"Real-time factor: {result['metadata']['transcription_time']/result['metadata']['audio_duration']:.2f}x")
            print(f"Output saved to: {args.output}")
            print("="*60)
            
            # Tampilkan preview
            print("\nTranscript preview:")
            print("-" * 60)
            for segment in result['segments'][:3]:  # Tampilkan 3 segment pertama
                start_time = transcriber._format_time_human(segment['start'])
                print(f"[{start_time}] {segment['text'].strip()}")
            if len(result['segments']) > 3:
                print(f"... and {len(result['segments']) - 3} more segments")
            
            return result
    
    except KeyboardInterrupt:
        logger.info("\nTranscription interrupted by user")
        return
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return


# ============================================================================
# TEST FUNCTIONS (dipisahkan untuk modularitas)
# ============================================================================

def run_all_tests(transcriber=None):
    """Run semua test functions"""
    import shutil
    
    # Setup transcriber jika tidak diberikan
    if transcriber is None:
        config = TranscriptionConfig(model_size="base", structured_output=True)
        transcriber = PodcastTranscriber(config)
    
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    # Buat test directory
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Buat sample audio files dummy
    sample_audio = None
    
    # Cari file audio sample
    sample_locations = [
        "data/samples/sample_podcast_audio.mp3",
        "sample.mp3",
        "test.mp3"
    ]
    
    for loc in sample_locations:
        if os.path.exists(loc):
            sample_audio = loc
            break
    
    if not sample_audio:
        print("✗ Tidak ada file audio untuk testing")
        print("  Buat file audio sample di data/samples/sample_podcast_audio.mp3")
        return
    
    # Test 1: Single file
    print("\n1. Testing single file transcription...")
    try:
        result = transcriber.transcribe(sample_audio)
        print(f"✓ Single file test passed")
        print(f"  Language: {result.get('language', 'unknown')}")
        print(f"  Segments: {len(result['segments'])}")
    except Exception as e:
        print(f"✗ Single file test failed: {e}")
    
    # Test 2: Batch processing
    print("\n2. Testing batch processing...")
    try:
        # Buat beberapa test files
        batch_dir = os.path.join(test_dir, "batch")
        os.makedirs(batch_dir, exist_ok=True)
        
        for i in range(2):
            dst = os.path.join(batch_dir, f"test_{i}.mp3")
            shutil.copy(sample_audio, dst)
        
        audio_files = glob.glob(os.path.join(batch_dir, "*.mp3"))
        results = transcriber.transcribe_batch(audio_files, output_dir=os.path.join(test_dir, "output"))
        print(f"✓ Batch test passed - Processed {len(results)} files")
    except Exception as e:
        print(f"✗ Batch test failed: {e}")
    
    # Test 3: Directory processing
    print("\n3. Testing directory processing...")
    try:
        dir_results = transcriber.transcribe_directory(
            batch_dir,
            output_dir=os.path.join(test_dir, "dir_output")
        )
        print(f"✓ Directory test passed")
    except Exception as e:
        print(f"✗ Directory test failed: {e}")
    
    # Test 4: File list
    print("\n4. Testing file list processing...")
    try:
        # Buat file list
        filelist_path = os.path.join(test_dir, "filelist.txt")
        with open(filelist_path, 'w') as f:
            f.write(sample_audio + "\n")
        
        list_results = transcriber.transcribe_from_file_list(
            filelist_path,
            output_dir=os.path.join(test_dir, "list_output")
        )
        print(f"✓ File list test passed")
    except Exception as e:
        print(f"✗ File list test failed: {e}")
    
    # Test 5: Different output formats
    print("\n5. Testing different output formats...")
    try:
        config_flat = TranscriptionConfig(structured_output=False)
        transcriber_flat = PodcastTranscriber(config_flat)
        result = transcriber_flat.transcribe(sample_audio)
        
        # Save dengan format flat
        output_base = os.path.join(test_dir, "flat_output", "test")
        os.makedirs(os.path.dirname(output_base), exist_ok=True)
        saved_files = transcriber_flat.save_transcript(result, output_base)
        
        print(f"✓ Flat output test passed")
        print(f"  Generated files: {len(saved_files)}")
    except Exception as e:
        print(f"✗ Output format test failed: {e}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    
    # Bersihkan (opsional)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    main()