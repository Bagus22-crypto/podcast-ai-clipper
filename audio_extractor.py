"""
Podcast Audio Extractor
Script untuk mengekstrak audio dari video podcast dengan berbagai format dan enhancement.
Struktur folder output yang jelas untuk memudahkan manajemen.

Fitur:
- Ekstraksi audio dari berbagai format video
- Enhancement audio (normalisasi, noise reduction, dll)
- Konversi ke berbagai format audio
- Manajemen folder yang terstruktur
- Logging dan pelaporan
- Antarmuka menu interaktif
"""

import os
import glob
import shutil
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

# ============================================================================
# KONFIGURASI LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# KONSTANTA DAN KONFIGURASI
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_FOLDER = os.path.join(PROJECT_ROOT, "podcast_audio_output")

DEFAULT_STRUCTURE = {
    "raw_audio": "01_raw_audio",           # Audio mentah hasil ekstraksi
    "enhanced_audio": "02_enhanced_audio", # Audio yang sudah diproses
    "multiformat": "03_multiformat",       # Audio dalam berbagai format
    "processed_videos": "04_processed_videos",  # Video yang sudah diproses
    "logs": "05_logs",                     # Log file
    "metadata": "06_metadata",             # Metadata dan report
    "temp": "07_temp",                     # File sementara
    "backup": "08_backup"                  # Backup
}

SUPPORTED_VIDEO_FORMATS = [
    '*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv',
    '*.MP4', '*.AVI', '*.MOV', '*.MKV', '*.FLV', '*.WMV'
]

AUDIO_FORMAT_SETTINGS = {
    "mp3": {"codec": "mp3", "ext": "mp3", "ffmpeg_params": []},
    "wav": {"codec": "pcm_s16le", "ext": "wav", "ffmpeg_params": []},
    "aac": {"codec": "aac", "ext": "m4a", "ffmpeg_params": ["-strict", "experimental"]},
    "ogg": {"codec": "libvorbis", "ext": "ogg", "ffmpeg_params": []},
    "flac": {"codec": "flac", "ext": "flac", "ffmpeg_params": []},
}

BITRATE_MAP = {
    "high": {"mp3": "320k", "aac": "256k", "ogg": "192k", "wav": None, "flac": None},
    "medium": {"mp3": "192k", "aac": "192k", "ogg": "128k", "wav": None, "flac": None},
    "low": {"mp3": "128k", "aac": "128k", "ogg": "96k", "wav": None, "flac": None}
}


class FolderManager:
    """Mengelola struktur folder untuk podcast audio extractor"""
    
    def __init__(self, base_folder: str = DEFAULT_BASE_FOLDER):
        """
        Inisialisasi FolderManager.
        
        Args:
            base_folder: Path folder utama untuk output
        """
        self.base_folder = base_folder
        self.structure = DEFAULT_STRUCTURE
        self.setup_folders()
    
    def setup_folders(self) -> None:
        """Membuat semua folder yang diperlukan dalam struktur"""
        print(f"\n{'='*60}")
        print("📁 SETTING UP FOLDER STRUCTURE")
        print(f"{'='*60}")
        print(f"Base folder: {self.base_folder}")
        
        # Buat folder utama
        os.makedirs(self.base_folder, exist_ok=True)
        
        # Buat semua subfolder
        for folder_name in self.structure.values():
            folder_path = os.path.join(self.base_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            print(f"  ✓ Created: {folder_name}/")
        
        # Buat file README
        self.create_readme()
        print(f"\n✅ Folder structure ready!")
        print(f"{'='*60}")
    
    def create_readme(self) -> None:
        """Membuat file README dengan penjelasan struktur folder"""
        readme_content = """# PODCAST AUDIO EXTRACTOR - Folder Structure

## 📂 STRUKTUR FOLDER

### 01_raw_audio/
- Audio mentah hasil ekstraksi pertama
- Format: MP3 default
- Nama file: [original_name]_audio.mp3

### 02_enhanced_audio/
- Audio yang sudah diproses (noise reduction, normalize, dll)
- Format: MP3 enhanced
- Nama file: [original_name]_enhanced_[timestamp].mp3

### 03_multiformat/
- Audio dalam berbagai format (MP3, WAV, AAC, OGG, FLAC)
- Format: Sesuai pilihan pengguna
- Nama file: [original_name]_audio.[format]

### 04_processed_videos/
- Video yang sudah berhasil diproses (opsional)
- Untuk tracking video yang sudah diekstrak

### 05_logs/
- Log file proses ekstraksi
- Format: JSON dan TXT
- Nama file: extraction_log_[timestamp].json

### 06_metadata/
- Metadata audio dan video
- Report hasil processing
- Nama file: metadata_[timestamp].json

### 07_temp/
- File sementara selama proses
- Otomatis terhapus setelah proses selesai

### 08_backup/
- Backup dari file penting
- Manual backup jika diperlukan

## 🗂️ CONTOH STRUKTUR
podcast_audio_output/
├── 01_raw_audio/
│   ├── podcast_ep1_audio.mp3
│   └── interview_2023_audio.mp3
├── 02_enhanced_audio/
│   ├── podcast_ep1_enhanced_20231201_143022.mp3
│   └── interview_2023_enhanced_20231201_143025.mp3
├── 03_multiformat/
│   ├── podcast_ep1_audio.wav
│   ├── podcast_ep1_audio.aac
│   └── podcast_ep1_audio.flac
├── 04_processed_videos/
│   └── (video yang sudah diproses)
├── 05_logs/
│   ├── extraction_log_20231201.json
│   └── error_log_20231201.txt
├── 06_metadata/
│   ├── metadata_20231201.json
│   └── processing_report_20231201.md
├── 07_temp/
│   └── (file sementara)
└── 08_backup/
    └── (backup file)

## 📝 CATATAN
- Setiap folder memiliki nomor urut untuk memudahkan navigasi
- Folder 01-03 adalah output utama
- Folder 04-08 adalah supporting folders
- Logs dan metadata membantu tracking proses
"""
        
        readme_path = os.path.join(self.base_folder, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def get_folder_path(self, folder_key: str) -> str:
        """
        Mendapatkan path folder berdasarkan key.
        
        Args:
            folder_key: Key dari folder yang diinginkan
            
        Returns:
            Path lengkap ke folder
        """
        return os.path.join(self.base_folder, self.structure[folder_key])
    
    def get_unique_filename(self, base_name: str, folder_key: str, 
                           extension: str, suffix: str = "") -> str:
        """
        Membuat nama file unik dengan timestamp.
        
        Args:
            base_name: Nama dasar file
            folder_key: Key folder tujuan
            extension: Ekstensi file
            suffix: Suffix tambahan untuk nama file
            
        Returns:
            Path file yang unik
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{suffix}" if suffix else ""
        filename = f"{base_name}{suffix}_{timestamp}.{extension}"
        folder_path = self.get_folder_path(folder_key)
        
        return os.path.join(folder_path, filename)
    
    def list_all_files(self) -> Dict[str, List[str]]:
        """
        Mendapatkan daftar semua file dalam struktur folder.
        
        Returns:
            Dictionary dengan key folder dan list file
        """
        file_list = {}
        for key, folder_name in self.structure.items():
            folder_path = os.path.join(self.base_folder, folder_name)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) 
                        if os.path.isfile(os.path.join(folder_path, f))]
                file_list[key] = files
        
        return file_list


class PodcastAudioExtractor:
    """Main class untuk ekstraksi audio podcast"""
    
    def __init__(self, base_folder: str = DEFAULT_BASE_FOLDER):
        """
        Inisialisasi PodcastAudioExtractor.
        
        Args:
            base_folder: Path folder utama untuk output
        """
        self.folder_manager = FolderManager(base_folder)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging dengan file log terpisah"""
        self.log_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "operations": [],
            "errors": [],
            "warnings": []
        }
        
        logger.info(f"Session started: {self.session_id}")
    
    def log_operation(self, operation: str, status: str, details: Dict[str, Any]) -> None:
        """
        Mencatat operasi ke log.
        
        Args:
            operation: Nama operasi
            status: Status operasi (success/error/warning)
            details: Detail operasi
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "status": status,
            "details": details
        }
        
        self.log_data["operations"].append(log_entry)
        
        if status == "error":
            self.log_data["errors"].append(log_entry)
            logger.error(f"{operation}: {details.get('message', 'Unknown error')}")
        elif status == "warning":
            self.log_data["warnings"].append(log_entry)
            logger.warning(f"{operation}: {details.get('message', 'Warning')}")
        else:
            logger.info(f"{operation}: {status}")
    
    def save_logs(self) -> str:
        """
        Menyimpan log ke file JSON.
        
        Returns:
            Path ke file log yang disimpan
        """
        log_folder = self.folder_manager.get_folder_path("logs")
        log_file = os.path.join(log_folder, f"extraction_log_{self.session_id}.json")
        
        self.log_data["end_time"] = datetime.now().isoformat()
        self.log_data["total_operations"] = len(self.log_data["operations"])
        self.log_data["total_errors"] = len(self.log_data["errors"])
        self.log_data["total_warnings"] = len(self.log_data["warnings"])
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Logs saved to: {log_file}")
            return log_file
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")
            return ""
    
    def _clean_filename(self, filename: str) -> str:
        """
        Membersihkan nama file dari karakter yang tidak diinginkan.
        
        Args:
            filename: Nama file asli
            
        Returns:
            Nama file yang sudah dibersihkan
        """
        # Hapus ekstensi
        name = os.path.splitext(filename)[0]
        # Ganti spasi dengan underscore dan lowercase
        name = name.replace(" ", "_").lower()
        # Hapus karakter khusus
        name = ''.join(c for c in name if c.isalnum() or c in ['_', '-'])
        
        return name
    
    def extract_audio(self, video_path: str, move_processed: bool = False) -> Dict[str, Any]:
        """
        Mengekstrak audio dari video podcast.
        
        Args:
            video_path: Path ke file video
            move_processed: Pindahkan video ke folder processed setelah selesai
            
        Returns:
            Dictionary dengan hasil ekstraksi
        """
        result = {
            "video": video_path,
            "audio": None,
            "status": "pending",
            "message": "",
            "metadata": {}
        }
        
        try:
            # Validasi input file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            clean_name = self._clean_filename(base_name)
            
            # Dapatkan folder raw_audio
            raw_audio_folder = self.folder_manager.get_folder_path("raw_audio")
            output_audio_path = os.path.join(raw_audio_folder, f"{clean_name}_audio.mp3")
            
            # Hindari nama file duplikat
            counter = 1
            original_output_path = output_audio_path
            while os.path.exists(output_audio_path):
                output_audio_path = original_output_path.replace(
                    ".mp3", f"_{counter}.mp3"
                )
                counter += 1
            
            logger.info(f"📥 Starting extraction: {os.path.basename(video_path)}")
            
            # Load video dan ekstrak audio
            with VideoFileClip(video_path) as video:
                # Extract metadata
                duration = video.duration
                fps = video.fps
                size = video.size
                
                audio = video.audio
                if audio is None:
                    raise Exception("No audio track found in video")
                
                # Simpan sebagai MP3
                logger.info(f"💾 Saving audio to: {os.path.basename(output_audio_path)}")
                audio.write_audiofile(
                    output_audio_path,
                    codec='mp3',
                    bitrate='192k',
                    verbose=False,
                    logger=None
                )
            
            # Verifikasi file output
            if os.path.exists(output_audio_path):
                file_size = os.path.getsize(output_audio_path) / (1024 * 1024)  # MB
                
                # Update result
                result.update({
                    "audio": output_audio_path,
                    "status": "success",
                    "message": "Audio berhasil diekstrak",
                    "metadata": {
                        "original_video": video_path,
                        "duration": round(duration, 2),
                        "fps": fps,
                        "resolution": size,
                        "file_size_mb": round(file_size, 2),
                        "output_path": output_audio_path,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # Pindahkan video ke folder processed jika diminta
                if move_processed and os.path.exists(video_path):
                    processed_folder = self.folder_manager.get_folder_path("processed_videos")
                    video_filename = os.path.basename(video_path)
                    processed_path = os.path.join(processed_folder, video_filename)
                    
                    # Hindari duplikat
                    counter = 1
                    original_processed_path = processed_path
                    while os.path.exists(processed_path):
                        name, ext = os.path.splitext(video_filename)
                        processed_path = original_processed_path.replace(
                            ext, f"_{counter}{ext}"
                        )
                        counter += 1
                    
                    shutil.move(video_path, processed_path)
                    result["video_processed_path"] = processed_path
                
                logger.info(f"✅ Extraction successful! Size: {file_size:.2f} MB")
                
                # Log operation
                self.log_operation("extract_audio", "success", {
                    "video": os.path.basename(video_path),
                    "audio": os.path.basename(output_audio_path),
                    "size_mb": round(file_size, 2),
                    "duration": round(duration, 2)
                })
                
                # Save metadata
                self.save_metadata(result)
                
                return result
                
            else:
                raise Exception("Audio file not created after extraction")
                
        except Exception as e:
            error_msg = f"Failed to extract audio: {str(e)}"
            result.update({
                "status": "failed",
                "message": error_msg
            })
            
            logger.error(error_msg)
            self.log_operation("extract_audio", "error", {
                "video": os.path.basename(video_path) if 'video_path' in locals() else "Unknown",
                "error": str(e)
            })
            
            return result
    
    def batch_extract_audio(self, input_folder: str, file_pattern: str = "*.mp4",
                           move_processed: bool = False) -> List[Dict[str, Any]]:
        """
        Ekstrak audio dari semua video dalam folder.
        
        Args:
            input_folder: Folder berisi file video
            file_pattern: Pattern untuk mencari file video
            move_processed: Pindahkan video setelah diproses
            
        Returns:
            List hasil ekstraksi setiap file
        """
        # Validasi folder input
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Folder input tidak ditemukan: {input_folder}")
        
        # Cari semua file video yang didukung
        video_files = []
        for pattern in SUPPORTED_VIDEO_FORMATS:
            video_files.extend(glob.glob(os.path.join(input_folder, pattern)))
        
        # Juga coba pattern yang diberikan user
        if file_pattern:
            video_files.extend(glob.glob(os.path.join(input_folder, file_pattern)))
        
        # Hapus duplikat
        video_files = list(set(video_files))
        
        if not video_files:
            logger.warning(f"No video files found in folder: {input_folder}")
            return []
        
        results = []
        total_files = len(video_files)
        logger.info(f"📚 Processing {total_files} video files...")
        
        for i, video_file in enumerate(video_files, 1):
            try:
                print(f"\n[{i}/{total_files}] Processing: {os.path.basename(video_file)}")
                
                # Ekstrak audio
                result = self.extract_audio(video_file, move_processed)
                results.append(result)
                
                if result["status"] == "success":
                    print(f"   ✅ Success")
                else:
                    print(f"   ❌ Failed: {result['message']}")
                
            except Exception as e:
                error_result = {
                    "video": video_file,
                    "audio": None,
                    "status": "failed",
                    "message": str(e),
                    "metadata": {}
                }
                results.append(error_result)
                
                logger.error(f"Error processing {os.path.basename(video_file)}: {e}")
        
        # Generate batch report
        self.generate_batch_report(results)
        
        return results
    
    def extract_audio_multiformat(self, video_path: str,
                                 output_formats: Optional[List[str]] = None,
                                 quality: str = "medium") -> Dict[str, Any]:
        """
        Ekstrak audio ke berbagai format.
        
        Args:
            video_path: Path ke file video
            output_formats: List format output (mp3, wav, aac, ogg, flac)
            quality: Kualitas audio (high/medium/low)
            
        Returns:
            Dictionary dengan hasil untuk setiap format
        """
        if output_formats is None:
            output_formats = ["mp3", "wav", "aac"]
        
        result = {
            "video": video_path,
            "formats": {},
            "status": "pending",
            "message": ""
        }
        
        try:
            # Validasi input file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")
            
            # Validasi format output
            valid_formats = [fmt for fmt in output_formats if fmt.lower() in AUDIO_FORMAT_SETTINGS]
            if not valid_formats:
                raise ValueError(f"No valid formats specified. Supported: {list(AUDIO_FORMAT_SETTINGS.keys())}")
            
            # Ekstrak audio dulu ke raw
            extraction_result = self.extract_audio(video_path, move_processed=False)
            if extraction_result["status"] != "success":
                raise Exception(f"Failed to extract base audio: {extraction_result['message']}")
            
            raw_audio_path = extraction_result["audio"]
            
            # Dapatkan folder multiformat
            multiformat_folder = self.folder_manager.get_folder_path("multiformat")
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            clean_name = self._clean_filename(base_name)
            
            logger.info(f"🎵 Converting to {len(valid_formats)} formats: {', '.join(valid_formats)}")
            
            # Konversi ke setiap format
            for fmt in valid_formats:
                fmt_lower = fmt.lower()
                
                # Generate output path
                output_filename = f"{clean_name}_audio.{AUDIO_FORMAT_SETTINGS[fmt_lower]['ext']}"
                output_path = os.path.join(multiformat_folder, output_filename)
                
                # Hindari duplikat
                counter = 1
                original_output_path = output_path
                while os.path.exists(output_path):
                    name, ext = os.path.splitext(output_filename)
                    output_path = original_output_path.replace(
                        ext, f"_{counter}{ext}"
                    )
                    counter += 1
                
                # Load audio
                audio = AudioSegment.from_file(raw_audio_path)
                
                # Set bitrate
                bitrate = None
                if fmt_lower in BITRATE_MAP[quality]:
                    bitrate = BITRATE_MAP[quality][fmt_lower]
                
                # Export ke format yang diinginkan
                logger.info(f"  Converting to {fmt.upper()}...")
                audio.export(
                    output_path,
                    format=AUDIO_FORMAT_SETTINGS[fmt_lower]['ext'],
                    bitrate=bitrate,
                    parameters=AUDIO_FORMAT_SETTINGS[fmt_lower]['ffmpeg_params']
                )
                
                # Verifikasi
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    
                    result["formats"][fmt_lower] = {
                        "path": output_path,
                        "size_mb": round(file_size, 2),
                        "bitrate": bitrate,
                        "status": "success"
                    }
                    
                    logger.info(f"    ✓ {fmt.upper()}: {file_size:.2f} MB")
                else:
                    logger.warning(f"    ✗ {fmt.upper()}: File not created")
            
            # Hapus file audio raw temporary
            try:
                if os.path.exists(raw_audio_path):
                    os.remove(raw_audio_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {raw_audio_path}: {e}")
            
            result.update({
                "status": "success",
                "message": f"Successfully converted to {len(result['formats'])} formats"
            })
            
            self.log_operation("multiformat_extract", "success", {
                "video": os.path.basename(video_path),
                "formats": list(result["formats"].keys()),
                "total_formats": len(result["formats"])
            })
            
            return result
            
        except Exception as e:
            error_msg = f"Failed multi-format conversion: {str(e)}"
            result.update({
                "status": "failed",
                "message": error_msg
            })
            
            logger.error(error_msg)
            self.log_operation("multiformat_extract", "error", {
                "video": os.path.basename(video_path),
                "error": str(e)
            })
            
            return result
    
    def enhance_audio(self, video_path: str,
                     enhancements: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Audio enhancement dengan berbagai filter.
        
        Args:
            video_path: Path ke file video
            enhancements: Dictionary dengan pilihan enhancement
            
        Returns:
            Dictionary dengan hasil enhancement
        """
        if enhancements is None:
            enhancements = {
                "normalize": True,
                "noise_reduction": True,
                "trim_silence": False,
                "compression": False,
                "equalization": False
            }
        
        result = {
            "video": video_path,
            "enhanced_audio": None,
            "enhancements_applied": [],
            "status": "pending",
            "message": ""
        }
        
        try:
            # Ekstrak audio terlebih dahulu
            logger.info("🎚️ Starting audio enhancement...")
            extraction_result = self.extract_audio(video_path, move_processed=False)
            
            if extraction_result["status"] != "success":
                raise Exception(f"Failed to extract audio: {extraction_result['message']}")
            
            raw_audio_path = extraction_result["audio"]
            
            # Load audio dengan pydub
            logger.info("Loading audio for processing...")
            audio = AudioSegment.from_file(raw_audio_path)
            
            original_duration = len(audio) / 1000  # Convert to seconds
            
            # Apply enhancements
            if enhancements.get("normalize", False):
                logger.info("Applying normalization...")
                audio = pydub_normalize(audio)
                result["enhancements_applied"].append("normalize")
            
            if enhancements.get("noise_reduction", False):
                logger.info("Applying noise reduction...")
                # Simple noise reduction dengan filter
                audio = audio.low_pass_filter(3000).high_pass_filter(300)
                result["enhancements_applied"].append("noise_reduction")
            
            if enhancements.get("trim_silence", False):
                logger.info("Trimming silence...")
                audio = audio.strip_silence(
                    silence_len=1000,
                    silence_thresh=-40
                )
                result["enhancements_applied"].append("trim_silence")
            
            if enhancements.get("compression", False):
                logger.info("Applying compression...")
                # Simple compression
                audio = audio.compress_dynamic_range()
                result["enhancements_applied"].append("compression")
            
            # Generate output path di folder enhanced
            enhanced_folder = self.folder_manager.get_folder_path("enhanced_audio")
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            clean_name = self._clean_filename(base_name)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            enhanced_filename = f"{clean_name}_enhanced_{timestamp}.mp3"
            enhanced_path = os.path.join(enhanced_folder, enhanced_filename)
            
            # Simpan hasil
            logger.info(f"Saving enhanced audio to: {enhanced_filename}")
            audio.export(enhanced_path, format="mp3", bitrate="192k")
            
            # Hapus file audio raw
            try:
                if os.path.exists(raw_audio_path):
                    os.remove(raw_audio_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {raw_audio_path}: {e}")
            
            # Verifikasi
            if os.path.exists(enhanced_path):
                file_size = os.path.getsize(enhanced_path) / (1024 * 1024)
                new_duration = len(AudioSegment.from_file(enhanced_path)) / 1000
                
                result.update({
                    "enhanced_audio": enhanced_path,
                    "status": "success",
                    "message": f"Audio enhanced with {len(result['enhancements_applied'])} improvements",
                    "metadata": {
                        "original_video": video_path,
                        "enhancements": result["enhancements_applied"],
                        "file_size_mb": round(file_size, 2),
                        "original_duration": round(original_duration, 2),
                        "enhanced_duration": round(new_duration, 2),
                        "output_path": enhanced_path
                    }
                })
                
                logger.info(f"✅ Enhancement complete! Size: {file_size:.2f} MB")
                
                self.log_operation("audio_enhancement", "success", {
                    "video": os.path.basename(video_path),
                    "enhancements": result["enhancements_applied"],
                    "size_mb": round(file_size, 2),
                    "duration_reduction": round(original_duration - new_duration, 2)
                })
                
                # Save metadata
                self.save_metadata(result)
                
                return result
            
            else:
                raise Exception("Enhanced audio file not created")
                
        except Exception as e:
            error_msg = f"Failed to enhance audio: {str(e)}"
            result.update({
                "status": "failed",
                "message": error_msg
            })
            
            logger.error(error_msg)
            self.log_operation("audio_enhancement", "error", {
                "video": os.path.basename(video_path),
                "error": str(e)
            })
            
            return result
    
    def save_metadata(self, result: Dict[str, Any]) -> str:
        """
        Menyimpan metadata ke file JSON.
        
        Args:
            result: Hasil operasi yang akan disimpan
            
        Returns:
            Path ke file metadata
        """
        try:
            metadata_folder = self.folder_manager.get_folder_path("metadata")
            metadata_file = os.path.join(metadata_folder, f"metadata_{self.session_id}.json")
            
            # Load existing metadata atau buat baru
            metadata = {
                "session_id": self.session_id,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "operations": []
            }
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    metadata["last_updated"] = datetime.now().isoformat()
                except Exception as e:
                    logger.warning(f"Could not load existing metadata: {e}")
            
            # Tambahkan result baru
            metadata["operations"].append({
                "timestamp": datetime.now().isoformat(),
                "operation_type": result.get("operation", "audio_extraction"),
                "result": result
            })
            
            # Simpan
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return metadata_file
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
            return ""
    
    def generate_batch_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Membuat report batch processing dalam format Markdown.
        
        Args:
            results: List hasil ekstraksi
            
        Returns:
            Path ke file report
        """
        try:
            metadata_folder = self.folder_manager.get_folder_path("metadata")
            report_file = os.path.join(metadata_folder, f"batch_report_{self.session_id}.md")
            
            success_count = sum(1 for r in results if r["status"] == "success")
            failed_count = sum(1 for r in results if r["status"] == "failed")
            pending_count = sum(1 for r in results if r["status"] == "pending")
            
            total_size = 0
            for r in results:
                if r.get("metadata") and r["metadata"].get("file_size_mb"):
                    total_size += r["metadata"]["file_size_mb"]
            
            # Buat konten report
            report_content = f"""# BATCH PROCESSING REPORT

## 📊 SUMMARY
- **Session ID**: {self.session_id}
- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Files Processed**: {len(results)}
- **✅ Success**: {success_count}
- **❌ Failed**: {failed_count}
- **⏳ Pending**: {pending_count}
- **Total Audio Size**: {total_size:.2f} MB

## 📁 OUTPUT STRUCTURE
{DEFAULT_BASE_FOLDER}/
├── 01_raw_audio/          # Audio mentah
├── 02_enhanced_audio/     # Audio enhanced
├── 03_multiformat/        # Audio berbagai format
├── 04_processed_videos/   # Video yang sudah diproses
├── 05_logs/               # Log files
├── 06_metadata/           # Metadata dan report
├── 07_temp/               # File sementara
└── 08_backup/             # Backup

## 📋 DETAILED RESULTS

### ✅ SUCCESSFUL ({success_count})
"""
            
            # Tambahkan file yang berhasil
            success_files = [r for r in results if r["status"] == "success"]
            for r in success_files:
                video_name = os.path.basename(r["video"])
                audio_name = os.path.basename(r["audio"]) if r["audio"] else "N/A"
                size = r.get("metadata", {}).get("file_size_mb", 0)
                duration = r.get("metadata", {}).get("duration", 0)
                
                report_content += f"- **{video_name}**  \n"
                report_content += f"  → Output: {audio_name}  \n"
                report_content += f"  → Size: {size:.2f} MB | Duration: {duration:.2f}s\n\n"
            
            # Tambahkan file yang gagal
            if failed_count > 0:
                report_content += f"\n### ❌ FAILED ({failed_count})\n"
                failed_files = [r for r in results if r["status"] == "failed"]
                for r in failed_files:
                    video_name = os.path.basename(r["video"])
                    error_msg = r.get("message", "Unknown error")
                    report_content += f"- **{video_name}**: {error_msg}\n"
            
            # Statistik tambahan
            if success_count > 0:
                avg_size = total_size / success_count
                report_content += f"""
## 📈 STATISTICS

### File Size Distribution
- **Average Size**: {avg_size:.2f} MB
- **Small (< 10 MB)**: {sum(1 for r in success_files if r.get('metadata', {}).get('file_size_mb', 0) < 10)}
- **Medium (10-50 MB)**: {sum(1 for r in success_files if 10 <= r.get('metadata', {}).get('file_size_mb', 0) < 50)}
- **Large (> 50 MB)**: {sum(1 for r in success_files if r.get('metadata', {}).get('file_size_mb', 0) >= 50)}

### Duration Distribution
- **Short (< 5 min)**: {sum(1 for r in success_files if r.get('metadata', {}).get('duration', 0) < 300)}
- **Medium (5-30 min)**: {sum(1 for r in success_files if 300 <= r.get('metadata', {}).get('duration', 0) < 1800)}
- **Long (> 30 min)**: {sum(1 for r in success_files if r.get('metadata', {}).get('duration', 0) >= 1800)}

### Processing Time
- **Start**: {self.log_data.get('start_time', 'N/A')}
- **End**: {datetime.now().isoformat()}
"""
            
            # Rekomendasi
            report_content += f"""
## 🎯 RECOMMENDATIONS
1. {'Check failed files in logs/ folder' if failed_count > 0 else 'All files processed successfully!'}
2. Backup successful files to 08_backup/ folder
3. Clean temp files regularly using cleanup function
4. Review metadata for quality control
5. Consider running enhancement on successful files

---
*Report generated automatically by Podcast Audio Extractor v1.0*
"""
            
            # Simpan report
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"📄 Batch report saved to: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to generate batch report: {e}")
            return ""
    
    def cleanup_temp_files(self) -> Dict[str, Any]:
        """
        Membersihkan file temporary dari folder temp.
        
        Returns:
            Dictionary dengan hasil cleanup
        """
        result = {
            "deleted_files": 0,
            "deleted_size_mb": 0,
            "status": "pending",
            "message": ""
        }
        
        try:
            temp_folder = self.folder_manager.get_folder_path("temp")
            
            if not os.path.exists(temp_folder):
                result.update({
                    "status": "success",
                    "message": "Temp folder does not exist"
                })
                return result
            
            # Hapus semua file di folder temp
            deleted_count = 0
            deleted_size = 0
            
            for filename in os.listdir(temp_folder):
                file_path = os.path.join(temp_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        os.unlink(file_path)
                        deleted_count += 1
                        deleted_size += file_size
                except Exception as e:
                    logger.warning(f"Failed to delete {filename}: {e}")
            
            result.update({
                "deleted_files": deleted_count,
                "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
                "status": "success",
                "message": f"Deleted {deleted_count} files ({result['deleted_size_mb']} MB)"
            })
            
            logger.info(f"🧹 Cleanup complete: {result['message']}")
            self.log_operation("cleanup", "success", result)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to cleanup temp files: {e}"
            result.update({
                "status": "failed",
                "message": error_msg
            })
            
            logger.error(error_msg)
            self.log_operation("cleanup", "error", {"error": str(e)})
            
            return result
    
    def display_folder_structure(self) -> None:
        """Menampilkan struktur folder dengan informasi detail"""
        print(f"\n{'='*60}")
        print("📁 PODCAST AUDIO EXTRACTOR - FOLDER STRUCTURE")
        print(f"{'='*60}")
        
        total_files = 0
        total_size = 0
        
        for key, folder_name in self.folder_manager.structure.items():
            folder_path = os.path.join(self.folder_manager.base_folder, folder_name)
            
            if os.path.exists(folder_path):
                file_count = 0
                folder_size = 0
                
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        file_count += 1
                        folder_size += os.path.getsize(file_path)
                
                total_files += file_count
                total_size += folder_size
                
                size_mb = folder_size / (1024 * 1024)
                
                print(f"\n📂 {folder_name}/")
                print(f"   ├── Path: {folder_path}")
                print(f"   ├── Files: {file_count}")
                print(f"   └── Size: {size_mb:.2f} MB")
            else:
                print(f"\n📂 {folder_name}/")
                print(f"   └── [Folder tidak ditemukan]")
        
        print(f"\n{'='*60}")
        print(f"📊 TOTAL SUMMARY")
        print(f"   ├── Base Folder: {self.folder_manager.base_folder}")
        print(f"   ├── Total Files: {total_files}")
        print(f"   ├── Total Size: {total_size / (1024 * 1024):.2f} MB")
        print(f"   └── Session ID: {self.session_id}")
        print(f"{'='*60}")


def main_menu() -> None:
    """Menu utama program"""
    
    extractor = PodcastAudioExtractor()
    
    while True:
        print(f"\n{'='*60}")
        print("🎙️  PODCAST AUDIO EXTRACTOR")
        print(f"{'='*60}")
        print("1. 🏗️  Setup folder structure")
        print("2. 📹 Extract audio from single file")
        print("3. 📚 Batch extract from folder")
        print("4. 🎵 Extract to multiple formats")
        print("5. ⚡ Enhance audio quality")
        print("6. 📁 View folder structure")
        print("7. 📊 View session logs")
        print("8. 🧹 Cleanup temporary files")
        print("9. 🚪 Exit")
        print(f"{'='*60}")
        
        try:
            choice = input("\n🎯 Select option (1-9): ").strip()
            
            if choice == "1":
                print("\n📁 SETUP FOLDER STRUCTURE")
                new_base = input(f"Base folder (default: {DEFAULT_BASE_FOLDER}): ").strip()
                if new_base:
                    extractor = PodcastAudioExtractor(new_base)
                else:
                    extractor = PodcastAudioExtractor()
            
            elif choice == "2":
                print("\n📹 SINGLE FILE EXTRACTION")
                video_path = input("Enter video file path: ").strip()
                
                if os.path.exists(video_path):
                    move_option = input("Move video to processed folder after extraction? (y/n): ").lower()
                    move_video = move_option == 'y'
                    
                    result = extractor.extract_audio(video_path, move_video)
                    
                    if result["status"] == "success":
                        print(f"\n✅ Success! Audio saved to: {result['audio']}")
                        print(f"   Size: {result['metadata']['file_size_mb']} MB")
                        print(f"   Duration: {result['metadata']['duration']} seconds")
                    else:
                        print(f"\n❌ Failed: {result['message']}")
                else:
                    print(f"\n❌ File not found: {video_path}")
            
            elif choice == "3":
                print("\n📚 BATCH EXTRACTION")
                input_folder = input("Enter folder path containing videos: ").strip()
                
                if os.path.exists(input_folder):
                    file_pattern = input("File pattern (default: *.mp4): ").strip() or "*.mp4"
                    move_option = input("Move videos after processing? (y/n): ").lower()
                    move_videos = move_option == 'y'
                    
                    print("\n🔄 Processing...")
                    results = extractor.batch_extract_audio(input_folder, file_pattern, move_videos)
                    
                    success = sum(1 for r in results if r["status"] == "success")
                    failed = sum(1 for r in results if r["status"] == "failed")
                    
                    print(f"\n📊 Batch complete!")
                    print(f"   ✅ Success: {success}")
                    print(f"   ❌ Failed: {failed}")
                    print(f"   📄 Check report in: {extractor.folder_manager.get_folder_path('metadata')}")
                else:
                    print(f"\n❌ Folder not found: {input_folder}")
            
            elif choice == "4":
                print("\n🎵 MULTI-FORMAT EXTRACTION")
                video_path = input("Enter video file path: ").strip()
                
                if os.path.exists(video_path):
                    print("\n📋 Supported formats: mp3, wav, aac, ogg, flac")
                    formats_input = input("Enter formats (comma separated, default: mp3,wav,aac): ").strip()
                    
                    if formats_input:
                        formats = [f.strip().lower() for f in formats_input.split(",")]
                    else:
                        formats = ["mp3", "wav", "aac"]
                    
                    print("\n🎚️ Quality options: high, medium, low")
                    quality = input("Select quality (default: medium): ").strip().lower() or "medium"
                    
                    if quality not in ["high", "medium", "low"]:
                        print("⚠️ Invalid quality, using medium")
                        quality = "medium"
                    
                    result = extractor.extract_audio_multiformat(video_path, formats, quality)
                    
                    if result["status"] == "success":
                        print(f"\n✅ Success! Converted to {len(result['formats'])} formats:")
                        for fmt, info in result["formats"].items():
                            print(f"   • {fmt.upper()}: {info['size_mb']} MB ({info.get('bitrate', 'N/A')})")
                    else:
                        print(f"\n❌ Failed: {result['message']}")
                else:
                    print(f"\n❌ File not found: {video_path}")
            
            elif choice == "5":
                print("\n⚡ AUDIO ENHANCEMENT")
                video_path = input("Enter video file path: ").strip()
                
                if os.path.exists(video_path):
                    print("\n🔧 Available enhancements:")
                    print("1. Normalize volume")
                    print("2. Noise reduction")
                    print("3. Trim silence")
                    print("4. Compression")
                    
                    enhancements_input = input("Select enhancements (comma separated, default: 1,2): ").strip()
                    
                    if enhancements_input:
                        selections = [s.strip() for s in enhancements_input.split(",")]
                    else:
                        selections = ["1", "2"]
                    
                    enhancements = {
                        "normalize": "1" in selections,
                        "noise_reduction": "2" in selections,
                        "trim_silence": "3" in selections,
                        "compression": "4" in selections,
                        "equalization": False
                    }
                    
                    result = extractor.enhance_audio(video_path, enhancements)
                    
                    if result["status"] == "success":
                        print(f"\n✅ Success! Enhanced audio saved to: {result['enhanced_audio']}")
                        print(f"   Enhancements applied: {', '.join(result['enhancements_applied'])}")
                        print(f"   Size: {result['metadata']['file_size_mb']} MB")
                    else:
                        print(f"\n❌ Failed: {result['message']}")
                else:
                    print(f"\n❌ File not found: {video_path}")
            
            elif choice == "6":
                extractor.display_folder_structure()
            
            elif choice == "7":
                log_folder = extractor.folder_manager.get_folder_path("logs")
                print(f"\n📋 SESSION LOGS")
                print(f"Location: {log_folder}")
                
                log_files = [f for f in os.listdir(log_folder) if f.endswith('.json')]
                
                if log_files:
                    print(f"\n📂 Available log files:")
                    for log_file in sorted(log_files, reverse=True)[:5]:
                        log_path = os.path.join(log_folder, log_file)
                        size = os.path.getsize(log_path) / 1024
                        print(f"  • {log_file} ({size:.1f} KB)")
                    
                    view_option = input("\nView latest log? (y/n): ").lower()
                    if view_option == 'y' and log_files:
                        latest_log = os.path.join(log_folder, sorted(log_files, reverse=True)[0])
                        try:
                            with open(latest_log, 'r', encoding='utf-8') as f:
                                log_data = json.load(f)
                            
                            print(f"\n📊 LOG SUMMARY - {os.path.basename(latest_log)}")
                            print(f"Session ID: {log_data.get('session_id')}")
                            print(f"Start Time: {log_data.get('start_time')}")
                            print(f"End Time: {log_data.get('end_time', 'N/A')}")
                            print(f"Total Operations: {log_data.get('total_operations', 0)}")
                            print(f"Total Errors: {log_data.get('total_errors', 0)}")
                            print(f"Total Warnings: {log_data.get('total_warnings', 0)}")
                            
                        except Exception as e:
                            print(f"Error reading log: {e}")
                else:
                    print("No log files found.")
            
            elif choice == "8":
                confirm = input("🧹 Cleanup all temporary files? This cannot be undone. (y/n): ").lower()
                if confirm == 'y':
                    result = extractor.cleanup_temp_files()
                    if result["status"] == "success":
                        print(f"✅ {result['message']}")
                    else:
                        print(f"❌ {result['message']}")
                else:
                    print("Cleanup cancelled.")
            
            elif choice == "9":
                # Save logs sebelum exit
                log_file = extractor.save_logs()
                print(f"\n👋 Goodbye!")
                print(f"📁 Session logs saved to: {log_file}")
                break
            
            else:
                print("\n❌ Invalid option!")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Program interrupted by user.")
            extractor.save_logs()
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def test_functionality() -> None:
    """Test semua fungsi dengan sample"""
    print(f"\n{'='*60}")
    print("🧪 TESTING PODCAST AUDIO EXTRACTOR")
    print(f"{'='*60}")
    
    # Setup extractor dengan folder test
    test_folder = os.path.join(PROJECT_ROOT, "test_output")
    extractor = PodcastAudioExtractor(test_folder)
    
    # Cari sample video untuk testing
    sample_found = None
    for pattern in SUPPORTED_VIDEO_FORMATS[:3]:  # Coba 3 format pertama
        sample_files = glob.glob(pattern)
        if sample_files:
            sample_found = sample_files[0]
            break
    
    if sample_found:
        print(f"📹 Found sample video: {sample_found}")
        
        try:
            # Test 1: Single extraction
            print("\n1. Testing single file extraction...")
            result = extractor.extract_audio(sample_found, move_processed=False)
            print(f"   Result: {result['status']}")
            
            if result["status"] == "success":
                # Test 2: Multi-format
                print("\n2. Testing multi-format extraction...")
                result2 = extractor.extract_audio_multiformat(sample_found, ["mp3", "wav"])
                print(f"   Result: {result2['status']}")
                
                # Test 3: Enhancement
                print("\n3. Testing audio enhancement...")
                result3 = extractor.enhance_audio(sample_found)
                print(f"   Result: {result3['status']}")
            
            # Display structure
            print("\n4. Displaying folder structure...")
            extractor.display_folder_structure()
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
    
    else:
        print("❌ No sample video found for testing.")
        print("   Please add a video file (mp4, avi, mov) to the script directory.")
    
    # Save logs
    log_file = extractor.save_logs()
    print(f"\n✅ Testing complete!")
    print(f"   Check results in: {test_folder}")
    print(f"   Logs saved to: {log_file}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("🎙️  PODCAST AUDIO EXTRACTOR")
    print("   Clear Folder Structure for Easy Management")
    print(f"{'='*60}")
    
    # Pilihan mode
    print("\nSelect mode:")
    print("1. 🚀 Main menu (interactive)")
    print("2. 🧪 Run tests")
    print("3. 📁 Setup folder structure only")
    
    try:
        mode = input("\nSelect mode (1-3): ").strip()
        
        if mode == "1":
            main_menu()
        elif mode == "2":
            test_functionality()
        elif mode == "3":
            folder_manager = FolderManager()
            folder_manager.display_folder_structure()
        else:
            print("❌ Invalid mode selection!")
    
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")