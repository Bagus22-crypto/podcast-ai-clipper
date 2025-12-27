"""
PODCAST PROCESSING PIPELINE
Script terintegrasi untuk ekstraksi audio dan transkripsi podcast
"""

import os
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import modul dari SCRIPT1 (audio extraction)
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
import glob

# Import modul dari SCRIPT2 (transcription) - perlu dibuat
# Asumsi: modul podcast_transcriber.py sudah tersedia
try:
    from podcast_transcriber import PodcastTranscriber, TranscriptionConfig
    TRANSCRIBER_AVAILABLE = True
except ImportError:
    logger.warning("Modul transkripsi tidak ditemukan. Fitur transkripsi tidak tersedia.")
    TRANSCRIBER_AVAILABLE = False


class PodcastAudioExtractor:
    """Kelas untuk mengekstrak audio dari video podcast"""
    
    def __init__(self, output_dir: str = "audio_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_audio(self, video_path: str, output_format: str = "mp3", 
                      quality: str = "medium") -> str:
        """Ekstrak audio dari video dengan berbagai format"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File tidak ditemukan: {video_path}")
        
        # Mapping format dan bitrate
        bitrate_map = {
            "high": {"mp3": "320k", "aac": "256k", "ogg": "192k"},
            "medium": {"mp3": "192k", "aac": "192k", "ogg": "128k"},
            "low": {"mp3": "128k", "aac": "128k", "ogg": "96k"}
        }
        
        format_settings = {
            "mp3": {"codec": "mp3", "ext": "mp3"},
            "wav": {"codec": "pcm_s16le", "ext": "wav"},
            "aac": {"codec": "aac", "ext": "m4a"},
            "ogg": {"codec": "libvorbis", "ext": "ogg"},
            "flac": {"codec": "flac", "ext": "flac"},
        }
        
        # Validasi format
        output_format = output_format.lower()
        if output_format not in format_settings:
            raise ValueError(f"Format tidak didukung: {output_format}")
        
        # Generate output path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            self.output_dir, 
            f"{base_name}_audio.{format_settings[output_format]['ext']}"
        )
        
        # Ekstrak audio
        logger.info(f"Mengekstrak audio dari: {video_path}")
        with VideoFileClip(video_path) as video:
            audio = video.audio
            
            # Set bitrate jika tersedia
            bitrate = None
            if output_format in bitrate_map.get(quality, {}):
                bitrate = bitrate_map[quality][output_format]
            
            audio.write_audiofile(
                output_path,
                codec=format_settings[output_format]['codec'],
                bitrate=bitrate,
                verbose=False,
                logger=None
            )
        
        logger.info(f"Audio berhasil diekstrak: {output_path}")
        return output_path
    
    def batch_extract(self, input_folder: str, 
                      output_format: str = "mp3") -> List[Dict[str, Any]]:
        """Ekstrak audio dari semua video dalam folder"""
        results = []
        
        # Cari semua file video
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not video_files:
            logger.warning(f"Tidak ada file video di: {input_folder}")
            return results
        
        logger.info(f"Memproses {len(video_files)} file video...")
        
        for video_file in video_files:
            try:
                audio_path = self.extract_audio(video_file, output_format)
                results.append({
                    "video": video_file,
                    "audio": audio_path,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "video": video_file,
                    "error": str(e),
                    "status": "failed"
                })
                logger.error(f"Gagal memproses {video_file}: {e}")
        
        return results
    
    def enhance_audio(self, audio_path: str, 
                      noise_reduction: bool = True,
                      normalize: bool = True) -> str:
        """Enhance audio dengan noise reduction dan normalisasi"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Apply enhancements
        if normalize:
            audio = pydub_normalize(audio)
        
        if noise_reduction:
            # Simple noise reduction
            audio = audio.low_pass_filter(3000).high_pass_filter(300)
        
        # Save enhanced audio
        enhanced_path = audio_path.replace(".mp3", "_enhanced.mp3")
        audio.export(enhanced_path, format="mp3", bitrate="192k")
        
        return enhanced_path


class PodcastTranscriptionPipeline:
    """Pipeline lengkap untuk ekstraksi dan transkripsi podcast"""
    
    def __init__(self, 
                 audio_output_dir: str = "audio_output",
                 transcript_output_dir: str = "transcripts",
                 model_size: str = "base",
                 language: str = "id"):
        
        self.audio_extractor = PodcastAudioExtractor(audio_output_dir)
        self.transcript_output_dir = transcript_output_dir
        
        # Setup transcriber jika tersedia
        if TRANSCRIBER_AVAILABLE:
            config = TranscriptionConfig(
                model_size=model_size,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            self.transcriber = PodcastTranscriber(config)
        else:
            self.transcriber = None
        
        # Buat folder output
        os.makedirs(transcript_output_dir, exist_ok=True)
    
    def process_single_video(self, video_path: str, 
                             extract_only: bool = False,
                             enhance_audio: bool = True,
                             transcribe: bool = True) -> Dict[str, Any]:
        """Process single video file end-to-end"""
        results = {
            "video": video_path,
            "audio": None,
            "transcript": None,
            "status": "processing"
        }
        
        try:
            # Step 1: Extract audio
            logger.info(f"Step 1: Mengekstrak audio dari {video_path}")
            audio_path = self.audio_extractor.extract_audio(video_path)
            results["audio"] = audio_path
            
            # Step 2: Enhance audio (optional)
            if enhance_audio:
                logger.info("Step 2: Enhancing audio")
                audio_path = self.audio_extractor.enhance_audio(audio_path)
                results["enhanced_audio"] = audio_path
            
            # Step 3: Transcribe (optional)
            if transcribe and self.transcriber and not extract_only:
                logger.info("Step 3: Mentranskripsi audio")
                transcript_result = self.transcriber.transcribe(audio_path)
                
                # Save transcript
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_base = os.path.join(self.transcript_output_dir, base_name)
                saved_files = self.transcriber.save_transcript(transcript_result, output_base)
                
                results["transcript"] = saved_files
                results["transcript_data"] = transcript_result
            
            results["status"] = "completed"
            logger.info(f"Proses selesai untuk {video_path}")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Gagal memproses {video_path}: {e}")
        
        return results
    
    def process_batch(self, input_folder: str, 
                      extract_only: bool = False,
                      enhance_audio: bool = True) -> List[Dict[str, Any]]:
        """Process semua video dalam folder"""
        all_results = []
        
        # Ekstrak audio untuk semua file
        audio_results = self.audio_extractor.batch_extract(input_folder)
        
        for audio_result in audio_results:
            if audio_result["status"] == "success":
                video_file = audio_result["video"]
                audio_file = audio_result["audio"]
                
                # Enhance audio jika diperlukan
                if enhance_audio:
                    try:
                        audio_file = self.audio_extractor.enhance_audio(audio_file)
                    except Exception as e:
                        logger.warning(f"Gagal enhance audio {audio_file}: {e}")
                
                # Transcribe jika diperlukan
                transcript_info = None
                if not extract_only and self.transcriber:
                    try:
                        logger.info(f"Transcribing: {audio_file}")
                        transcript_result = self.transcriber.transcribe(audio_file)
                        
                        # Save transcript
                        base_name = os.path.splitext(os.path.basename(video_file))[0]
                        output_base = os.path.join(self.transcript_output_dir, base_name)
                        saved_files = self.transcriber.save_transcript(transcript_result, output_base)
                        
                        transcript_info = saved_files
                    except Exception as e:
                        logger.error(f"Gagal transcribe {audio_file}: {e}")
                        transcript_info = {"error": str(e)}
                
                all_results.append({
                    "video": video_file,
                    "audio": audio_file,
                    "transcript": transcript_info,
                    "status": "completed"
                })
            else:
                all_results.append({
                    "video": audio_result["video"],
                    "error": audio_result.get("error", "Unknown error"),
                    "status": "failed"
                })
        
        return all_results


def main():
    """Main function dengan command line interface"""
    parser = argparse.ArgumentParser(description="Podcast Processing Pipeline")
    
    # Mode operasi
    parser.add_argument("mode", choices=["extract", "transcribe", "pipeline", "batch"],
                       help="Mode operasi: extract (audio saja), transcribe (audio ke teks), pipeline (end-to-end), batch (batch processing)")
    
    # Input/output
    parser.add_argument("--input", required=True, help="Input file atau folder")
    parser.add_argument("--audio-output", default="audio_output", help="Folder output audio")
    parser.add_argument("--transcript-output", default="transcripts", help="Folder output transkripsi")
    
    # Audio options
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "aac", "ogg", "flac"],
                       help="Format output audio")
    parser.add_argument("--enhance", action="store_true", help="Enable audio enhancement")
    
    # Transcription options
    parser.add_argument("--model", default="base", help="Model size untuk transkripsi")
    parser.add_argument("--language", default="id", help="Bahasa untuk transkripsi")
    parser.add_argument("--extract-only", action="store_true", help="Hanya ekstrak audio, tidak transcribe")
    
    args = parser.parse_args()
    
    # Validasi input
    if args.mode in ["extract", "transcribe", "pipeline"]:
        if not os.path.exists(args.input):
            logger.error(f"File tidak ditemukan: {args.input}")
            return
    
    # Inisialisasi pipeline
    pipeline = PodcastTranscriptionPipeline(
        audio_output_dir=args.audio_output,
        transcript_output_dir=args.transcript_output,
        model_size=args.model,
        language=args.language
    )
    
    # Eksekusi berdasarkan mode
    if args.mode == "extract":
        # Hanya ekstrak audio
        audio_path = pipeline.audio_extractor.extract_audio(args.input, args.format)
        print(f"\n✅ Audio berhasil diekstrak: {audio_path}")
        
        if args.enhance:
            enhanced_path = pipeline.audio_extractor.enhance_audio(audio_path)
            print(f"✅ Audio enhanced: {enhanced_path}")
    
    elif args.mode == "transcribe":
        # Hanya transkripsi (dari file audio)
        if not pipeline.transcriber:
            print("❌ Modul transkripsi tidak tersedia!")
            return
        
        transcript_result = pipeline.transcriber.transcribe(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_base = os.path.join(args.transcript_output, base_name)
        saved_files = pipeline.transcriber.save_transcript(transcript_result, output_base)
        
        print(f"\n✅ Transkripsi selesai!")
        for fmt, path in saved_files.items():
            print(f"   {fmt.upper()}: {path}")
    
    elif args.mode == "pipeline":
        # End-to-end pipeline
        result = pipeline.process_single_video(
            args.input,
            extract_only=args.extract_only,
            enhance_audio=args.enhance,
            transcribe=not args.extract_only
        )
        
        if result["status"] == "completed":
            print(f"\n✅ Pipeline selesai!")
            print(f"   Audio: {result['audio']}")
            if "enhanced_audio" in result:
                print(f"   Enhanced Audio: {result['enhanced_audio']}")
            if result["transcript"]:
                print(f"   Transcript: {result['transcript']}")
        else:
            print(f"\n❌ Pipeline gagal: {result.get('error', 'Unknown error')}")
    
    elif args.mode == "batch":
        # Batch processing
        if not os.path.isdir(args.input):
            print(f"❌ {args.input} bukan folder!")
            return
        
        results = pipeline.process_batch(
            args.input,
            extract_only=args.extract_only,
            enhance_audio=args.enhance
        )
        
        # Summary
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"   Total files: {len(results)}")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        
        # Save summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(args.transcript_output, f"batch_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("BATCH PROCESSING REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Input Folder: {args.input}\n")
            f.write(f"Total Files: {len(results)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\nFile {i}: {result['video']}\n")
                f.write(f"  Status: {result['status']}\n")
                if result["status"] == "completed":
                    f.write(f"  Audio: {result.get('audio', 'N/A')}\n")
                    if result.get('transcript'):
                        f.write(f"  Transcript: {result['transcript']}\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown')}\n")
        
        print(f"📄 Detail report disimpan di: {report_file}")


if __name__ == "__main__":
    # Jika tidak ada argumen command line, jalankan interactive mode
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("PODCAST PROCESSING PIPELINE")
        print("="*60)
        print("Mode:")
        print("  1. Ekstrak audio dari video")
        print("  2. Transkripsi file audio")
        print("  3. Pipeline lengkap (video → audio → transkripsi)")
        print("  4. Batch processing (folder)")
        print("  5. Keluar")
        
        choice = input("\nPilih mode (1-5): ").strip()
        
        if choice == "1":
            video_path = input("Path file video: ").strip()
            if os.path.exists(video_path):
                sys.argv = ["script.py", "extract", "--input", video_path]
                main()
            else:
                print("❌ File tidak ditemukan!")
        
        elif choice == "2":
            audio_path = input("Path file audio: ").strip()
            if os.path.exists(audio_path):
                sys.argv = ["script.py", "transcribe", "--input", audio_path]
                main()
            else:
                print("❌ File tidak ditemukan!")
        
        elif choice == "3":
            video_path = input("Path file video: ").strip()
            if os.path.exists(video_path):
                sys.argv = ["script.py", "pipeline", "--input", video_path, "--enhance"]
                main()
            else:
                print("❌ File tidak ditemukan!")
        
        elif choice == "4":
            folder_path = input("Path folder video: ").strip()
            if os.path.isdir(folder_path):
                sys.argv = ["script.py", "batch", "--input", folder_path, "--enhance"]
                main()
            else:
                print("❌ Folder tidak ditemukan!")
        
        elif choice == "5":
            print("👋 Program selesai")
    
    else:
        main()