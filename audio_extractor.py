import os
from moviepy.editor import VideoFileClip
import logging

# Setup logging untuk tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio(video_path, output_audio_path=None):
    """
    Mengekstrak audio dari video podcast.
    
    Args:
        video_path (str): Path ke file video
        output_audio_path (str): Path output audio (opsional)
    
    Returns:
        str: Path ke file audio yang dihasilkan
        
    Raises:
        FileNotFoundError: Jika video tidak ditemukan
        Exception: Jika ekstraksi gagal
    """
    try:
        # Validasi input file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")
        
        # Generate output path jika tidak diberikan
        if output_audio_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_audio_path = f"{base_name}_audio.mp3"
        
        logger.info(f"Memulai ekstraksi audio dari: {video_path}")
        
        # Load video file
        video = VideoFileClip(video_path)
        
        # Ekstrak audio
        audio = video.audio
        
        # Simpan sebagai MP3
        logger.info(f"Menyimpan audio ke: {output_audio_path}")
        audio.write_audiofile(
            output_audio_path,
            codec='mp3',
            bitrate='192k',  # Bitrate standar untuk kualitas baik
            verbose=False,
            logger=None
        )
        
        # Tutup video untuk free memory
        video.close()
        
        # Verifikasi file output
        if os.path.exists(output_audio_path):
            file_size = os.path.getsize(output_audio_path) / (1024 * 1024)  # MB
            logger.info(f"Ekstraksi berhasil! Ukuran file: {file_size:.2f} MB")
            return output_audio_path
        else:
            raise Exception("File audio tidak tercreate setelah ekstraksi")
            
    except Exception as e:
        logger.error(f"Gagal mengekstrak audio: {str(e)}")
        raise

def test_audio_extraction():
    """Fungsi testing untuk ekstraksi audio"""
    print("\n=== TESTING AUDIO EXTRACTION ===")
    
    # Test case 1: File valid
    sample_video = "data/samples/sample_podcast.mp4"
    
    if os.path.exists(sample_video):
        print(f"✓ Sample video ditemukan: {sample_video}")
        
        try:
            audio_path = extract_audio(sample_video)
            print(f"✓ Audio berhasil diekstrak: {audio_path}")
            return audio_path
        except Exception as e:
            print(f"✗ Gagal: {e}")
            return None
    else:
        print("✗ Sample video tidak ditemukan!")
        print("  Pastikan file ada di: data/samples/sample_podcast.mp4")
        return None

if __name__ == "__main__":
    # Jalankan test
    test_audio_extraction()