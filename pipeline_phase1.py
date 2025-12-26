import os
import sys
import argparse
from datetime import datetime
from audio_extractor import extract_audio
from transcriber import PodcastTranscriber

def run_phase1_pipeline(video_path, model_size="base", output_dir=None):
    """
    Pipeline lengkap Fase 1: Video -> Audio -> Transcript
    
    Args:
        video_path (str): Path ke video podcast
        model_size (str): Ukuran model Whisper
        output_dir (str): Directory untuk output
    
    Returns:
        dict: Path ke semua file yang dihasilkan
    """
    print("\n" + "="*60)
    print("FASE 1: PENGOLAHAN DATA AWAL")
    print("="*60)
    
    # Setup output directory
    if output_dir is None:
        output_dir = "data/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp untuk unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base = f"{output_dir}/{base_name}_{timestamp}"
    
    results = {
        'video_input': video_path,
        'timestamp': timestamp,
        'output_dir': output_dir
    }
    
    # STEP 1: Ekstraksi Audio
    print("\n[1/3] 🎵 Ekstraksi Audio dari Video...")
    try:
        audio_output = f"{output_base}_audio.mp3"
        audio_path = extract_audio(video_path, audio_output)
        results['audio'] = audio_path
        print(f"   ✅ Audio disimpan: {audio_path}")
    except Exception as e:
        print(f"   ❌ Gagal ekstraksi audio: {e}")
        return None
    
    # STEP 2: Transkripsi dengan Whisper
    print("\n[2/3] 🗣️  Transkripsi Audio ke Text...")
    print(f"   Menggunakan model: {model_size}")
    print("   Ini mungkin butuh beberapa menit...")
    
    try:
        transcriber = PodcastTranscriber(model_size=model_size)
        transcript_result = transcriber.transcribe(audio_path)
        results['transcript_raw'] = transcript_result
        
        # STEP 3: Simpan Hasil
        print("\n[3/3] 💾 Menyimpan Hasil...")
        json_path, srt_path = transcriber.save_transcript(
            transcript_result, 
            f"{output_base}_transcript"
        )
        
        results['transcript_json'] = json_path
        results['transcript_srt'] = srt_path
        results['transcript_txt'] = f"{output_base}_transcript.txt"
        
        # Tampilkan summary
        print("\n" + "="*60)
        print("SUMMARY HASIL FASE 1")
        print("="*60)
        print(f"Input Video: {video_path}")
        print(f"Output Directory: {output_dir}")
        print(f"\nFile yang dihasilkan:")
        print(f"  1. Audio: {os.path.basename(audio_path)}")
        print(f"  2. Transcript JSON: {os.path.basename(json_path)}")
        print(f"  3. Transcript SRT: {os.path.basename(srt_path)}")
        print(f"  4. Transcript TXT: {os.path.basename(results['transcript_txt'])}")
        
        durasi = len(transcript_result['text'].split()) / 130  # Approx words per minute
        print(f"\n📊 Statistik:")
        print(f"  - Detected Language: {transcript_result.get('language', 'unknown')}")
        print(f"  - Total Segments: {len(transcript_result['segments'])}")
        print(f"  - Estimated Duration: {durasi:.1f} menit")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Gagal transkripsi: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Pipeline Fase 1: Video ke Transcript')
    parser.add_argument('--input', '-i', required=True, help='Path ke file video podcast')
    parser.add_argument('--model', '-m', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Model Whisper yang digunakan (default: base)')
    parser.add_argument('--output', '-o', default='data/output',
                       help='Output directory (default: data/output)')
    
    args = parser.parse_args()
    
    # Validasi input
    if not os.path.exists(args.input):
        print(f"Error: File tidak ditemukan: {args.input}")
        sys.exit(1)
    
    # Jalankan pipeline
    results = run_phase1_pipeline(
        video_path=args.input,
        model_size=args.model,
        output_dir=args.output
    )
    
    if results:
        print("\n✅ FASE 1 BERHASIL DICOMPLETE!")
        print(f"Hasil disimpan di: {args.output}")
    else:
        print("\n❌ FASE 1 GAGAL")
        sys.exit(1)

if __name__ == "__main__":
    main()