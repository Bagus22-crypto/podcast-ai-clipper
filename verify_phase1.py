import os
import json

def verify_phase1_output():
    """Verifikasi semua output Fase 1"""
    print("=== VERIFIKASI FASE 1 ===")
    
    # Cek file-file penting
    required_files = [
        'audio_extractor.py',
        'transcriber.py', 
        'pipeline_phase1.py'
    ]
    
    print("\n1. Cek Script:")
    all_ok = True
    for f in required_files:
        if os.path.exists(f):
            print(f"   ✓ {f}")
        else:
            print(f"   ✗ {f} - TIDAK ADA!")
            all_ok = False
    
    # Cek hasil output terbaru
    print("\n2. Cek Output Files:")
    output_dir = "data/output"
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            print(f"   Output directory: {output_dir}")
            for f in files[:5]:  # Tampilkan 5 file pertama
                print(f"   - {f}")
            
            # Coba baca salah satu JSON
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                latest_json = max([os.path.join(output_dir, f) for f in json_files], 
                                key=os.path.getmtime)
                
                try:
                    with open(latest_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    print(f"\n3. Cek Isi Transcript:")
                    print(f"   File: {os.path.basename(latest_json)}")
                    print(f"   Language: {data.get('language', 'unknown')}")
                    print(f"   Segments: {len(data.get('segments', []))}")
                    print(f"   Text length: {len(data.get('text', ''))} characters")
                    
                    # Tampilkan contoh
                    if 'segments' in data and len(data['segments']) > 0:
                        sample = data['segments'][0]
                        print(f"\n   Sample segment pertama:")
                        print(f"   Time: {sample.get('start', 0):.1f}s - {sample.get('end', 0):.1f}s")
                        print(f"   Text: {sample.get('text', '')[:80]}...")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ✗ Gagal baca JSON: {e}")
                    return False
        else:
            print("   ✗ Output directory kosong!")
            print("   Jalankan: python pipeline_phase1.py --input data/samples/sample_podcast.mp4")
            return False
    else:
        print("   ✗ Output directory belum dibuat!")
        return False

if __name__ == "__main__":
    success = verify_phase1_output()
    
    if success:
        print("\n" + "="*50)
        print("✅ FASE 1 SUDAH BERHASIL!")
        print("="*50)
        print("\n📁 Struktur folder sekarang:")
        
        # Print tree sederhana
        for root, dirs, files in os.walk("."):
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            
            subindent = " " * 2 * (level + 1)
            for file in files[:3]:  # Tampilkan 3 file pertama per folder
                if file.endswith(('.py', '.json', '.mp3', '.mp4', '.txt')):
                    print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... dan {len(files)-3} file lainnya")
                
    else:
        print("\n❌ MASIH ADA YANG HARUS DIPERBAIKI")