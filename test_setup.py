import sys
print(f"Python version: {sys.version}")
print(f"Virtual env: {sys.prefix}")

# Test import library dasar
try:
    import numpy
    print("✅ numpy bisa diimport")
except ImportError:
    print("❌ numpy belum terinstall")

print("\nSetup selesai! Lanjut ke Fase 1.")