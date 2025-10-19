import os
import tkinter as tk
from model import HumanSegmentator
from gui import SimpleSegmentationApp

print("=" * 70)
print("🚀 AI HUMAN SEGMENTATION - HACKATHON READY!")
print("=" * 70)
print("🧠 Neural Network: U-Net with ResNet18")
print("🎯 Features: Real-time segmentation & background replacement")
print("💻 Device: CPU/GPU Auto-detection")
print("=" * 70)

def main():
    # Создаем папки
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("examples", exist_ok=True)
    
    # Загружаем AI модель
    print("🔄 Loading AI Model...")
    segmentator = HumanSegmentator()
    
    # Запускаем GUI
    print("🖼️ Starting GUI Application...")
    root = tk.Tk()
    app = SimpleSegmentationApp(root, segmentator)
    root.mainloop()

if __name__ == "__main__":
    main()