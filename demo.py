import cv2
import numpy as np
from model import HumanSegmentator, create_demo_image

def quick_demo():
    """Быстрая демонстрация без GUI"""
    print("🚀 QUICK AI DEMONSTRATION")
    
    # Создаем AI
    ai = HumanSegmentator()
    
    # Создаем демо изображение
    print("🎨 Creating demo image...")
    demo_img = create_demo_image()
    cv2.imwrite("demo_input.jpg", demo_img)
    
    # Обрабатываем разными фонами
    backgrounds = ["blue", "green", "gradient"]
    
    for bg in backgrounds:
        print(f"🔄 Processing with {bg} background...")
        mask, result = ai.process_image(demo_img, bg)
        cv2.imwrite(f"demo_result_{bg}.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print("✅ DEMO COMPLETE! Check generated images:")
    print("   - demo_input.jpg")
    print("   - demo_result_blue.jpg") 
    print("   - demo_result_green.jpg")
    print("   - demo_result_gradient.jpg")

if __name__ == "__main__":
    quick_demo()