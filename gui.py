import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class SimpleSegmentationApp:
    """Минимальный но эффективный GUI"""
    
    def __init__(self, root, segmentator):
        self.root = root
        self.segmentator = segmentator
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка интерфейса за 5 минут"""
        self.root.title("🧠 AI Human Segmentation - HACKATHON")
        self.root.geometry("1000x600")
        
        # Заголовок
        title = tk.Label(self.root, text="AI HUMAN SEGMENTATION", 
                        font=("Arial", 16, "bold"), fg="blue")
        title.pack(pady=10)
        
        # Кнопки управления
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.btn_load = tk.Button(button_frame, text="📁 LOAD IMAGE", 
                                 command=self.load_image, width=15, height=2)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_blue = tk.Button(button_frame, text="🔵 BLUE BG", 
                                 command=lambda: self.process_image("blue"), 
                                 width=10, height=2, bg="lightblue")
        self.btn_blue.pack(side=tk.LEFT, padx=5)
        
        self.btn_green = tk.Button(button_frame, text="🟢 GREEN BG", 
                                  command=lambda: self.process_image("green"), 
                                  width=10, height=2, bg="lightgreen")
        self.btn_green.pack(side=tk.LEFT, padx=5)
        
        self.btn_gradient = tk.Button(button_frame, text="🌈 GRADIENT BG", 
                                     command=lambda: self.process_image("gradient"), 
                                     width=12, height=2, bg="lightyellow")
        self.btn_gradient.pack(side=tk.LEFT, padx=5)
        
        # Области изображений
        image_frame = tk.Frame(self.root)
        image_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Оригинал
        self.original_frame = tk.LabelFrame(image_frame, text="ORIGINAL IMAGE")
        self.original_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        self.original_label = tk.Label(self.original_frame, text="Load an image to start", 
                                      bg="lightgray", relief="sunken")
        self.original_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Результат
        self.result_frame = tk.LabelFrame(image_frame, text="AI RESULT")
        self.result_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)
        
        self.result_label = tk.Label(self.result_frame, text="Result will appear here", 
                                    bg="lightgray", relief="sunken")
        self.result_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Статус
        self.status = tk.StringVar(value="🟢 Ready to load image")
        status_bar = tk.Label(self.root, textvariable=self.status, relief="sunken")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Демо кнопка
        btn_demo = tk.Button(self.root, text="🎨 LOAD DEMO", 
                            command=self.load_demo, bg="violet", fg="white")
        btn_demo.pack(pady=5)
    
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.current_image = file_path
                image = Image.open(file_path)
                self.display_image(image, self.original_label)
                self.status.set(f"📁 Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load image: {e}")
    
    def process_image(self, background_type):
        """Обработка изображения"""
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status.set("🧠 AI Processing...")
            self.root.update()
            
            # Обработка AI
            mask, result = self.segmentator.process_image(self.current_image, background_type)
            
            # Конвертация для отображения
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            result_pil = Image.fromarray(result_bgr)
            
            # Показ результата
            self.display_image(result_pil, self.result_label)
            self.status.set("✅ Done! Result saved in outputs/")
            
            # Автосохранение
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/result_{background_type}.jpg"
            cv2.imwrite(output_path, result_bgr)
            
        except Exception as e:
            messagebox.showerror("Error", f"AI processing failed: {e}")
            self.status.set("❌ Processing failed")
    
    def load_demo(self):
        """Загрузка демо примера"""
        try:
            from model import create_demo_image
            import cv2
            
            # Создаем демо изображение
            demo_img = create_demo_image()
            cv2.imwrite("demo_human.jpg", demo_img)
            
            self.current_image = "demo_human.jpg"
            image = Image.open("demo_human.jpg")
            self.display_image(image, self.original_label)
            self.status.set("🎨 Demo loaded - Click buttons to process!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Demo failed: {e}")
    
    def display_image(self, image, label):
        """Отображение изображения"""
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo