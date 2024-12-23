from main import run as model_train
import numpy as np
import os

Tc_list = np.linspace(0.4, 1.6, 13)

L = 30

for Tc in Tc_list:
    T = Tc * 2.269
    model_path = f"model_confusion_demo/{L}/model_{T:.2f}.pth"
    if not os.path.exists(f"model_confusion_demo/{L}"):
        os.makedirs(f"model_confusion_demo/{L}")
    model_train(T, L, model_path)