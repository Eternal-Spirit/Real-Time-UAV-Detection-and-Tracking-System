@echo off
echo ============================================
echo  Phase 1 - Install dependencies
echo ============================================

echo.
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing PyTorch with CUDA 12.1 (RTX 4060)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Installing Ultralytics + OpenCV...
pip install ultralytics opencv-python

echo.
echo ============================================
echo  Verifying GPU is visible to PyTorch...
echo ============================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

echo.
echo ============================================
echo  Done. Run train.bat to start training.
echo ============================================
pause
