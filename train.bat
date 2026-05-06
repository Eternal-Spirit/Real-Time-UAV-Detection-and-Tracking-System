@echo off
echo ============================================
echo  Phase 1 - YOLOv8s + CLAHE Training
echo  Expected time: ~25 min on RTX 4060
echo ============================================
echo.

python training.py

echo.
echo ============================================
echo  Training finished. Check results above.
echo  Weights saved to: models\best.pt
echo ============================================
pause
