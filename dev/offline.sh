
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:/root/miniconda3/lib:${LD_LIBRARY_PATH}
python text_to_video.py \
  --model "dg845/LTX-2.3-Diffusers" \
  --model-class-name LTX23Pipeline \
  --prompt "A cinematic close-up of ocean waves at golden hour." \
  --height 256 \
  --width 256 \
  --num-frames 17 \
  --num-inference-steps 20 \
  --guidance-scale 4.0 \
  --frame-rate 24 \
  --output ltx2_smoke.mp4 