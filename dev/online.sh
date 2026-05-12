mkdir /efs/kwinsheng/vllm-omni/dev/output
export VLLM_OMNI_STORAGE_PATH=/efs/kwinsheng/vllm-omni/dev/output
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "dg845/LTX-2.3-Diffusers" --omni \
  --host 0.0.0.0 \
  --port 8098 \
  --model-class-name LTX23Pipeline \
  --data-parallel-size 8


curl -sS -X POST http://localhost:8098/v1/videos/sync \
    -F "prompt=A cinematic close-up of ocean waves at golden hour" \
    -F "width=1280" \
    -F "height=736" \
    -F "num_frames=241" \
    -F "fps=24" \
    -F "num_inference_steps=20" \
    -F "guidance_scale=4.0" \
    -F "seed=41" \
    -o "ltx2_dp8_1.mp4"