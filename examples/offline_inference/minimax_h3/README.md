CUDA_VISIBLE_DEVICES=0,1  python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ --task t2va \
  --aspect-ratio 16:9 \
  --prompts "A quiet cinematic night scene with matching ambient sound."

CUDA_VISIBLE_DEVICES=4,5,6,7 python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ --task t2va \
  --usp 4 \
  --height 768 --width 1344 --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  --num-warmup 0 \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..."

python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 8 \
  --height 768 \
  --width 1344 \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..."

# FL2VA-only TeaCache acceleration. The calibrated default threshold is 0.17.
python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ \
  --task t2va \
  --aspect-ratio 16:9 \
  --cache-backend tea_cache \
  --prompts "A quiet cinematic night scene with matching ambient sound."


  python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 8 \
  --height 768 \
  --width 1344 \
  --quality high \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..."

python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 8 \
  --height 768 \
  --width 1344 \
  --quality high \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --cache-backend cache_dit \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..." | tee output.log

python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 8 \
  --steps 10 \
  --height 768 \
  --width 1344 \
  --quality high \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --cache-backend cache_dit \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./h3_prof","torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_record_shapes":false}' \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..." | tee output2.log


python examples/offline_inference/minimax_h3/end2end.py \
  --model /data/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 4 \
  --steps 10 \
  --height 768 \
  --width 1344 \
  --quality high \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/data/h3_prof","torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_record_shapes":false}' \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..." | tee output3.log


python examples/offline_inference/minimax_h3/end2end.py \
  --model /data/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 4 \
  --steps 50 \
  --height 768 \
  --width 1344 \
  --aspect-ratio 16:9 \
  --duration 10 --seed 1101 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --prompts "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a bright medium-wide shot frames three adult female cosplayers dancing in formation at the center of a spacious modern city plaza in warm late-afternoon sunlight. The center dancer wears a blue-and-white magical-girl-inspired costume with a short layered skirt and matching hair ribbons; the dancer on the left wears a red-and-black fantasy idol costume with twin ponytails; the dancer on the right wears a pastel pink-and-lavender costume with a star-shaped hair accessory. Full-body framing keeps all three dancers clearly visible as they begin a synchronized upbeat otaku dance routine: two quick side steps, alternating arm swings, and playful finger-pointing gestures. A portable speaker behind them plays a fast Japanese electronic pop instrumental audible within the plaza. The camera pushes in with small amplitude at slow speed while preserving their complete choreography and the plaza architecture in the background. [Shot 2] At 00:04.000, the camera cuts to a slightly lower medium-wide angle and trucks right at normal speed as the three dancers turn together, raise both arms above their heads, and execute a synchronized half-spin; their layered skirts, ribbons, and ponytails move naturally with the rotation. They immediately return to a triangular formation and perform rapid hand-heart gestures followed by two energetic hops in time with the beat. [Shot 3] At 00:07.500, the camera cuts to a frontal wide shot and pulls out with small amplitude at slow speed as the dancers step forward together, cross their arms, then open them outward in a crisp symmetrical flourish. At the end of the 10-second routine, all three land simultaneously in a cheerful final pose facing the camera: the center dancer forms a heart with both hands while the two side dancers point toward her and lift one foot behind them. Their costumes and hair settle naturally as they hold the pose through the final frame.

overall_soundscape: The portable speaker produces a clear electronic dance beat across the open plaza, accompanied by synchronized sneaker impacts, light fabric swishes, and the soft jingle of costume accessories. A gentle city ambience continues underneath with distant pedestrian chatter, faint traffic, and a brief burst of applause as the dancers reach their final pose.

non_diegetic_music: N/A" | tee output5.log

python examples/offline_inference/minimax_h3/end2end.py \
  --model /data/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 4 \
  --steps 10 \
  --height 768 \
  --width 1344 \
  --quality high \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..." | tee output3.log


python examples/offline_inference/minimax_h3/end2end.py \
  --model /data1/models/MiniMax-H3/FL2VA/ \
  --task t2va \
  --quantization fp8 \
  --usp 8 \
  --steps 10 \
  --height 768 \
  --width 1344 \
  --quality high \
  --aspect-ratio 16:9 \
  --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./h3_prof","torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_record_shapes":false}' \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..." | tee output4.log