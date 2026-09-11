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
  --usp 4 \
  --steps 50 \
  --height 768 \
  --width 1344 \
  --aspect-ratio 16:9 \
  --duration 10 --seed 44211 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --prompts "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a single continuous medium-wide frontal shot frames three adult female cosplayers performing a synchronized otaku dance in the center of a spacious modern city plaza during bright late-afternoon daylight. The three dancers occupy most of the frame from head to below the knees, with their faces large enough to remain clearly visible. Each woman has a distinct, consistently recognizable face with natural symmetrical facial proportions, clear eyes, realistic skin texture, neatly defined lips, and stable facial features throughout the video. Soft, even frontal lighting illuminates all three faces without harsh shadows or backlighting. The center dancer wears a blue-and-white magical-girl-inspired costume with matching blue hair ribbons; the dancer on the left wears a red-and-black fantasy idol costume with dark twin ponytails; the dancer on the right wears a pastel pink-and-lavender costume with a star-shaped hair accessory. Their costumes, hairstyles, facial identities, body proportions, and positions remain consistent throughout the shot. All three face the camera continuously and maintain cheerful, relaxed expressions while performing a controlled synchronized dance consisting of two side steps, smooth alternating arm swings, coordinated finger-pointing gestures, and a heart-shaped hand gesture. Their hands remain away from their faces, and no dancer turns around, crosses in front of another dancer, or becomes occluded. The camera holds a stable frontal composition with only a very slow push in of small amplitude, keeping all three faces sharp and in focus while their hair ribbons and layered costumes move naturally. During the final two seconds, they step into a symmetrical formation and hold a clear finishing pose facing the camera: the center dancer forms a heart with both hands below her chin while the two side dancers raise one arm outward and smile naturally. The shot ends with all three faces unobstructed, sharply focused, and consistently illuminated.

overall_soundscape: A portable speaker plays a clear upbeat Japanese electronic dance instrumental across the plaza. Synchronized sneaker impacts, gentle costume-fabric movement, and the light jingle of accessories blend with distant pedestrian chatter and soft city traffic.

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