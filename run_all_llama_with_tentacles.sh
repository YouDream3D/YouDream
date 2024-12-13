text="a zoomed out photo of a llama with octopus tentacle body"
avatar_name="Llama_rerun_trial1"

python train.py \
  --log.exp_name "pretrained_balloon_animal_llama_tentacles" \
  --log.pretrain_only True \
  --prompt.scene canonical-A \
  --prompt.pose_prompt depth \
  --prompt.scale_init 1.5 \
  --optim.iters 10000 \
  --render.train_h 128 \
  --render.train_w 128 \
  --render.eval_h 512 \
  --render.eval_w 512 \
  --prompt.init_mesh "./Animal_priors/llama_tentacles/scene.obj" \
  --prompt.init_pose "./Animal_priors/llama_tentacles/modified_keypoints.json" \
  --guide.loss_type "dreamfusion" \
  --prompt.view_prompt "dreamfusion" \
  --render.nerf_type_int 3 \
  --render.blob_radius 0 

pretrained_ckpt="./outputs/pretrained_balloon_animal_llama_tentacles/checkpoints/step_010000.pth"

python train.py \
  --prompt.init_pose "./Animal_priors/llama_tentacles/modified_keypoints.json" \
  --prompt.init_mesh "./Animal_priors/llama_tentacles/scene.obj" \
  --guide.text "${text}" \
  --optim.ckpt "${pretrained_ckpt}" \
  --guide.lambda_rgb 0.01 \
  --guide.time_sampling "annealed_hifa" \
  --guide.max_timestep 0.98 \
  --guide.min_timestep 0.4 \
  --optim.seed 2345 \
  --log.exp_name "canonical/${avatar_name}" \
  --optim.iters 10000 \
  --optim.lr 0.001 \
  --optim.control_max 1 \
  --optim.control_min 1 \
  --render.train_h 128 \
  --render.train_w 128 \
  --render.eval_h 512 \
  --render.eval_w 512 \
  --render.theta_min 60 \
  --render.theta_max 120 \
  --prompt.scale_init 1.5 \
  --guide.guidance_adjust "anneal" \
  --optim.optimizer "adam" \
  --guide.grad_clip False \
  --guide.grad_norm True \
  --guide.loss_type "dreamfusion" \
  --prompt.view_prompt "dreamfusion" \
  --render.nerf_type_int 3 \
  --render.bg_mode "nerf" \
  --render.eval_bg_mode "nerf" \
  --render.blob_radius 0

