text="a tiger walking"
avatar_name="Tiger"

python train.py \
  --log.exp_name "pretrained_balloon_animal_tiger" \
  --log.pretrain_only True \
  --prompt.scene canonical-A \
  --prompt.pose_prompt depth \
  --prompt.scale_init 1.5 \
  --optim.iters 10000 \
  --render.train_h 128 \
  --render.train_w 128 \
  --render.eval_h 512 \
  --render.eval_w 512 \
  --prompt.init_mesh "./Animal_priors/tiger/scene.obj" \
  --prompt.init_pose "./Animal_priors/tiger/modified_keypoints.json" \
  --guide.loss_type "dreamfusion" \
  --prompt.view_prompt "dreamfusion" \
  --render.nerf_type_int 3 \
  --render.blob_radius 0 

pretrained_ckpt="./outputs/pretrained_balloon_animal_tiger/checkpoints/step_010000.pth"

python train.py \
  --guide.text "${text}" \
  --optim.seed 0 \
  --log.exp_name "canonical/${avatar_name}" \
  --optim.iters 10000 \
  --optim.lr 0.001 \
  --render.train_h 128 \
  --render.train_w 128 \
  --render.eval_h 512 \
  --render.eval_w 512 \
  --prompt.init_pose "./Animal_priors/tiger/modified_keypoints.json" \
  --prompt.init_mesh "./Animal_priors/tiger/scene.obj" \
  --prompt.scale_init 1.5 \
  --optim.control_max 1 \
  --optim.control_min 0.2 \
  --guide.guidance_adjust "anneal" \
  --guide.guidance_scale 50 \
  --guide.guidance_scale_multiplier 50 \
  --guide.min_timestep 0.4 \
  --render.theta_min 45 \
  --render.theta_max 120 \
  --optim.optimizer "adam" \
  --guide.grad_clip False \
  --guide.grad_norm True \
  --guide.loss_type "dreamfusion" \
  --prompt.view_prompt "dreamfusion" \
  --render.nerf_type_int 3 \
  --render.bg_mode "nerf" \
  --render.eval_bg_mode 'nerf' \
  --render.blob_radius 0 \
  --guide.lambda_rgb 0.01 \
  --optim.ckpt "${pretrained_ckpt}"
