from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from core.nerf.utils.nerf_utils import NeRFType


@dataclass
class RenderConfig:
    """ Parameters for the NeRF Renderer """
    # Whether to use CUDA raymarching
    cuda_ray: bool = False
    grid_size: int = 128
    # Maximal number of steps sampled per ray with cuda_ray
    max_steps: int = 1024
    # Number of steps sampled when rendering without cuda_ray
    num_steps: int = 128
    # Number of upsampled steps per ray
    upsample_steps: int = 0
    # Iterations between updates of extra status
    update_extra_interval: int = 16
    # batch size of rays at inference
    max_ray_batch: int = 4096
    # threshold for density grid to be occupied
    density_thresh: float = 10
    # Render width for training
    train_w: int = 64
    # Render height for training
    train_h: int = 64
    # Render width for inference
    eval_w: int = 128
    # Render height for inference
    eval_h: int = 128
    # Render angle theta for inference
    eval_theta: float = 80
    # Render radius rate
    eval_radius_rate: float = 1.2
    eval_disable_background: bool = True
    eval_save_video: bool = True
    eval_fix_camera: bool = False
    eval_fix_animation: bool = False
    eval_camera_track: str = 'circle'
    # Whether to randomly jitter sampled camera
    jitter_pose: bool = False
    # Assume the scene is bounded in box(-bound,bound)
    bound: float = 1.5
    # dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)
    dt_gamma: float = 0
    # minimum near distance for camera
    min_near: float = 0.1
    # training camera theta range, default: (0, 150), (0, 100), (0, 120)
    theta_min: float = 30
    theta_max: float = 150
    theta_range: Tuple[float, float] = (60, 120)  # (0, 180)
    # training camera phi range, default: (0, 360)
    phi_range: Tuple[float, float] = (0, 360)
    # training camera radius range
    radius_range: Tuple[float, float] = (1.0, 2.0)
    # training camera fovy range
    fovy_range: Tuple[float, float] = (40, 70)
    # Vertical jitter
    vertical_jitter: Optional[Tuple[float, float]] = (-0.5, +0.5)
    # Set [0, angle_overhead] as the overhead region
    angle_overhead: float = 30
    # Set [0, angle_front] as the front region
    angle_front: float = 90
    # Which NeRF backbone to use
    backbone: str = 'tiledgrid'  # 'tiledgrid'
    # Define the nerf output type
    nerf_type: NeRFType = NeRFType['rgb']
    nerf_type_int: int = 0 # 0: latent, 1: latent_tune, 2: dual, 3: rgb
    # For each batch, all views are covered
    batched_view: bool = False
    # uniform_sphere_rate
    uniform_sphere_rate: float = 0.0
    # density blob
    density_blob: str = 'none'
    blob_radius: float = 0.5

    # background augmentation
    bg_mode: str = 'nerf'  # {'nerf', 'random', 'white', 'black', 'gaussian', 'zero'}
    eval_bg_mode: str = 'nerf' # {'nerf', 'random', 'white', 'black', 'gaussian', 'zero'}
    # if positive, use a background model at sphere(bg_radius)
    bg_radius: float = 2.5

    # motion field
    skeletal_motion: str = 'ours'  # {'closest_joint', 'closest_vertex'}
    skeletal_motion_thres: float = 0.001
    skeletal_motion_encode: str = 'freq'
    non_rigid_motion: str = 'none'
    joint_train: bool = True

    lambda_zvar: float = 0.07
    lambda_2d_normal_smooth: float = 0.0
    lambda_3d_normal_smooth: float = 0.0

    save_mesh: bool = False

@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str = ""
    # Loss type
    loss_type: str = "sjc"
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # A huggingface diffusion model to use
    diffusion_name: str = "runwayml/stable-diffusion-v1-5"
    diffusion_fp16: bool = False
    # ControlNet
    use_controlnet: bool = True
    # Guidance scale
    guidance_scale: float = 50.0
    guidance_scale_multiplier: float = 50
    guidance_adjust: str = 'constant'
    # ControlNet scale
    controlnet_scale: float = 1.0
    lambda_rgb: float = 0.1
    lambda_sds: float = 1

    # Time step
    min_timestep: float = 0.5
    max_timestep: float = 0.98
    time_sampling: str = 'annealed_hifa' # ['uniform', 'linear', 'annealed', 'annealed_hifa', 'annealed_random_hifa']
    time_prior: str = 'uniform'

    # Gradient Clip
    grad_clip: bool = False
    grad_norm: bool = True
    perpneg: bool = False
    # weight_prompts: bool = True


@dataclass
class PromptConfig:
    # View
    view_prompt: str = 'sjc'
    append_direction: bool = True
    pose_prompt: str = 'pose'  # {'pose', 'depth', 'depth,pose', ...}
    num_person: int = 1
    scene: str = 'canonical-A'  # {'random', 'canonical', 'dance', 'basketball', ...}
    pop_transl: bool = True
    # Object
    num_object: int = 0
    occlusion_culling: str = True
    init_mesh: str = "" #/work/osaha_umass_edu/exp/EXPS/DreamWaltz/Animal_balloon_inits_and_kpts/Animals/alligator_v3/scene.obj"
    init_pose: str = "" #/work/osaha_umass_edu/exp/EXPS/DreamWaltz/Animal_balloon_inits_and_kpts/Animals/alligator_v3/modified_keypoints.obj"
    scale_init: float = 0.5

@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Batch size
    batch_size: int = 1
    # SDS Loss, default: 1.0
    lambda_guidance: float = 1.0
    # Shape Loss: mesh-guidance, default: 5e-6
    lambda_shape: float = 0.0
    # Sparsity Loss: opacity (default=[1e-3, 5e-3]), entropy (default=5e-4), emptiness (default=1.0)
    lambda_opacity: float = 0.0
    lambda_entropy: float = 1e-2
    lambda_emptiness: float = 0.0
    # Sparsity Scheduler
    sparsity_multiplier: float = 20
    sparsity_step: float = 1.0  # 0.5
    # Seed for experiment
    seed: int = 2345
    # Total iters
    iters: int = 10000
    # Learning rate
    lr: float = 1e-3
    # use amp mixed precision training
    fp16: bool = True
    # Start shading at this iteration
    start_shading_iter: Optional[int] = None
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None
    ckpt_extra: Optional[str] = None
    # LR Policy
    lr_policy: str = 'constant'
    # Optimizer
    optimizer: str = 'adam'
    # Grad Scale
    grad_scale: bool = True
    lambda_zvar: float = 0.07
    lambda_monotonic: float = 1
    lambda_zentropy: float = 0.01
    lambda_bentropy_sum: float = 0.0001
    lambda_sparsity: float = 1.0
    lambda_2d_normal_smooth: float = 0.0
    lambda_3d_normal_smooth: float = 0.0
    control_scaling: str = "annealed" #"constant" #"annealed"
    control_max: float = 1.0
    control_min: float = 0.2



@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str = 'default'
    # Experiment output dir
    exp_root: Path = Path('outputs/')
    # How many steps between save step
    save_interval: int = 5000
    # How many steps between snapshot step
    snapshot_interval: int = 500
    evaluate_interval: int = 500
    # Run only test
    eval_only: bool = False
    # Run only pretrain
    pretrain_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 8
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 1
    # Skip decoding and vis only depth and normals
    skip_rgb: bool = False

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    animation: bool = False

    def __post_init__(self):
        if self.log.eval_only and (self.optim.ckpt is None and not self.optim.resume):
            logger.warning('NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True