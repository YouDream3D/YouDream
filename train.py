import pyrallis
from configs import TrainConfig
from core.trainer import Trainer
from core.pretrainer import PreTrainer
from core.nerf.utils.nerf_utils import NeRFType

@pyrallis.wrap()
def main(cfg: TrainConfig):

    if cfg.render.nerf_type_int == 0:
        cfg.render.nerf_type = NeRFType['latent']
    if cfg.render.nerf_type_int == 1:
        cfg.render.nerf_type = NeRFType['latent_tune']
    if cfg.render.nerf_type_int == 2:
        cfg.render.nerf_type = NeRFType['dual']
    if cfg.render.nerf_type_int == 3:
        cfg.render.nerf_type = NeRFType['rgb']

    cfg.render.theta_range = (cfg.render.theta_min, cfg.render.theta_max)

    if cfg.log.pretrain_only:
        trainer = PreTrainer(cfg)
    else:
        trainer = Trainer(cfg)

    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
