from collections import OrderedDict
from tqdm import tqdm
from dataset.bi_sequence_dataset import get_dataloader
from config.config import Config
from config.file_utils import cycle
from trainer.trainer import TrainerED

def main():
    # create experiment cfg containing all hyperparameters
    cfg = Config('train')

    # create network and training agent
    tr_agent = TrainerED(cfg)

    # load from checkpoint if provided
    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader_all = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        
        # Initialize running loss accumulators
        running_losses = {}
        running_count = 0
        
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data)

            # Update running averages
            for k, v in losses.items():
                if k not in running_losses:
                    running_losses[k] = 0.0
                running_losses[k] += v.item()
            running_count += 1
            
            # Calculate running averages
            avg_losses = {k: v / running_count for k, v in running_losses.items()}
            
            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({f"{k}_avg": f"{v:.4f}" for k, v in avg_losses.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                outputs, losses = tr_agent.val_func(data)

            clock.tick()

            tr_agent.update_learning_rate()

        if clock.epoch % 10 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        if clock.epoch % 10 == 0:
            tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
