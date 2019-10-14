import argparse
import os
import pickle

from envs.nine_rooms_env import NineRoomsEnv
from algos.dqn.dqn import DeepQLearner
from algos.dqn.replay_buffer import ReplayBuffer as DQNReplayBuffer
from tensorboardX import SummaryWriter

def get_args():
    arguments = argparse.ArgumentParser()
    
    arguments.add_argument('--run_id', type=str, default='main')
    # Global Training Parameters 
    arguments.add_argument('--num_episodes', type=int, default=1e5)

    # Global Q Learning Parameters
    arguments.add_argument('--buff_capacity', type=int, default=1e10)
    arguments.add_argument('--batch_size', type=int, default=512)
    arguments.add_argument('--eps_start', type=float, default=1)
    arguments.add_argument('--eps_end', type=float, default=0.01)
    arguments.add_argument('--eps_decay', type=int, default=200)

    # DQN Parameters
    arguments.add_argument('--target_update', type=int, default=10)
    arguments.add_argument('--update_steps', type=int, default=32)
    arguments.add_argument('--lr', type=float, default=1e-3)
    arguments.add_argument('--hidden_size', type=int, default=128)
    arguments.add_argument('--ddqn', type=int, default=0)
    arguments.add_argument('--opt', default='adam')

    # Environment
    arguments.add_argument('--k', type=int, default=1)
    arguments.add_argument('--grid_size', type=int, default=10)
    arguments.add_argument('--pct', type=float, default=0.75)
    arguments.add_argument('--pcf', type=float, default=0.5)
    arguments.add_argument('--max_steps', type=int, default=100)
    arguments.add_argument('--rnd_start', type=int, default=0)
    arguments.add_argument('--gamma', type=float, default=0.9)
    arguments.add_argument('--start_state_exclude_rooms', nargs='*', type=int, default=[])

    # Validation
    arguments.add_argument('--val_episode', type=int, default=10)
    arguments.add_argument('--val_rollouts', type=int, default=10)
    arguments.add_argument('--unseen', type=int, default=0)

    # Checkpointing
    arguments.add_argument('--save_root', type=str, default='save')

    # Others
    arguments.add_argument('--gpu', type=int, default=1)

    arguments.add_argument('--seed', type=int, default=101)

    return arguments.parse_args()

args = get_args()

if __name__ == '__main__':

    # Complete checkpointing logistics
    save_dir = os.path.join(args.save_root, args.run_id)

    ckpt_dir = os.path.join(save_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Write the argument
    with open(os.path.join(save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)
    print("Arguments Dumped.")

    writer = SummaryWriter(save_dir)
    
    buffer = DQNReplayBuffer(capacity=int(args.buff_capacity)) 

    # Replace with other initial start distribution
    rnd_start = (args.rnd_start == 1)

    env = NineRoomsEnv(
        grid_size=args.grid_size,
        K=args.k,
        pct=args.pct,
        pcf=args.pcf,
        max_steps=args.max_steps,
        rnd_start=rnd_start,
        start_state_exclude_rooms=args.start_state_exclude_rooms

    )

    # Dump the category map for the environment. This will be used to
    # instantiate the exact same environment when, say, we perform RL / demonstration data gathering for this environment.
    with open(os.path.join(save_dir, 'cell_cat_map.pkl'), 'wb') as f:
        pickle.dump({'cell_cat_map' : env.cell_cat_map}, f)
    print("Cell Category Map dumped.")

    algo_kwargs = dict(
        env=env,
        ddqn=args.ddqn,
        opt=args.opt,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        memory=buffer,
    ) 

    algo = DeepQLearner(**algo_kwargs)

    algo.train(
        num_episodes=int(args.num_episodes),
        batch_size=args.batch_size,
        target_update=args.target_update,
        update_steps=args.update_steps,
        val_rollouts=int(args.val_rollouts),
        val_episode=int(args.val_episode),
        writer=writer,
        ckpt_dir=ckpt_dir,
        unseen=args.unseen
    )

    writer.flush()