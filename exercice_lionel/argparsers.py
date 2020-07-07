import argparse


def argparser(description="DQN Exercise"):
    """Create an argparse.ArgumentParser"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--with_layernorm', type=bool, default=True)
    parser.add_argument('--rollout_len', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--target_update', type=int, default=200)
    parser.add_argument('--memory_size', type=int, default=int(5e4))
    parser.add_argument('--num_iters', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=int(3e-4))
    parser.add_argument('--clip_norm', type=float, default=1.0)
    # Exploration-related hyper-parameters
    parser.add_argument('--eps_beg', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--eps_decay', type=float, default=int(2e4))
    parser.add_argument('--learning_rate', type=float, default=int(1e-6))

    return parser
