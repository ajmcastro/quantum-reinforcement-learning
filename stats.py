from collections import namedtuple

EpisodeStats = namedtuple(
    'EpisodeStats', [
        'episode_results', 'episode_steps', 'episode_rewards', 'explored_states'
    ]
)
