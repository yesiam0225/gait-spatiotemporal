"""Physical constants and default analysis configuration."""

G = 9.81  # m/s^2

DEFAULT_PHYSIOLOGICAL_BOUNDS = {
    'adult': {
        'stride_time_s': (0.7, 1.5),
        'stride_length_norm': (0.6, 1.7),
        'stance_pct': (50, 75),
        'swing_pct': (25, 50),
        'double_support_1_pct': (5, 25),
        'double_support_2_pct': (5, 25),
        'single_support_pct': (25, 50),
        'gait_speed_norm': (0.2, 0.6),
    },
    'child': {
        'stride_time_s': (0.5, 1.4),
        'stride_length_norm': (0.5, 1.6),
        'stance_pct': (50, 75),
        'swing_pct': (25, 50),
        'double_support_1_pct': (5, 30),
        'double_support_2_pct': (5, 30),
        'single_support_pct': (25, 50),
        'gait_speed_norm': (0.2, 0.7),
    },
}

OUTLIER_VARS = [
    'stride_time_s', 'stride_length_norm',
    'stance_pct', 'swing_pct',
    'double_support_1_pct', 'double_support_2_pct',
    'single_support_pct', 'gait_speed_norm',
]

VARS_TO_AGGREGATE = [
    'stride_time_s', 'stride_length_mm',
    'step_time_s', 'step_length_mm', 'step_width_mm',
    'stance_pct', 'swing_pct',
    'double_support_1_pct', 'double_support_2_pct', 'single_support_pct',
    'gait_speed_m_s',
    'stride_length_norm', 'step_length_norm', 'step_width_norm',
    'stride_time_norm', 'step_time_norm', 'gait_speed_norm',
]
