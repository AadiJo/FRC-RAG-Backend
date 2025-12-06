# Configuration file: backend/src/core/filter_config.py
"""
Configuration settings for post-processing filter
"""

# Post-processing filter configuration
POST_PROCESSING_CONFIG = {
    # Enable/disable post-processing
    'enable_post_processing': True,
    
    # Reranking configuration
    'enable_reranking': True,
    'cross_encoder_model': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',
    
    # Relevance filtering configuration
    'enable_relevance_filtering': True,
    'min_relevance_score': 0.35,  # Documents below this score are filtered out
    
    # Diversity filtering configuration  
    'enable_diversity_filtering': True,
    'max_similarity_threshold': 0.75,  # Remove documents more similar than this
    
    # Processing parameters
    'initial_k_multiplier': 2.0,  # Retrieve k * multiplier documents for filtering
    'min_document_length': 50,    # Minimum words in document
    'max_document_length': 1000,  # Maximum words in document
}

# FRC-specific keyword weights for relevance scoring
FRC_KEYWORD_WEIGHTS = {
    # High-value technical terms
    'high_value': {
        'weight': 3,
        'keywords': [
            'motor', 'gear', 'ratio', 'wheel', 'sensor', 'encoder', 'gyro',
            'autonomous', 'teleop', 'programming', 'pid', 'control', 'feedback',
            'intake', 'shooter', 'drivetrain', 'elevator', 'arm', 'chassis',
            'swerve', 'tank', 'holonomic', 'camera', 'vision', 'apriltag',
            'pathfinding', 'odometry', 'kinematics', 'trajectory'
        ]
    },
    
    # Medium-value design terms
    'medium_value': {
        'weight': 2,
        'keywords': [
            'design', 'build', 'manufacture', 'material', 'aluminum', 'steel',
            'polycarbonate', 'weight', 'strength', 'durability', 'testing',
            'prototype', 'iteration', 'optimization', 'efficiency', 'power',
            'battery', 'pneumatic', 'hydraulic', 'mechanical', 'electrical'
        ]
    },
    
    # Low-value general terms
    'low_value': {
        'weight': 1,
        'keywords': [
            'robot', 'team', 'competition', 'match', 'alliance', 'field',
            'game', 'piece', 'score', 'points', 'strategy', 'first', 'frc',
            'robotics', 'student', 'mentor', 'sponsor', 'award', 'regional'
        ]
    }
}