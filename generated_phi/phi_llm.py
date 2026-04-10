# Auto-generated potential Φ module (LLM). UTC: 2026-03-22T00:39:42Z
# Validate with: python scripts/validate_phi_poc.py --phi-module <import_path>
import numpy as np

def phi(state: dict, role: str) -> float:
    """
    Potential function for GRF multi-agent shaping.
    Returns value in [0.0, 1.0] based on state and role.
    """
    # Extract state components
    my_pos = np.array(state['my_pos'], dtype=np.float32)
    ball_pos = np.array(state['ball_pos'][:2], dtype=np.float32)  # Use only x,y
    has_ball = state['has_ball']
    ball_dist_to_goal = state['ball_dist_to_goal']
    dist_to_ball = state['dist_to_ball']
    nearest_opp_dist = state['nearest_opp_dist']
    teammates_pos = [np.array(p, dtype=np.float32) for p in state['teammates_pos']]
    opponents_pos = [np.array(p, dtype=np.float32) for p in state['opponents_pos']]
    score_diff = state['score_diff']
    
    # Normalization constants for GRF field
    FIELD_X_MAX = 1.0
    FIELD_Y_MAX = 0.42
    GOAL_X = 1.0 if state['is_left_team'] else -1.0  # Attack direction
    
    # Base potential components
    components = []
    
    # 1. Ball progress toward opponent goal (most important)
    # Normalized: 1.0 at opponent goal line, 0.0 at own goal line
    ball_progress = 1.0 - (ball_dist_to_goal / (2.0 * FIELD_X_MAX))
    components.append(0.4 * np.clip(ball_progress, 0.0, 1.0))
    
    # 2. Ball possession bonus
    if has_ball:
        components.append(0.2)
    
    # Role-specific components
    if role == "ball_carrier":
        # For ball carrier: encourage moving toward goal while maintaining control
        if has_ball:
            # Distance to nearest opponent (safety)
            safety = np.clip(nearest_opp_dist / 0.3, 0.0, 1.0)  # 0.3 is typical challenge distance
            components.append(0.15 * safety)
            
            # Progress toward goal with ball
            my_to_goal = np.abs(GOAL_X - my_pos[0]) / (2.0 * FIELD_X_MAX)
            progress = 1.0 - my_to_goal
            components.append(0.25 * np.clip(progress, 0.0, 1.0))
    
    elif role in ["left_winger", "right_winger"]:
        # For wingers: encourage width and forward positioning
        side = 1.0 if role == "right_winger" else -1.0
        ideal_y = side * FIELD_Y_MAX * 0.7  # Wide position
        
        # Width alignment
        width_alignment = 1.0 - np.abs(my_pos[1] - ideal_y) / (2.0 * FIELD_Y_MAX)
        components.append(0.15 * np.clip(width_alignment, 0.0, 1.0))
        
        # Forward positioning (but not too far ahead of ball)
        forwardness = np.clip((my_pos[0] - ball_pos[0] + FIELD_X_MAX) / (2.0 * FIELD_X_MAX), 0.0, 1.0)
        components.append(0.1 * forwardness)
        
        # Support distance to ball (not too close, not too far)
        support_dist = np.clip(1.0 - np.abs(dist_to_ball - 0.2) / 0.3, 0.0, 1.0)
        components.append(0.05 * support_dist)
    
    elif role == "trailing_mid":
        # For trailing midfielder: support behind ball
        # Position behind ball relative to goal direction
        if state['is_left_team']:
            behind = np.clip((ball_pos[0] - my_pos[0]) / (2.0 * FIELD_X_MAX), 0.0, 1.0)
        else:
            behind = np.clip((my_pos[0] - ball_pos[0]) / (2.0 * FIELD_X_MAX), 0.0, 1.0)
        components.append(0.2 * behind)
        
        # Central positioning
        centrality = 1.0 - np.abs(my_pos[1]) / FIELD_Y_MAX
        components.append(0.1 * centrality)
    
    else:  # "default" or any unknown role
        # Default behavior: support based on ball proximity and positioning
        # Encourage being closer to ball than nearest opponent
        if len(opponents_pos) > 0:
            opp_dists = [np.linalg.norm(ball_pos - opp) for opp in opponents_pos]
            nearest_opp_to_ball = min(opp_dists) if opp_dists else 1.0
            ball_advantage = np.clip((nearest_opp_to_ball - dist_to_ball) / 0.5, 0.0, 1.0)
            components.append(0.15 * ball_advantage)
        
        # Positioning relative to ball and goal
        if state['is_left_team']:
            forward = np.clip((my_pos[0] - ball_pos[0] + FIELD_X_MAX) / (2.0 * FIELD_X_MAX), 0.0, 1.0)
        else:
            forward = np.clip((ball_pos[0] - my_pos[0] + FIELD_X_MAX) / (2.0 * FIELD_X_MAX), 0.0, 1.0)
        components.append(0.15 * forward)
    
    # 3. Team spacing (for all roles except ball carrier)
    if role != "ball_carrier" and len(teammates_pos) > 0:
        # Avoid clustering with teammates
        teammate_dists = []
        for teammate in teammates_pos:
            dist = np.linalg.norm(my_pos - teammate)
            teammate_dists.append(dist)
        
        if teammate_dists:
            avg_teammate_dist = np.mean(teammate_dists)
            spacing = np.clip(avg_teammate_dist / 0.4, 0.0, 1.0)  # 0.4 is good spacing distance
            components.append(0.1 * spacing)
    
    # 4. Score advantage (small bonus for leading)
    score_bonus = np.clip(score_diff * 0.05, -0.1, 0.1) + 0.1  # Map to [0.0, 0.2]
    components.append(0.1 * score_bonus)
    
    # Combine components with weighted sum
    potential = sum(components)
    
    # Ensure bounded output
    return float(max(0.0, min(1.0, potential)))