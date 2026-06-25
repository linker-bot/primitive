string label
string prompt
float32 score
uint32 instance_id
string frame_id
float32[3] center
float32[3] size
float32[3] min
float32[3] max

# Oriented bounding box: principal axis quaternion (x,y,z,w)
# Identity when orientation is unavailable.
float32[4] orientation

# Approach direction: average top-surface normal in frame_id
float32[3] top_normal

# Grasp affordance hint: "precision" / "lateral" / "power" / ""
string grasp_type

# Track stability mode: fast_track / refined / corrected / recovered / reinitialized / ""
string track_mode
