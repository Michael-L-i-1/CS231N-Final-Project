import matplotlib.pyplot as plt
import numpy as np
import random
import json
import os
from tqdm import tqdm

# ------------------------------------------------------------------
# Tunables
FIGSIZE      = (12, 12)            # inches
DPI          = 100
CIRCLE_R     = 0.04                # person-circle radius
LABEL_PAD    = 0.45                # label radius from centre
MIN_SEPARATE = 0.22                # min distance between circles
EPS_OUT      = 0.005               # step past bbox so tail is "just outside"
# ------------------------------------------------------------------

def draw_person(ax, x, y):
    ax.add_patch(plt.Circle((x, y), CIRCLE_R, fill=False, linewidth=2))

def _intersection_with_bbox(lx, ly, fx, fy, xmin, xmax, ymin, ymax):
    """Return the first point where the (lx,ly)→(fx,fy) ray meets the bbox."""
    dx, dy = fx - lx, fy - ly
    t_candidates = []

    if dx:  # left / right edges
        t = (xmin - lx) / dx
        if t >= 0:
            y = ly + t * dy
            if ymin <= y <= ymax: t_candidates.append(t)
        t = (xmax - lx) / dx
        if t >= 0:
            y = ly + t * dy
            if ymin <= y <= ymax: t_candidates.append(t)

    if dy:  # bottom / top edges
        t = (ymin - ly) / dy
        if t >= 0:
            x = lx + t * dx
            if xmin <= x <= xmax: t_candidates.append(t)
        t = (ymax - ly) / dy
        if t >= 0:
            x = lx + t * dx
            if xmin <= x <= xmax: t_candidates.append(t)

    if not t_candidates:      # should never happen
        return lx, ly
    t_edge = min(t_candidates)
    # Step EPS_OUT beyond the edge so arrow tail is outside bbox
    return lx + (t_edge + EPS_OUT) * dx, ly + (t_edge + EPS_OUT) * dy

def generate_diagram(names_pool, min_n=2, max_n=6,
                     figsize=FIGSIZE, dpi=DPI, save_to=None, seed=None, max_attempts=1000):
    """Generate a diagram with randomly positioned names.
    
    Args:
        names_pool: List of names to sample from
        save_to: Path to save the generated diagram image
        seed: Random seed for reproducibility
        max_attempts: Maximum placement attempts before repositioning strategy changes
    
    Returns:
        Tuple of (fig, ordered_names, positions)
    """
    random.seed(seed)

    # ------------------- 1. pick names ------------------------
    n     = random.randint(min_n, max_n)
    names = random.sample(names_pool, n)

    # ------------------- 2. set up canvas ---------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # ------------------- 3. place circles ---------------------
    pos = {}
    attempt_count = 0
    while len(pos) < n:
        x, y = random.uniform(0.2, 0.8), random.uniform(0.25, 0.8)
        if all(np.hypot(x - px, y - py) > MIN_SEPARATE for px, py in pos.values()):
            pos[names[len(pos)]] = (x, y)
        else:
            attempt_count += 1
            if attempt_count > max_attempts:
                # Strategy change: reduce MIN_SEPARATE temporarily or
                # clear some positions and retry with more space
                print(f"Reached max attempts ({max_attempts}), adjusting strategy...")
                if len(pos) > 1:  # Keep at least one point
                    # Remove the last few points and try again
                    for _ in range(min(3, len(pos))):
                        pos.popitem()
                
                # Temporarily reduce minimum separation requirement
                temp_min_separate = MIN_SEPARATE * 0.9
                
                # Try with reduced constraints
                if all(np.hypot(x - px, y - py) > temp_min_separate for px, py in pos.values()):
                    pos[names[len(pos)]] = (x, y)
                
                # Reset counter
                attempt_count = 0
    for (x, y) in pos.values():
        draw_person(ax, x, y)

    # ------------------- 4. add labels ------------------------
    texts = {}
    for i, name in enumerate(names):
        angle = 2 * np.pi * i / n
        lx = 0.5 + LABEL_PAD * np.cos(angle)
        ly = 0.5 + LABEL_PAD * np.sin(angle)
        txt = ax.text(
            lx, ly, name,
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1)
        )
        texts[name] = txt

    # IMPORTANT: draw once so we can get accurate text bboxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ------------------- 5. draw arrows -----------------------
    inv = ax.transData.inverted()
    for name in names:
        fx, fy = pos[name]          # figure centre
        txt = texts[name]
        # bbox in display-coords →  data-coords
        bb_disp = txt.get_window_extent(renderer=renderer)
        (xmin, ymin), (xmax, ymax) = inv.transform(bb_disp)
        lx, ly = txt.get_position()  # label centre

        tail_x, tail_y = _intersection_with_bbox(
            lx, ly, fx, fy, xmin, xmax, ymin, ymax
        )

        ax.annotate(
            '', xy=(fx, fy), xytext=(tail_x, tail_y),
            arrowprops=dict(arrowstyle='->', lw=2)
        )

    if save_to:
        fig.savefig(save_to, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # Get the left-to-right ordering of names based on x-coordinates
    ordered_names = sorted(pos.keys(), key=lambda name: pos[name][0])
    
    return names, ordered_names, pos   # Return both original and ordered names

# ------------------------------------------------------------------
if __name__ == '__main__':
    names_pool = [
        'Alice', 'Bob', 'Charlie', 'David', 'Emily',
        'Frank', 'Grace', 'Hannah', 'Ivan', 'Judy'
    ]
    
    # Generate multiple diagrams (1000 at a time)
    batch_size = 500
    
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    # Check if metadata.json exists and load it
    metadata_path = 'dataset/metadata.json'
    all_metadata = []
    start_index = 0
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        # Find the highest index used so far
        if all_metadata:
            # Extract indices from image paths (format: 'dataset/image_{i}.png')
            existing_indices = [int(item['image_path'].split('_')[1].split('.')[0]) 
                               for item in all_metadata]
            start_index = max(existing_indices) + 1
    
    # Calculate the end index for this batch
    end_index = start_index + batch_size
    
    # Use tqdm to show progress
    for i in tqdm(range(start_index, end_index), desc="Generating diagrams", unit="image"):
        seed = i  # Use index as seed for reproducibility
        filename = f'dataset/image_{i}.png'
        _, ordered_names, positions = generate_diagram(names_pool, save_to=filename, seed=seed)
        
        # Create metadata entry for this image
        metadata = {
            "order": ordered_names,
            "image_path": filename
        }
        
        # Add to our collection
        all_metadata.append(metadata)
    
    # Save all metadata to a single JSON file
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nGenerated images {start_index} to {end_index-1}")
    print(f"Updated metadata saved to {metadata_path}")
    print(f"Total dataset size: {len(all_metadata)} images")
