from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

def plot_adaptation_process(support_set, query_set, predictions):
    """Visualize meta-learning adaptation process"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot support set adaptation
    ax1.scatter(support_set[:,0], support_set[:,1], c='blue', label='Support')
    ax1.plot(predictions['pre_adapt'], label='Pre-adaptation', linestyle='--')
    ax1.plot(predictions['post_adapt'], label='Post-adaptation')
    ax1.set_title("Adaptation Process")
    ax1.legend()
    
    # Plot query set performance
    errors = np.abs(query_set - predictions['final'])
    ax2.hist(errors, bins=20, alpha=0.7)
    ax2.set_title("Final Prediction Errors")
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)

def display_env_state(env, render_mode: str = 'rgb_array'):
    """Render environment state in notebooks"""
    frame = env.render(mode=render_mode)
    plt.imshow(frame)
    plt.axis('off')
    display(plt.gcf())
    plt.close()