import gradio as gr
import numpy as np
from PIL import Image

def generate_image(seed1, seed2, content, style, truncation, c0, c1, c2, c3, c4, c5, c6, start_layer, end_layer):
    seed1 = int(seed1)
    seed2 = int(seed2)

    scale = 1
    params = {'c0': c0,
          'c1': c1,
          'c2': c2,
          'c3': c3,
          'c4': c4,
          'c5': c5,
          'c6': c6}

    param_indexes = {'c0': 0,
              'c1': 1,
              'c2': 2,
              'c3': 3,
              'c4': 4,
              'c5': 5,
              'c6': 6}

    directions = []
    distances = []
    for k, v in params.items():
        directions.append(latent_dirs[param_indexes[k]])
        distances.append(v)

    w1 = model.sample_latent(1, seed=seed1).detach().cpu().numpy()
    w1 = [w1]*model.get_max_latents() # one per layer
    im1 = model.sample_np(w1)

    w2 = model.sample_latent(1, seed=seed2).detach().cpu().numpy()
    w2 = [w2]*model.get_max_latents() # one per layer
    im2 = model.sample_np(w2)
    combined_im = np.concatenate([im1, im2], axis=1)
    input_im = Image.fromarray((combined_im * 255).astype(np.uint8))
    
    mixed_w = mix_w(w1, w2, content, style)
    return input_im, display_sample_pytorch(seed1, truncation, directions, distances, scale, int(start_layer), int(end_layer), w=mixed_w, disp=False)

# Create input components with better organization and descriptions
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 ClothingGAN: AI-Powered Clothing Design Generator")
    gr.Markdown("Generate and customize clothing designs using AI. Mix different styles and adjust various attributes to create unique fashion pieces.")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Input Images", type="pil")
            output_image = gr.Image(label="Generated Design", type="pil")
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎲 Generation Settings")
                seed1 = gr.Number(default=0, label="Seed 1", precision=0, info="Random seed for first image")
                seed2 = gr.Number(default=0, label="Seed 2", precision=0, info="Random seed for second image")
                truncation = gr.Slider(minimum=0, maximum=1, default=0.5, label="Truncation", info="Controls image quality vs diversity")
            
            with gr.Group():
                gr.Markdown("### 🎯 Style Mixing")
                content = gr.Slider(label="Structure", minimum=0, maximum=1, default=0.5, info="How much structure to take from first image")
                style = gr.Slider(label="Style", minimum=0, maximum=1, default=0.5, info="How much style to take from second image")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Layer Settings")
                start_layer = gr.Number(default=0, label="Start Layer", precision=0, info="Starting layer for style mixing")
                end_layer = gr.Number(default=14, label="End Layer", precision=0, info="Ending layer for style mixing")
            
            with gr.Group():
                gr.Markdown("### 🎨 Style Attributes")
                slider_max_val = 20
                slider_min_val = -20
                
                c0 = gr.Slider(label="Sleeve & Size", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Adjust sleeve length and overall size")
                c1 = gr.Slider(label="Dress - Jacket", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Transform between dress and jacket styles")
                c2 = gr.Slider(label="Female Coat", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Add feminine coat elements")
                c3 = gr.Slider(label="Coat", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Add coat elements")
                c4 = gr.Slider(label="Graphics", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Add graphic patterns")
                c5 = gr.Slider(label="Dark", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Adjust darkness of colors")
                c6 = gr.Slider(label="Less Cleavage", minimum=slider_min_val, maximum=slider_max_val, default=0, info="Adjust neckline coverage")

    inputs = [seed1, seed2, content, style, truncation, c0, c1, c2, c3, c4, c5, c6, start_layer, end_layer]
    outputs = [input_image, output_image]
    
    gr.Markdown("""
    ### How to use:
    1. Adjust the seeds to generate different base images
    2. Use Structure and Style sliders to mix the two images
    3. Fine-tune the style attributes to customize your design
    4. The result will update automatically as you make changes
    """)

demo.launch(share=True) 