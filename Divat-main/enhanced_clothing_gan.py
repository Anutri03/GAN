import gradio as gr
import numpy as np
from PIL import Image
import random
import time
import json
import os
import torch
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config

# --- Model Setup ---
selected_model = 'lookbook'

# Speed up computation
torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

config = Config(
    model='StyleGAN2',
    layer='style',
    output_class=selected_model,
    components=80,
    use_w=True,
    batch_size=5_000,
)

inst = get_instrumented_model(config.model, config.output_class,
                            config.layer, torch.device('cpu'), use_w=config.use_w)

path_to_components = get_or_compute(config, inst)
model = inst.model

# Load components
comps = np.load(path_to_components)
lst = comps.files
latent_dirs = []
latent_stdevs = []

load_activations = False
for item in lst:
    if load_activations:
        if item == 'act_comp':
            for i in range(comps[item].shape[0]):
                latent_dirs.append(comps[item][i])
        if item == 'act_stdev':
            for i in range(comps[item].shape[0]):
                latent_stdevs.append(comps[item][i])
    else:
        if item == 'lat_comp':
            for i in range(comps[item].shape[0]):
                latent_dirs.append(comps[item][i])
        if item == 'lat_stdev':
            for i in range(comps[item].shape[0]):
                latent_stdevs.append(comps[item][i])

# --- State Management ---
# Dictionary to store clothing catalog entries
clothing_catalog = {}
image_cache = {}

# --- Core Functions ---
def mix_w(w1, w2, content, style):
    """Mixes the latent codes w1 and w2 based on content and style ratios."""
    mixed_w = [None] * model.get_max_latents()
    num_layers = model.get_max_latents()
    split_point = 5  # Layers 0-4 for content, 5+ for style

    for i in range(0, min(split_point, num_layers)):
        mixed_w[i] = w1[i] * (1 - content) + w2[i] * content
    for i in range(split_point, num_layers):
        mixed_w[i] = w1[i] * (1 - style) + w2[i] * style
    return mixed_w

def generate_random_seeds():
    """Generate random seed values for both images"""
    seed1 = random.randint(0, 10000)
    seed2 = random.randint(0, 10000)
    image_cache.clear()
    return seed1, seed2

def generate_image(seed1, seed2, content, style, truncation, sleeve_length, size, dress_jacket, female_coat, coat, graphics, dark, cleavage, start_layer, end_layer):
    # Ensure inputs are integers where needed
    seed1 = int(seed1)
    seed2 = int(seed2)
    start_layer = int(start_layer)
    end_layer = int(end_layer)

    # Create a cache key based on base generation parameters
    cache_key = f"base_{seed1}_{seed2}_{content}_{style}_{truncation}_{start_layer}_{end_layer}"

    # Check cache for base latent codes (w1, w2) and mixed base
    if cache_key in image_cache:
        w1, w2, mixed_w_base, im1, im2 = image_cache[cache_key]
    else:
        # Generate base latents and images
        w1_raw = model.sample_latent(1, seed=seed1).detach().cpu().numpy()
        w1 = [w1_raw.copy() for _ in range(model.get_max_latents())]
        im1 = model.sample_np(w1)

        w2_raw = model.sample_latent(1, seed=seed2).detach().cpu().numpy()
        w2 = [w2_raw.copy() for _ in range(model.get_max_latents())]
        im2 = model.sample_np(w2)

        # Mix the styles for the base mixed latent
        mixed_w_base = mix_w(w1, w2, content, style)

        # Cache the results
        image_cache[cache_key] = (w1, w2, mixed_w_base, im1, im2)

    # Create displayable input images
    input_im1 = Image.fromarray((im1 * 255).astype(np.uint8))
    input_im2 = Image.fromarray((im2 * 255).astype(np.uint8))

    # Apply style attributes
    mixed_w_final = [w.copy() for w in mixed_w_base]
    scale = 1.0
    params = {
        'sleeve_length': sleeve_length * 0.5,
        'size': size * 0.8,
        'dress_jacket': dress_jacket * 1.0,
        'female_coat': female_coat * 1.0,
        'coat': coat * 1.0,
        'graphics': graphics * 1.2,
        'dark': dark * 1.0,
        'cleavage': cleavage * 0.8
    }
    param_indexes = {
        'sleeve_length': 0,
        'size': 1,
        'dress_jacket': 2,
        'female_coat': 3,
        'coat': 4,
        'graphics': 5,
        'dark': 6,
        'cleavage': 7
    }

    directions = []
    distances = []
    valid_indices = list(range(len(latent_dirs)))

    for k, v in params.items():
        idx = param_indexes.get(k)
        if idx is not None and idx in valid_indices:
            directions.append(latent_dirs[idx])
            distances.append(v * scale)

    # Apply attribute modifications
    if directions:
        for layer_idx in range(start_layer, min(end_layer + 1, len(mixed_w_final))):
            current_w = mixed_w_final[layer_idx]
            for dir_vec, dist in zip(directions, distances):
                if current_w.shape == dir_vec.shape:
                    current_w = current_w + dir_vec * dist
            mixed_w_final[layer_idx] = current_w

    # Generate final output image
    model.truncation = truncation
    torch.cuda.empty_cache()
    out_np = model.sample_np(mixed_w_final)
    output_im = Image.fromarray((out_np * 255).astype(np.uint8)).resize((500, 500), Image.LANCZOS)

    return input_im1, input_im2, output_im

# --- Catalog Functions ---
def save_to_catalog(seed1, seed2, description, image):
    """Save current clothing style to catalog"""
    global clothing_catalog
    
    # Generate unique ID for the entry
    entry_id = len(clothing_catalog) + 1
    
    # Save the image
    image_path = f"catalog_images/style_{entry_id}.png"
    os.makedirs("catalog_images", exist_ok=True)
    image.save(image_path)
    
    # Add to catalog
    clothing_catalog[entry_id] = {
        "seed1": int(seed1),
        "seed2": int(seed2),
        "description": description,
        "image_path": image_path
    }
    
    # Save catalog to file
    with open("clothing_catalog.json", "w") as f:
        json.dump(clothing_catalog, f)
    
    return gr.update(value=f"Style {entry_id} saved to catalog!")

def load_catalog():
    """Load the clothing catalog from file"""
    global clothing_catalog
    try:
        with open("clothing_catalog.json", "r") as f:
            clothing_catalog = json.load(f)
    except FileNotFoundError:
        clothing_catalog = {}
    return create_catalog_html()

def create_catalog_html():
    """Create HTML table of catalog entries"""
    if not clothing_catalog:
        return "<p>No styles saved in catalog yet.</p>"
    
    html = """
    <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 10px; border: 1px solid #ddd;">ID</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Image</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Description</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Seeds</th>
        </tr>
    """
    
    for entry_id, entry in clothing_catalog.items():
        html += f"""
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">Style {entry_id}</td>
            <td style="padding: 10px; border: 1px solid #ddd;">
                <img src="{entry['image_path']}" style="max-width: 200px;">
            </td>
            <td style="padding: 10px; border: 1px solid #ddd;">{entry['description']}</td>
            <td style="padding: 10px; border: 1px solid #ddd;">
                Seed1: {entry['seed1']}<br>
                Seed2: {entry['seed2']}
            </td>
        </tr>
        """
    
    html += "</table>"
    return html

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 ClothingGAN: AI-Powered Clothing Design Generator")
    
    with gr.Tabs():
        # --- Generator Tab ---
        with gr.TabItem("Generator"):
            with gr.Row():
                # --- Image Display Column ---
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_image1 = gr.Image(label="Input Image 1 (Seed 1)", type="pil", interactive=False)
                        with gr.Column(scale=1):
                            input_image2 = gr.Image(label="Input Image 2 (Seed 2)", type="pil", interactive=False)
                    output_image = gr.Image(label="Generated Design", type="pil", interactive=False)

                # --- Controls Column ---
                with gr.Column(scale=1):
                    # --- Generation Settings ---
                    with gr.Group():
                        gr.Markdown("### 🎲 Generation Settings")
                        with gr.Row():
                            seed1 = gr.Number(value=0, label="Seed 1", precision=0, info="Base image 1 seed")
                            seed2 = gr.Number(value=0, label="Seed 2", precision=0, info="Base image 2 seed")
                        random_seeds_btn = gr.Button("🎲 Generate Random Seeds")
                        truncation = gr.Slider(minimum=0, maximum=1, value=0.7, label="Truncation", info="Quality vs Diversity (0=avg, 1=diverse)")

                    # --- Style Mixing ---
                    with gr.Group():
                        gr.Markdown("### 🎯 Style Mixing")
                        content = gr.Slider(label="Structure Mix", minimum=0, maximum=1, value=0.5, info="0=Img1 Structure, 1=Img2 Structure")
                        style = gr.Slider(label="Appearance Mix", minimum=0, maximum=1, value=0.5, info="0=Img1 Appearance, 1=Img2 Appearance")

                    # --- Layer Settings ---
                    with gr.Group():
                        gr.Markdown("### ⚙️ Layer Settings (For Attributes)")
                        start_layer = gr.Number(value=0, label="Start Layer", precision=0, minimum=0, maximum=15, info="Apply attributes from this layer")
                        end_layer = gr.Number(value=7, label="End Layer", precision=0, minimum=0, maximum=15, info="Apply attributes up to this layer (inclusive)")

                    # --- Style Attributes ---
                    with gr.Group():
                        gr.Markdown("### 🎨 Style Attributes")
                        slider_max_val = 10
                        slider_min_val = -10
                        step_val = 0.5

                        sleeve_length = gr.Slider(label="Sleeve Length", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Adjust sleeve length only", step=step_val)
                        size = gr.Slider(label="Size", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Adjust overall clothing size", step=step_val)
                        dress_jacket = gr.Slider(label="Dress <-> Jacket", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Transform dress/jacket style", step=step_val)
                        female_coat = gr.Slider(label="Feminine Coat", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Add feminine coat style", step=step_val)
                        coat = gr.Slider(label="Coat Style", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Add general coat style", step=step_val)
                        graphics = gr.Slider(label="Graphics/Patterns", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Add/remove graphic patterns", step=step_val)
                        dark = gr.Slider(label="Light <-> Dark", minimum=slider_min_val, maximum=slider_max_val, value=0, info="Adjust overall darkness", step=step_val)
                        cleavage = gr.Slider(label="Neckline Coverage", minimum=slider_min_val, maximum=slider_max_val, value=0, info="More(+) / Less(-) Cleavage", step=step_val)

                    # --- Save to Catalog ---
                    with gr.Group():
                        gr.Markdown("### 💾 Save to Catalog")
                        catalog_description = gr.Textbox(
                            label="Style Description",
                            placeholder="Describe this clothing style (e.g., 'Red summer dress with floral pattern')"
                        )
                        save_to_catalog_btn = gr.Button("💾 Save to Catalog")
                        catalog_status = gr.Textbox(label="Status", interactive=False)

        # --- Catalog Tab ---
        with gr.TabItem("Clothing Catalog"):
            gr.Markdown("### 📚 Saved Clothing Styles")
            refresh_catalog_btn = gr.Button("🔄 Refresh Catalog")
            catalog_display = gr.HTML()

    # --- Define Inputs & Outputs for Main Function ---
    main_inputs = [seed1, seed2, content, style, truncation, sleeve_length, size, dress_jacket, female_coat, coat, graphics, dark, cleavage, start_layer, end_layer]
    main_outputs = [input_image1, input_image2, output_image]

    # --- Connect UI Events ---
    # 1. Random Seeds Button
    random_seeds_btn.click(
        fn=generate_random_seeds,
        inputs=None,
        outputs=[seed1, seed2],
        queue=False
    ).then(
        fn=generate_image,
        inputs=main_inputs,
        outputs=main_outputs
    )

    # 2. Save to Catalog Button
    save_to_catalog_btn.click(
        fn=save_to_catalog,
        inputs=[seed1, seed2, catalog_description, output_image],
        outputs=[catalog_status]
    )

    # 3. Refresh Catalog Button
    refresh_catalog_btn.click(
        fn=load_catalog,
        inputs=None,
        outputs=[catalog_display]
    )

    # 4. Connect sliders/number inputs to the main generation function
    for input_component in main_inputs:
        if isinstance(input_component, gr.Slider):
            input_component.release(
                fn=generate_image,
                inputs=main_inputs,
                outputs=main_outputs
            )
        elif isinstance(input_component, gr.Number):
            input_component.change(
                fn=generate_image,
                inputs=main_inputs,
                outputs=main_outputs
            )

    # --- Markdown Instructions ---
    gr.Markdown("""
    ### How to use:
    1.  **Generate/Load Seeds:** Use sliders, the random button, or load saved seeds.
    2.  **Save Seeds:** Enter a descriptive label in 'Style Description' and click 'Save to Catalog'.
    3.  **Mix:** Adjust 'Structure Mix' and 'Appearance Mix' sliders.
    4.  **Attributes:** Use the sliders below to modify specific features.
    5.  **Results Update:** The design updates automatically when you interact with controls.

    ### Tips for Accuracy:
    *   **Attribute Layers:** Attributes apply to layers `Start Layer` to `End Layer`. Change `End Layer` to target effects (e.g., 4-7 structure, 8-15 details).
    *   **Small Adjustments:** Use small slider changes first. Range is now -10 to 10.
    *   **Direction Mapping:** Accuracy depends *critically* on the `latent_dirs` mapping (Index 0=Sleeve, 1=Size, etc. is an *assumption*).
    """)

# --- Launch App ---
# Clear cache and saved seeds before launching
image_cache.clear()
saved_seeds.clear()
demo.launch(share=True) 