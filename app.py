import os
import gradio as gr
from PIL import Image
import numpy as np

# Model loading is optional - app will work without it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionImg2ImgPipeline = None  # type: ignore


# =========================
# DEVICE
# =========================

device = "cpu"
if TORCH_AVAILABLE and torch is not None:
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore
    except:
        device = "cpu"

# =========================
# LOAD MODEL (Optional)
# =========================

pipe = None
USE_AI_MODEL = False

if TORCH_AVAILABLE and DIFFUSERS_AVAILABLE and torch is not None and StableDiffusionImg2ImgPipeline is not None:
    try:
        print("Loading AI model...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float32  # type: ignore
        )
        pipe = pipe.to(device)
        USE_AI_MODEL = True
        print("✓ AI model loaded successfully!")
    except Exception as e:
        print(f"⚠ Could not load AI model: {e}")
        print("Will use image blending instead")
        pipe = None

# =========================
# FOLDERS
# =========================

POKEMON_FOLDER = "pokemon"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================
# FUSION FUNCTION
# =========================

def fuse_pokemon(id1, id2):
    """Fuse two Pokemon images. Uses AI if available, otherwise uses image blending."""

    try:
        id1 = int(id1)
        id2 = int(id2)

        path1 = os.path.join(POKEMON_FOLDER, f"{id1}.png")
        path2 = os.path.join(POKEMON_FOLDER, f"{id2}.png")

        if not os.path.exists(path1) or not os.path.exists(path2):
            return None, f"❌ Pokemon files not found. (IDs: {id1}, {id2})"

        img1 = Image.open(path1).convert("RGB").resize((512, 512))
        img2 = Image.open(path2).convert("RGB").resize((512, 512))

        name1 = f"Pokemon{id1}"
        name2 = f"Pokemon{id2}"
        fusion_name = name1[:len(name1) // 2] + name2[len(name2) // 2:]

        # Try AI-powered fusion first
        if USE_AI_MODEL and pipe is not None:
            try:
                print(f"🎨 Fusing {name1} + {name2} with AI...")
                blended = Image.blend(img1, img2, alpha=0.5)

                prompt = f"""
                A creative fusion between {name1} and {name2},
                blending their best features,
                pokemon anime style,
                vibrant and colorful,
                cute and powerful,
                4k quality,
                professional pokemon artwork,
                game freak character design
                """

                result = pipe(  # type: ignore
                    prompt=prompt,
                    image=blended,
                    strength=0.8,
                    guidance_scale=7.5
                ).images[0]

                print("✓ AI fusion complete!")
            except Exception as e:
                print(f"⚠ AI fusion failed: {e}, using image blending instead")
                result = Image.blend(img1, img2, alpha=0.5)
        else:
            # Fallback: Image blending
            print(f"🖼 Blending {name1} + {name2}...")

            # Convert to numpy arrays
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)

            # Blend with weights
            blended_arr = (arr1 * 0.5 + arr2 * 0.5).astype(np.uint8)

            # Create result
            result = Image.fromarray(blended_arr)
            print("✓ Image blending complete!")

        # Save result
        save_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name1}_{name2}.png"
        )
        result.save(save_path)
        print(f"💾 Saved to: {save_path}")

        return result, f"✨ {fusion_name}"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return None, error_msg


# =========================
# UI
# =========================

interface = gr.Interface(
    fn=fuse_pokemon,
    inputs=[
        gr.Number(label="Pokemon ID 1", value=25),
        gr.Number(label="Pokemon ID 2", value=6)
    ],
    outputs=[
        gr.Image(label="Fusion Pokemon"),
        gr.Text(label="Fusion Name")
    ],
    title="🔥 Pokemon Fusion AI",
    description="""
    Enter two Pokemon IDs to fuse them!
    
    **Examples:**
    - 25 = Pikachu
    - 6 = Charizard
    - 1 = Bulbasaur
    - 4 = Charmander
    
    ℹ️ Uses image blending if AI model is not available.
    """,
    examples=[
        [25, 6],
        [1, 4],
        [25, 25]
    ]
)

print("\n" + "="*50)
print("🎮 Pokemon Fusion App is Starting...")
print("="*50)
if USE_AI_MODEL:
    print("✓ AI Model Status: LOADED")
else:
    print("ℹ️ AI Model Status: Using Image Blending")
print("="*50 + "\n")

interface.launch(
    share=False,
    server_name="127.0.0.1",
    server_port=7862,
    show_error=True,
    quiet=False
)