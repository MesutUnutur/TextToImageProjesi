import base64
import io

from flask import Flask, request, render_template
from PIL import Image
import torch as th

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

def load_model():
    has_cuda = th.cuda.is_available()
    global device
    device = th.device('cpu' if not has_cuda else 'cuda')
    print("device: ", device)
    
    # Create base model.
    global options
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    global model
    global diffusion
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    # Create CLIP model.
    global clip_model
    clip_model = create_clip_model(device=device)
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))

def save_image(batch: th.Tensor):
        scaled= ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped =scaled.permute(2,0,3,1).reshape([batch.shape[2], -1, 3 ])
        imgg = Image.fromarray(reshaped.numpy())
        path = "static/img/"+str(prompt)+".png"
        imgg.save(path)



app = Flask(__name__)

@app.route("/",  methods=['GET', 'POST'])
def predict():
    if request.method == "POST":

        global prompt
        prompt = request.form.get("prompt")
        batch_size = 1
        guidance_scale = 3.0

        # Tune this parameter to control the sharpness of 256x256 images.
        # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
        upsample_temp = 0.997

        ##############################
        # Sample from the base model #
        ##############################

        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor([tokens] * batch_size, device=device),
            mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
        )

        # Setup guidance function for CLIP model.
        cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)

        # Sample from the base model.
        model.del_cache()
        global samples
        samples = diffusion.p_sample_loop(
            model,
            (batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )
        model.del_cache()
           
        save_image(samples)
       
        im = Image.open("static/img/"+prompt+".png")
        data = io.BytesIO()
        im.save(data, "PNG")
        encoded_img_data = base64.b64encode(data.getvalue())

        # Show the output
        return render_template("form.html", img_data=encoded_img_data.decode('utf-8'))
    else:
        im = Image.open("static/img/sunflower.png")
        data = io.BytesIO()
        im.save(data, "PNG")
        encoded_img_data = base64.b64encode(data.getvalue())
        print("get metodu çalıştı")

        # Show the output
        return render_template("form.html", img_data=encoded_img_data.decode('utf-8'))
      
if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
        "\nplease wait until server has fully started"))
    load_model()
    app.run(debug=True)