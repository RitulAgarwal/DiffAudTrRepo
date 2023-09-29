# from diffusers import UNet2DModel#, DDIMScheduler, VQModel
# import torch
# # import PIL.Image
# # import numpy as np
# # import tqdm

# seed = 3

# # # load all models
# unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")

# # print(unet)
# from diffusers import DiffusionPipeline

# model_id = "CompVis/ldm-celebahq-256"
# pipeline = DiffusionPipeline.from_pretrained(model_id)
# print(unet.config.in_channels, unet.sample_size, unet.sample_size)
# generator = torch.manual_seed(seed)
# noise = torch.randn(
#     (64,64),
#     generator=generator,
# )
# # noise.save("/home/earth/RITUL_NSUT/Architectures/noise.png")



# #.to(torch_device)
# # run pipeline in inference (sample random noise and denoise)
# image = pipeline(num_inference_steps=200)[noise]

# # save image
# image[0].save("/home/earth/RITUL_NSUT/Architectures/ldm_generated_image.png")


# from diffusers import UNet2DModel, DDIMScheduler, VQModel
# import torch
# import PIL.Image
# import numpy as np
# import tqdm

# seed = 3

# # load all models
# unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
# vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
# scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

# # set to cuda
# torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# unet.to(torch_device)
# vqvae.to(torch_device)

# # generate gaussian noise to be decoded
# generator = torch.manual_seed(seed)
# noise = torch.randn(
#     (1, unet.in_channels, unet.sample_size, unet.sample_size),
#     generator=generator,
# ).to(torch_device)

# # set inference steps for DDIM
# scheduler.set_timesteps(num_inference_steps=200)

# image = noise
# for t in tqdm.tqdm(scheduler.timesteps):
#     # predict noise residual of previous image
#     with torch.no_grad():
#         residual = unet(image, t)["sample"]

#     # compute previous image x_t according to DDIM formula
#     prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

#     # x_t-1 -> x_t
#     image = prev_image

# # decode image with vae
# with torch.no_grad():
#     image = vqvae.decode(image)
# # print(input.__init__)
# # process image
# image_processed = torch.permute(image,(0, 2, 3, 1))
# image_processed = (image_processed + 1.0) * 127.5
# image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
# image_pil = PIL.Image.fromarray(image_processed[0])

# image_pil.save(f"generated_image_{seed}.png")


from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
image = ddpm().images[0]

# save image
image.save("ddpm_generated_image.png")
