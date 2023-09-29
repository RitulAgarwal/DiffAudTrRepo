import torch
import torch.nn as nn 

class SimpleDiffusion:
    """
    initialises the basic alpha,beta noise schedule (0<beta1<beta2<----betan<1),
    all preprocessing of underroot 1-alpha and all that's required before forward and reverse diffusion process takes place
    """
    def __init__(self,
                 num_diffusion_timesteps:int=1000,
                 device:str="cpu"):
        
        self.num_diffusion_timesteps = num_diffusion_timesteps
    
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
        
        self_sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
         
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
        )

class Diffusion(nn.Module) :
    def __init__(self,timesteps:int = 1000) :
        super().__init__()
        self.timesteps = timesteps
        self.simple_diffusion = SimpleDiffusion(self.timesteps)

    def get(self,element: torch.Tensor, t: torch.Tensor):
        """
        Get value at index position "t" in "element" and
            reshape it to have the same dimension as a batch of images.
        """
        ele = element.gather(-1, t)#extract the value from the input tensor along with the specified dimension that we want.
        return ele.reshape(-1, 1, 1)
        
    def calcs(self,sample):
        print(sample)
        random_noise = torch.randn(sample.shape)
        print(random_noise,'starign ')
        num_images = sample.shape[0]
        
        for time_step in reversed(range(self.timesteps)):

            ts = torch.ones(num_images, dtype=torch.long) * time_step
            print(ts,'ts')
            
            z = torch.randn_like(random_noise) if time_step > 1 else torch.zeros_like(random_noise)
            print(z,'z')
        
            beta_t = self.get(self.simple_diffusion.beta, ts) # 4,1,1,1
            print(beta_t,beta_t.shape,'beatT')
            # print(self.simple_diffusion.beta.shape,'origianl')
            # print(beta_t.shape,'new latest ')
            
            print(self.simple_diffusion.one_by_sqrt_alpha,self.simple_diffusion.one_by_sqrt_alpha.shape,'simple 1 by sqrt alpah')         
            
            one_by_sqrt_alpha_t = self.get(self.simple_diffusion.one_by_sqrt_alpha, ts) # 4,1,1,1
            
            print(one_by_sqrt_alpha_t,one_by_sqrt_alpha_t.shape,'1/sqalphaT')

            sqrt_one_minus_alpha_cumulative_t = self.get(self.simple_diffusion.sqrt_one_minus_alpha_cumulative, ts)
            print(sqrt_one_minus_alpha_cumulative_t,sqrt_one_minus_alpha_cumulative_t.shape,'sqrt 1-alpha')
            
            a = beta_t / sqrt_one_minus_alpha_cumulative_t #([4, 1, 1, 1])
            print(a,a.shape,'a')
        
            random_noise = (
                one_by_sqrt_alpha_t
                * (random_noise - a * sample)
                + torch.sqrt(beta_t) * z
            )
            print(random_noise,random_noise.shape,'ranodm noise')
            
        return random_noise
